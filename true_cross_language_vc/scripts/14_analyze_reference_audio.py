"""
Script 14: Reference Audio Quality Analysis and Deterministic Segment Selection.
Analyzes full source audio files for all 12 pilot speakers, extracts acoustic properties
(duration, RMS, clipping, silence ratio, SNR proxy, voiced fraction), and implements
an automatic, deterministic 10-second clean voiced segment selection rule.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

PROJECT_ROOT = Path(r"C:\pd-speech-crosslingual")
WORKSPACE_ROOT = PROJECT_ROOT / "true_cross_language_vc"
REF_DIR = WORKSPACE_ROOT / "19_pilot_refinement" / "01_reference_audio_analysis"
CLEAN_REF_DIR = REF_DIR / "clean_references"

def compute_audio_properties(y, sr):
    duration = len(y) / sr
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    clipping = bool(peak >= 0.999)

    # Frame energy for silence and SNR proxy
    hop_length = int(sr * 0.02)  # 20ms
    frame_length = int(sr * 0.04) # 40ms
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))

    silence_thresh = max(1e-5, peak * 0.05) # -26 dB relative to peak
    speech_frames = frame_rms >= silence_thresh
    silence_ratio = float(np.mean(~speech_frames))

    # SNR proxy: active speech RMS vs. lowest 10% energy frames (noise floor)
    if np.any(speech_frames):
        active_rms = np.sqrt(np.mean(frame_rms[speech_frames] ** 2))
    else:
        active_rms = rms
    noise_floor = np.percentile(frame_rms, 10)
    snr_proxy = float(20 * np.log10(max(1e-5, active_rms) / max(1e-5, noise_floor)))

    # Voiced fraction estimation via zero crossing rate and energy
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    n_f = min(len(speech_frames), len(zcr))
    voiced_mask = speech_frames[:n_f] & (zcr[:n_f] < 0.25)
    voiced_fraction = float(np.mean(voiced_mask))

    return {
        "duration": round(duration, 4),
        "peak": round(peak, 4),
        "rms": round(rms, 4),
        "clipping": clipping,
        "silence_ratio": round(silence_ratio, 4),
        "snr_proxy_db": round(snr_proxy, 2),
        "voiced_fraction": round(voiced_fraction, 4)
    }

def select_clean_segment(y, sr, target_sec=10.0, step_sec=0.5):
    """
    Deterministically selects a clean speech segment of `target_sec` duration.
    Evaluates sliding windows and picks the window maximizing voiced speech
    while minimizing silence and avoiding edge transients.
    """
    total_sec = len(y) / sr
    if total_sec <= target_sec:
        # File is shorter than target; return trimmed speech
        trimmed_y, _ = librosa.effects.trim(y, top_db=30)
        return trimmed_y, 0.0, len(trimmed_y) / sr

    target_samples = int(target_sec * sr)
    step_samples = int(step_sec * sr)

    best_score = -1.0
    best_start = 0

    num_windows = (len(y) - target_samples) // step_samples + 1
    for w in range(num_windows):
        start = w * step_samples
        end = start + target_samples
        seg = y[start:end]

        props = compute_audio_properties(seg, sr)
        # Scoring function: penalize clipping, reward voiced fraction and healthy RMS, penalize silence
        if props["clipping"]:
            score = 0.0
        else:
            rms_factor = min(1.0, props["rms"] / 0.05)
            score = props["voiced_fraction"] * (1.0 - props["silence_ratio"]) * rms_factor

        if score > best_score:
            best_score = score
            best_start = start

    best_seg = y[best_start:best_start + target_samples].copy()
    # Apply 25ms cosine fade in/out to eliminate boundary clicks
    fade_len = int(0.025 * sr)
    fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, fade_len)))
    fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
    best_seg[:fade_len] *= fade_in
    best_seg[-fade_len:] *= fade_out

    return best_seg, best_start / sr, (best_start + target_samples) / sr

def main():
    REF_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_REF_DIR.mkdir(parents=True, exist_ok=True)

    pilot_csv = WORKSPACE_ROOT / "04_pilot_selection" / "pilot_12_selection.csv"
    df_pilot = pd.read_csv(pilot_csv)

    analysis_records = []

    print("=== Step 1: Analyzing Reference Audio Files & Selecting Clean Segments ===")
    for _, r in df_pilot.iterrows():
        sid = r["safe_id"]
        lang = r["language"]
        diag = r["diagnosis"]
        src_path = Path(r["audio_path"])

        y, sr = sf.read(str(src_path))
        if y.ndim > 1:
            y = y[:, 0]

        # 1. Full original audio properties
        full_props = compute_audio_properties(y, sr)

        # 2. Select clean 10.0s segment
        clean_seg, start_s, end_s = select_clean_segment(y, sr, target_sec=10.0)
        clean_props = compute_audio_properties(clean_seg, sr)

        # Save clean segment
        clean_filename = f"{sid.replace('.wav', '')}_clean_10s_ref.wav"
        clean_path = CLEAN_REF_DIR / clean_filename
        sf.write(str(clean_path), clean_seg, sr)

        analysis_records.append({
            "safe_id": sid,
            "language": lang,
            "diagnosis": diag,
            "full_duration": full_props["duration"],
            "full_rms": full_props["rms"],
            "full_silence_ratio": full_props["silence_ratio"],
            "full_snr_db": full_props["snr_proxy_db"],
            "full_voiced_fraction": full_props["voiced_fraction"],
            "full_clipping": full_props["clipping"],
            "clean_start_sec": round(start_s, 2),
            "clean_end_sec": round(end_s, 2),
            "clean_duration": clean_props["duration"],
            "clean_rms": clean_props["rms"],
            "clean_silence_ratio": clean_props["silence_ratio"],
            "clean_snr_db": clean_props["snr_proxy_db"],
            "clean_voiced_fraction": clean_props["voiced_fraction"],
            "clean_reference_path": str(clean_path)
        })
        print(f"  Processed {sid} ({lang} {diag}): full={full_props['duration']}s (silence={full_props['silence_ratio']*100:.1f}%) -> clean 10s [{start_s:.1f}s - {end_s:.1f}s] (silence={clean_props['silence_ratio']*100:.1f}%, voiced={clean_props['voiced_fraction']*100:.1f}%)")

    df_res = pd.DataFrame(analysis_records)
    out_csv = REF_DIR / "reference_audio_analysis.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved reference audio analysis to {out_csv}.")

    # Generate reference selection method documentation
    method_md = REF_DIR / "reference_selection_method.md"
    method_content = f"""# Deterministic Speaker-Reference Selection Method

## 1. Problem Definition & Motivation
In the initial 12-file pilot, full unsegmented source recordings were supplied directly to the XTTS-v2 speaker encoder:
- Spanish source recordings ranged from 15.1s to 26.1s.
- Especially in Parkinson's disease subjects (e.g. `SP_PD_002` at 26.1s and `SP_PD_003` at 25.1s), the recordings contain prolonged pauses, hesitation breaks, and variable mic-distance effects.
- Analysis revealed that `SP_PD_002` had a full-audio silence ratio of **{df_res[df_res['safe_id']=='SP_PD_002.wav']['full_silence_ratio'].iloc[0]*100:.1f}%**, which destabilized the autoregressive synthesis attention and caused decoder stalling (only 5 words generated).

## 2. Deterministic Selection Rule
To prevent manual cherry-picking and ensure strict scientific objectivity, an automatic deterministic window selection algorithm was implemented:

1. **Target Duration Window**: Exactly $T = 10.0$ seconds.
2. **Window Scanning**: An 10.0-second rectangular window slides across the raw audio with step $\\Delta t = 0.5$ seconds.
3. **Objective Window Score**:
   $$\\text{{Score}}(w) = \\text{{VoicedFraction}}(w) \\times (1.0 - \\text{{SilenceRatio}}(w)) \\times \\min\\left(1.0, \\frac{{\\text{{RMS}}(w)}}{{0.05}}\\right)$$
   - Disqualify any window exhibiting digital clipping ($\ge 0.999$).
   - Maximize continuous phonated speech while minimizing unvoiced hesitations and room noise.
4. **Edge Smoothing**: Apply a 25-millisecond raised cosine fade-in and fade-out to the boundary samples of the selected segment to prevent spectral click artifacts.
5. **Exact Preservation**: The selected 10-second reference audio comes directly from the original recording without any pitch modification, frequency alteration, or acoustic morphing.

## 3. Comparative Summary: Full Audio vs. Clean Reference Segment

| Safe ID | Lang | Diagnosis | Full Duration (s) | Full Silence % | Full Voiced % | Clean Window (s) | Clean Silence % | Clean Voiced % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, r in df_res.iterrows():
        method_content += (
            f"| `{r['safe_id']}` | {r['language']} | **{r['diagnosis']}** | {r['full_duration']:.1f} | "
            f"{r['full_silence_ratio']*100:.1f}% | {r['full_voiced_fraction']*100:.1f}% | "
            f"[{r['clean_start_sec']:.1f}s – {r['clean_end_sec']:.1f}s] | "
            f"**{r['clean_silence_ratio']*100:.1f}%** | **{r['clean_voiced_fraction']*100:.1f}%** |\n"
        )

    method_content += r"""
## 4. Availability of Clean References
All 12 extracted clean reference audio segments have been generated and saved under:
`19_pilot_refinement/01_reference_audio_analysis/clean_references/`
"""
    with open(method_md, "w", encoding="utf-8") as f:
        f.write(method_content)
    print(f"Saved reference selection method documentation to {method_md}.")

if __name__ == "__main__":
    main()
