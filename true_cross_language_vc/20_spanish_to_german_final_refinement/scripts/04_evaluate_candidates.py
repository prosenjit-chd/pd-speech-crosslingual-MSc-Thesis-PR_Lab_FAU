#!/usr/bin/env python3
"""
04_evaluate_candidates.py
Rigorous Automated Evaluation of XTTS-v2 Candidates on 4 Search Speakers.

Implements Sections 14, 15, 16, 17, 18:
- Audio QC (peak, RMS, clipping, NaN/Inf, silence ratio)
- Language ID (Whisper LID: predicted language, P(German))
- Content Fidelity (Whisper ASR: WER, CER, WIP against normalized German target text)
- Speaker Preservation (ECAPA-TDNN cosine similarity against original Spanish source audio)
- Hard Rejection Gates:
    1. German LID == 'de'
    2. P(German) >= 0.85
    3. Audio valid (no NaN/Inf)
    4. 0 clipping detected
    5. ECAPA speaker similarity >= 0.40
- Candidate Ranking:
    1. lowest mean WER
    2. lowest mean CER
    3. highest mean WIP
    4. highest speaker similarity
- Early Stop Check (Section 18):
    100% German LID AND mean WER <= 0.45 AND mean WIP >= 0.55 AND mean similarity >= 0.55 AND 0 clipping.
"""

import sys
import os
import re
import json
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from pathlib import Path
from jiwer import wer, cer, wip

WORKSPACE = Path(r"C:\pd-speech-crosslingual\true_cross_language_vc\20_spanish_to_german_final_refinement")
PROJECT_ROOT = Path(r"C:\pd-speech-crosslingual")
CANDIDATE_DIR = WORKSPACE / "07_candidate_audio"
EVAL_DIR = WORKSPACE / "08_candidate_evaluation"
SELECTION_DIR = WORKSPACE / "09_model_selection"
PILOT_CSV = Path(r"C:\pd-speech-crosslingual\true_cross_language_vc\04_pilot_selection\pilot_12_selection.csv")
REF_TEXT_FILE = WORKSPACE / "02_text_analysis" / "german_target_text_normalized.txt"

def normalize_text_for_eval(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\sÄÖÜäöüß]", " ", text)
    return " ".join(text.split())

def compute_audio_qc(y: np.ndarray, sr: int) -> dict:
    dur = len(y) / sr
    has_nan = bool(np.isnan(y).any())
    has_inf = bool(np.isinf(y).any())
    peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) > 0 else 0.0
    is_clipped = bool(peak >= 0.999)
    
    # Silence ratio
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    num_frames = max(1, (len(y) - frame_len) // hop_len + 1)
    frame_rms = []
    for i in range(num_frames):
        st = i * hop_len
        f = y[st:st + frame_len]
        frame_rms.append(np.sqrt(np.mean(f ** 2)))
    frame_rms = np.array(frame_rms)
    sil_thresh = max(0.005, 0.1 * rms)
    sil_ratio = float(np.sum(frame_rms < sil_thresh) / len(frame_rms)) if len(frame_rms) > 0 else 0.0
    
    return {
        "duration": dur,
        "sample_rate": sr,
        "peak": peak,
        "rms": rms,
        "is_clipped": is_clipped,
        "has_nan_inf": (has_nan or has_inf),
        "silence_ratio": sil_ratio
    }

def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load normalized reference German text
    norm_target_text = normalize_text_for_eval(REF_TEXT_FILE.read_text(encoding="utf-8"))
    ref_word_count = len(norm_target_text.split())
    print(f"Loaded normalized German reference text ({ref_word_count} words):")
    print(f"  '{norm_target_text}'\n")
    
    # Load pilot original metadata for ground truth speaker references
    df_pilot = pd.read_csv(PILOT_CSV)
    df_pilot_es = df_pilot[df_pilot["language"] == "Spanish"].copy()
    speaker_orig_paths = {
        row["safe_id"].replace(".wav", ""): Path(row["audio_path"])
        for _, row in df_pilot_es.iterrows()
    }
    
    # Candidate manifest
    manifest_csv = WORKSPACE / "04_xtts_parameter_search" / "candidate_synthesis_manifest.csv"
    if not manifest_csv.exists():
        print(f"Error: Manifest {manifest_csv} not found! Run 03_run_xtts_candidate_search.py first.")
        sys.exit(1)
    df_manifest = pd.read_csv(manifest_csv)
    
    print("=== Loading Evaluation Models (Whisper LID/ASR & ECAPA-TDNN) ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device for evaluation models: {device}")
    
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from speechbrain.inference.speaker import EncoderClassifier
    
    whisper_id = "openai/whisper-base"
    print(f"Loading Whisper model ({whisper_id})...")
    processor = WhisperProcessor.from_pretrained(whisper_id)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_id).to(device)
    
    print("Loading ECAPA-TDNN speaker encoder...")
    cache_dir = PROJECT_ROOT / "thesis_final_evaluation" / "models_cache" / "spkrec-ecapa-voxceleb"
    spk_classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"},  # ECAPA inference is very lightweight on CPU and avoids CUDA collisions
        savedir=str(cache_dir)
    )
    
    # Pre-encode original speaker references to avoid recomputing
    orig_embeddings = {}
    for spk, path in speaker_orig_paths.items():
        y_orig, sr_orig = sf.read(str(path))
        if y_orig.ndim > 1:
            y_orig = y_orig[:, 0]
        t_orig = torch.tensor(y_orig, dtype=torch.float32).unsqueeze(0)
        if sr_orig != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr_orig, new_freq=16000)
            t_orig = resampler(t_orig)
        with torch.no_grad():
            emb = spk_classifier.encode_batch(t_orig)[0, 0, :].numpy()
        orig_embeddings[spk] = emb / np.linalg.norm(emb)
    print("Original speaker embeddings cached successfully.\n")
    
    records = []
    
    print("=== Evaluating Candidate Outputs ===")
    for idx, row in df_manifest.iterrows():
        cand_id = row["candidate_id"]
        safe_id = row["safe_id"]
        out_path = Path(row["output_path"])
        
        if not out_path.exists():
            print(f"  [MISSING] {out_path}")
            continue
            
        y, sr = sf.read(str(out_path))
        if y.ndim > 1:
            y = y[:, 0]
            
        # 1. Audio QC
        qc = compute_audio_qc(y, sr)
        
        # Resample to 16kHz for Whisper and SpeechBrain
        t_sig = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        if sr != 16000:
            resampler_16k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            t_sig = resampler_16k(t_sig)
        y_16k = t_sig.squeeze(0).numpy()
        
        # 2. Whisper LID
        chunk_lid = y_16k[:16000 * 30]
        inputs_lid = processor(chunk_lid, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            dec_in = torch.tensor([[whisper_model.config.decoder_start_token_id]], device=device)
            out = whisper_model(input_features=inputs_lid.input_features, decoder_input_ids=dec_in)
            first_step_logits = out.logits[0, -1, :]
            
        lang_tokens = [tok for tok in processor.tokenizer.get_vocab().keys() if tok.startswith("<|") and len(tok) == 6 and tok[2:4].isalpha()]
        lang_token_ids = {tok[2:4]: processor.tokenizer.convert_tokens_to_ids(tok) for tok in lang_tokens}
        lang_logits = torch.tensor([first_step_logits[tid].item() for tid in lang_token_ids.values()])
        lang_probs = torch.softmax(lang_logits, dim=0)
        lang_codes = list(lang_token_ids.keys())
        best_idx = torch.argmax(lang_probs).item()
        pred_lang = lang_codes[best_idx]
        
        de_idx = lang_codes.index("de") if "de" in lang_codes else -1
        p_german = float(lang_probs[de_idx].item()) if de_idx >= 0 else 0.0
        is_german = (pred_lang == "de")
        
        # 3. Whisper ASR
        forced_ids = processor.get_decoder_prompt_ids(language="de", task="transcribe")
        with torch.no_grad():
            pred_ids = whisper_model.generate(inputs_lid.input_features, forced_decoder_ids=forced_ids)
        asr_transcript = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        
        if len(y_16k) > 16000 * 30:
            chunk_2 = y_16k[16000 * 30:16000 * 60]
            inputs_2 = processor(chunk_2, sampling_rate=16000, return_tensors="pt").to(device)
            with torch.no_grad():
                pred_ids_2 = whisper_model.generate(inputs_2.input_features, forced_decoder_ids=forced_ids)
            transcript_2 = processor.batch_decode(pred_ids_2, skip_special_tokens=True)[0]
            asr_transcript = asr_transcript + " " + transcript_2
            
        norm_hyp = normalize_text_for_eval(asr_transcript)
        
        if len(norm_hyp) > 0:
            calc_wer = float(wer(norm_target_text, norm_hyp))
            calc_cer = float(cer(norm_target_text, norm_hyp))
            calc_wip = float(wip(norm_target_text, norm_hyp))
        else:
            calc_wer, calc_cer, calc_wip = 1.0, 1.0, 0.0
            
        # 4. ECAPA Speaker Similarity
        with torch.no_grad():
            emb_synth = spk_classifier.encode_batch(t_sig)[0, 0, :].numpy()
        emb_synth = emb_synth / np.linalg.norm(emb_synth)
        spk_sim = float(np.dot(orig_embeddings[safe_id], emb_synth))
        
        # 5. Hard Rejection Checks (Section 15)
        # German LID incorrect OR P(German) < 0.85 OR audio invalid OR clipping detected OR speaker cosine similarity < 0.40
        rejection_reasons = []
        if not is_german:
            rejection_reasons.append(f"LID was '{pred_lang}', not 'de'")
        if p_german < 0.85:
            rejection_reasons.append(f"P(German)={p_german:.3f} < 0.85")
        if qc["has_nan_inf"]:
            rejection_reasons.append("Audio contains NaN or Inf")
        if qc["is_clipped"]:
            rejection_reasons.append(f"Audio clipped (peak={qc['peak']:.3f})")
        if spk_sim < 0.40:
            rejection_reasons.append(f"Speaker similarity {spk_sim:.3f} < 0.40")
        if qc["duration"] > 65.0:
            rejection_reasons.append(f"Severe repetition loop (duration={qc['duration']:.1f}s)")
        if len(norm_hyp.split()) < (ref_word_count * 0.4):
            rejection_reasons.append(f"Major text dropout ({len(norm_hyp.split())}/{ref_word_count} words)")
            
        passed_gates = (len(rejection_reasons) == 0)
        
        records.append({
            "candidate_id": cand_id,
            "safe_id": safe_id,
            "duration": round(qc["duration"], 2),
            "peak": round(qc["peak"], 3),
            "rms": round(qc["rms"], 4),
            "is_clipped": qc["is_clipped"],
            "silence_ratio": round(qc["silence_ratio"], 3),
            "predicted_language": pred_lang,
            "p_german": round(p_german, 4),
            "is_german": is_german,
            "asr_transcript": asr_transcript,
            "wer": round(calc_wer, 4),
            "cer": round(calc_cer, 4),
            "wip": round(calc_wip, 4),
            "speaker_similarity": round(spk_sim, 4),
            "passed_hard_gates": passed_gates,
            "rejection_reasons": "; ".join(rejection_reasons) if rejection_reasons else "NONE"
        })
        
        print(f"[{cand_id}][{safe_id}] LID={pred_lang} ({p_german:.3f}) | WER={calc_wer:.3f} | WIP={calc_wip:.3f} | Sim={spk_sim:.3f} | Gates: {'PASS' if passed_gates else 'FAIL'}")
        
    df_eval = pd.DataFrame(records)
    matrix_csv = EVAL_DIR / "candidate_matrix.csv"
    df_eval.to_csv(matrix_csv, index=False)
    print(f"\nFull candidate evaluation matrix saved to: {matrix_csv}")
    
    # Aggregate by Candidate
    summary_records = []
    for cand_id, group in df_eval.groupby("candidate_id"):
        n_total = len(group)
        n_pass = int(group["passed_hard_gates"].sum())
        all_passed = (n_pass == n_total)
        
        mean_wer = float(group["wer"].mean())
        median_wer = float(group["wer"].median())
        mean_cer = float(group["cer"].mean())
        mean_wip = float(group["wip"].mean())
        mean_sim = float(group["speaker_similarity"].mean())
        mean_p_de = float(group["p_german"].mean())
        german_acc = float(group["is_german"].mean())
        clipped_count = int(group["is_clipped"].sum())
        
        # Check early stop rule (Section 18):
        # 100% German LID AND mean WER <= 0.45 AND mean WIP >= 0.55 AND mean speaker similarity >= 0.55 AND 0 clipping
        meets_early_stop = (
            german_acc == 1.0 and
            mean_wer <= 0.45 and
            mean_wip >= 0.55 and
            mean_sim >= 0.55 and
            clipped_count == 0 and
            all_passed
        )
        
        summary_records.append({
            "candidate_id": cand_id,
            "passed_hard_gates": all_passed,
            "files_passed_gates": f"{n_pass}/{n_total}",
            "german_lid_accuracy": german_acc,
            "mean_p_german": round(mean_p_de, 4),
            "mean_wer": round(mean_wer, 4),
            "median_wer": round(median_wer, 4),
            "mean_cer": round(mean_cer, 4),
            "mean_wip": round(mean_wip, 4),
            "mean_speaker_similarity": round(mean_sim, 4),
            "clipped_files": clipped_count,
            "meets_early_stop": meets_early_stop
        })
        
    df_summary = pd.DataFrame(summary_records)
    
    # Sort candidates by:
    # 1. passed_hard_gates (True first)
    # 2. lowest mean_wer
    # 3. lowest mean_cer
    # 4. highest mean_wip
    # 5. highest speaker similarity
    df_summary = df_summary.sort_values(
        by=["passed_hard_gates", "mean_wer", "mean_cer", "mean_wip", "mean_speaker_similarity"],
        ascending=[False, True, True, False, False]
    ).reset_index(drop=True)
    
    summary_csv = EVAL_DIR / "candidate_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print("\n=== Candidate Search Summary Ranking ===")
    print(df_summary.to_string())
    
    # Best candidate selection
    best_candidate = df_summary.iloc[0].to_dict()
    print(f"\nWINNING CANDIDATE: {best_candidate['candidate_id']}")
    print(f"  Mean WER: {best_candidate['mean_wer']:.4f} | WIP: {best_candidate['mean_wip']:.4f} | Sim: {best_candidate['mean_speaker_similarity']:.4f}")
    print(f"  Passed hard gates: {best_candidate['passed_hard_gates']}")
    print(f"  Meets early-stop threshold: {best_candidate['meets_early_stop']}")
    
    # Save selection decision
    with open(SELECTION_DIR / "selection_decision.json", "w", encoding="utf-8") as f:
        json.dump(best_candidate, f, indent=2)
        
    # Write summary markdown report
    md_lines = [
        "# XTTS-v2 Parameter Search & Candidate Evaluation Report",
        "",
        "## 1. Candidate Matrix Summary",
        "",
        df_summary.to_markdown(index=False),
        "",
        "## 2. Winning Configuration Analysis",
        f"- **Selected Candidate:** `{best_candidate['candidate_id']}`",
        f"- **Mean WER:** `{best_candidate['mean_wer']:.4f}` (Baseline Config C on subset: ~0.81)",
        f"- **Mean WIP:** `{best_candidate['mean_wip']:.4f}`",
        f"- **Mean Speaker Similarity:** `{best_candidate['mean_speaker_similarity']:.4f}`",
        f"- **German LID Accuracy:** `{best_candidate['german_lid_accuracy']*100:.1f}%` (Mean P(de) = `{best_candidate['mean_p_german']:.4f}`)",
        f"- **Clipped Files:** `{best_candidate['clipped_files']}`",
        f"- **Meets Early-Stop Rule:** `{best_candidate['meets_early_stop']}`",
        "",
        "## 3. Decision for Secondary Model Testing (Section 19)",
    ]
    if best_candidate["mean_wer"] > 0.60:
        md_lines.append(f"> [!WARNING]\n> XTTS-v2 best mean WER ({best_candidate['mean_wer']:.4f}) remains > 0.60. Testing secondary multilingual model per Section 19.")
    else:
        md_lines.append(f"> [!NOTE]\n> XTTS-v2 best mean WER ({best_candidate['mean_wer']:.4f}) successfully broke through <= 0.60 threshold. Proceeding to freeze configuration and regenerate full 6-file cohort.")
    
    (EVAL_DIR / "candidate_evaluation_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nReport written to: {EVAL_DIR / 'candidate_evaluation_report.md'}")

if __name__ == "__main__":
    main()
