"""
Script 04: Master Audio QC, Target-Language ID, Whisper Content Accuracy, and ECAPA-TDNN Speaker Preservation for 276 Cohort.

Implements Sections 13, 14, 15, 17:
- Section 13: Full audio QC (07_audio_qc/audio_qc_276.csv, audio_qc_report.md)
- Section 14: Whisper LID evaluation (08_language_id/language_id_276.csv, language_id_report.md)
- Section 15: Whisper ASR & content fidelity (WER, CER, WIP) (09_content_accuracy/content_accuracy_276.csv, asr_transcriptions_276.csv, content_accuracy_report.md)
- Section 17: Pretrained ECAPA-TDNN speaker biometric preservation & negative controls (10_speaker_preservation/speaker_similarity_276.csv, different_speaker_controls.csv, speaker_report.md)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer as calc_wer, cer as calc_cer, wip as calc_wip
import speechbrain
from speechbrain.inference.speaker import EncoderClassifier

PROJECT_ROOT = Path(r"C:\pd-speech-crosslingual")
WORKSPACE = PROJECT_ROOT / "true_cross_language_vc"
EVAL_DIR = WORKSPACE / "22_full_276_evaluation"

INV_CSV = EVAL_DIR / "01_inventory" / "full_276_inventory.csv"
TAGS_CSV = EVAL_DIR / "02_cohort_tags" / "cohort_provenance.csv"
TARGET_CSV = EVAL_DIR / "04_translation" / "frozen_target_translations.csv"
SYNTH_DIR = EVAL_DIR / "06_synthesis"

QC_DIR = EVAL_DIR / "07_audio_qc"
LID_DIR = EVAL_DIR / "08_language_id"
CONTENT_DIR = EVAL_DIR / "09_content_accuracy"
SPK_DIR = EVAL_DIR / "10_speaker_preservation"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_audio_qc(audio_path):
    try:
        data, sr = sf.read(str(audio_path))
        dur = len(data) / sr
        channels = 1 if data.ndim == 1 else data.shape[1]
        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(data**2)))
        has_nan = bool(np.isnan(data).any())
        has_inf = bool(np.isinf(data).any())
        is_clipped = bool(peak >= 0.999)
        is_zero = bool(peak == 0.0)

        # Silence ratio
        frame_len = int(0.025 * sr)
        hop_len = int(0.010 * sr)
        num_frames = max(1, (len(data) - frame_len) // hop_len + 1)
        frame_rms = [np.sqrt(np.mean(data[i*hop_len : i*hop_len + frame_len]**2)) for i in range(num_frames)]
        frame_rms = np.array(frame_rms)
        silence_thresh = max(0.005, 0.1 * rms)
        silence_ratio = float(np.mean(frame_rms < silence_thresh)) if len(frame_rms) > 0 else 0.0

        abnormal_dur = bool(dur < 5.0 or dur > 120.0)
        qc_pass = (not has_nan) and (not has_inf) and (not is_zero) and (not is_clipped) and (not abnormal_dur)

        return {
            "duration": round(dur, 4),
            "sample_rate": sr,
            "channels": channels,
            "peak_amplitude": round(peak, 4),
            "rms_energy": round(rms, 4),
            "has_nan": has_nan,
            "has_inf": has_inf,
            "is_clipped": is_clipped,
            "silence_ratio": round(silence_ratio, 4),
            "abnormal_duration": abnormal_dur,
            "qc_status": "PASSED" if qc_pass else "FAILED"
        }
    except Exception as e:
        return {
            "duration": 0.0,
            "sample_rate": 0,
            "channels": 0,
            "peak_amplitude": 0.0,
            "rms_energy": 0.0,
            "has_nan": True,
            "has_inf": True,
            "is_clipped": False,
            "silence_ratio": 1.0,
            "abnormal_duration": True,
            "qc_status": f"ERROR: {e}"
        }

def main():
    QC_DIR.mkdir(parents=True, exist_ok=True)
    LID_DIR.mkdir(parents=True, exist_ok=True)
    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    SPK_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading Cohort Metadata and Verifying Synthesis Outputs ===")
    df_inv = pd.read_csv(INV_CSV)
    df_tags = pd.read_csv(TAGS_CSV)
    df_tgt = pd.read_csv(TARGET_CSV)

    df_meta = df_inv.merge(df_tags[["safe_id", "cohort_tag", "in_set_b_dev_excluded_264", "in_set_c_val_80"]], on="safe_id")

    # Locate generated file paths
    gen_paths = []
    missing = []
    for _, r in df_meta.iterrows():
        s_lang = r["language"]
        t_fn = r["target_filename"]
        sub_dir = "spanish_to_german" if s_lang == "Spanish" else "german_to_spanish"
        p = SYNTH_DIR / sub_dir / t_fn
        if p.exists() and p.stat().st_size > 1000:
            gen_paths.append(str(p))
        else:
            missing.append(t_fn)
            gen_paths.append("")

    df_meta["generated_path"] = gen_paths
    assert len(missing) == 0, f"Cannot evaluate: {len(missing)} files missing from synthesis!"
    print(f"All {len(df_meta)} generated files verified on disk.")

    # ========================================================
    # STEP 1: AUDIO QC (Section 13)
    # ========================================================
    qc_csv = QC_DIR / "audio_qc_276.csv"
    if qc_csv.exists() and len(pd.read_csv(qc_csv)) == len(df_meta):
        print(f"Loaded existing audio QC data from {qc_csv}")
        df_qc = pd.read_csv(qc_csv)
    else:
        print("\n=== Running Audio QC on 276 Generated Recordings ===")
        qc_records = []
        for idx, r in df_meta.iterrows():
            sid = r["safe_id"]
            gp = r["generated_path"]
            qc = compute_audio_qc(gp)
            rec = {
                "safe_id": sid,
                "source_language": r["language"],
                "target_language": r["target_language"],
                "diagnosis": r["diagnosis"],
                "cohort_tag": r["cohort_tag"],
                "generated_filename": Path(gp).name,
                "generated_path": gp,
                **qc
            }
            qc_records.append(rec)
        df_qc = pd.DataFrame(qc_records)
        df_qc.to_csv(qc_csv, index=False)
        print(f"Saved audio QC data to {qc_csv}")

    pass_cnt = (df_qc["qc_status"] == "PASSED").sum()
    clip_cnt = df_qc["is_clipped"].sum()
    print(f"Audio QC Summary: {pass_cnt}/{len(df_qc)} PASSED ({pass_cnt/len(df_qc)*100:.1f}%), {clip_cnt} Clipped ({clip_cnt/len(df_qc)*100:.1f}%)")

    # Generate Audio QC Report
    qc_rep_path = QC_DIR / "audio_qc_report.md"
    qc_md = [
        "# Audio Quality Control (QC) Report — Full 276 Evaluation",
        "",
        "## Overall Cohort QC Summary",
        f"- **Total files evaluated**: {len(df_qc)}",
        f"- **Files passed QC gate**: {pass_cnt} ({pass_cnt/len(df_qc)*100:.2f}%)",
        f"- **Clipped files**: {clip_cnt} ({clip_cnt/len(df_qc)*100:.2f}%)",
        f"- **Mean Duration**: {df_qc['duration'].mean():.2f}s (Min: {df_qc['duration'].min():.2f}s, Max: {df_qc['duration'].max():.2f}s)",
        f"- **Mean Peak Amplitude**: {df_qc['peak_amplitude'].mean():.4f}",
        f"- **Mean RMS Energy**: {df_qc['rms_energy'].mean():.4f}",
        f"- **Mean Silence Ratio**: {df_qc['silence_ratio'].mean():.4f}",
        "",
        "## Stratified Audio QC Breakdown",
        "",
        "| Stratification | N | Passed (%) | Clipped (%) | Mean Duration (s) | Mean Peak | Mean RMS | Mean Silence Ratio |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for grp_name, sub in [
        ("Spanish -> German", df_qc[df_qc["source_language"] == "Spanish"]),
        ("German -> Spanish", df_qc[df_qc["source_language"] == "German"]),
        ("Healthy Controls (HC)", df_qc[df_qc["diagnosis"] == "HC"]),
        ("Parkinson's Disease (PD)", df_qc[df_qc["diagnosis"] == "PD"]),
        ("Full Cohort (Set A)", df_qc)
    ]:
        p_pct = (sub["qc_status"] == "PASSED").mean() * 100
        c_pct = sub["is_clipped"].mean() * 100
        dur = sub["duration"].mean()
        pk = sub["peak_amplitude"].mean()
        rms = sub["rms_energy"].mean()
        sil = sub["silence_ratio"].mean()
        qc_md.append(f"| {grp_name} | {len(sub)} | {p_pct:.1f}% | {c_pct:.1f}% | {dur:.2f} | {pk:.4f} | {rms:.4f} | {sil:.4f} |")

    qc_rep_path.write_text("\n".join(qc_md), encoding="utf-8")
    print(f"Saved Audio QC Report to {qc_rep_path}")

    # ========================================================
    # STEP 2 & 3: WHISPER LID AND ASR (Sections 14 & 15)
    # ========================================================
    lid_csv = LID_DIR / "language_id_276.csv"
    content_csv = CONTENT_DIR / "content_accuracy_276.csv"
    trans_csv = CONTENT_DIR / "asr_transcriptions_276.csv"

    if lid_csv.exists() and len(pd.read_csv(lid_csv)) == len(df_meta) and content_csv.exists() and len(pd.read_csv(content_csv)) == len(df_meta):
        print(f"Loaded existing LID data from {lid_csv}")
        df_lid = pd.read_csv(lid_csv)
        print(f"Loaded existing Content Accuracy data from {content_csv}")
        df_content = pd.read_csv(content_csv)
    else:
        print("\n=== Initializing Whisper for Target LID & ASR Content Accuracy ===")
        whisper_id = "openai/whisper-base"
        processor = WhisperProcessor.from_pretrained(whisper_id)
        model = WhisperForConditionalGeneration.from_pretrained(whisper_id).to(DEVICE)
        model.eval()

        # Whisper LID tokens
        de_token = processor.tokenizer.convert_tokens_to_ids("<|de|>")
        es_token = processor.tokenizer.convert_tokens_to_ids("<|es|>")
        lang_tokens = {k: v for k, v in processor.tokenizer.get_vocab().items() if k.startswith("<|") and len(k) == 6 and k.endswith("|>")}
        valid_lang_ids = list(lang_tokens.values())

        lid_records = []
        content_records = []
        asr_transcriptions = []

        print("Running Whisper LID and ASR transcription on 276 files...")
        for idx, r in df_meta.iterrows():
            sid = r["safe_id"]
            gp = r["generated_path"]
            s_lang = r["language"]
            t_lang = r["target_language"]
            diag = r["diagnosis"]
            tag = r["cohort_tag"]

            row_tgt = df_tgt[df_tgt["safe_id"] == sid].iloc[0]
            ref_text = row_tgt["target_text_full"]

            # Load audio at 16kHz
            data, sr = sf.read(gp)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                data_t = resampler(torch.tensor(data, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            else:
                data_t = data

            # Feature extraction
            input_features = processor(data_t, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)

            # 1. LID: Inspect logits of first generated token
            with torch.no_grad():
                decoder_input_ids = torch.tensor([[processor.tokenizer.prefix_tokens[0]]], device=DEVICE)
                logits = model(input_features, decoder_input_ids=decoder_input_ids).logits[:, -1, :]
                lang_logits = logits[0, valid_lang_ids]
                lang_probs = torch.softmax(lang_logits, dim=-1)
                pred_idx = torch.argmax(lang_probs).item()
                pred_token_id = valid_lang_ids[pred_idx]
                pred_lang_code = processor.tokenizer.decode([pred_token_id]).replace("<|", "").replace("|>", "")

                p_de = torch.softmax(logits[0, [de_token, es_token]], dim=-1)[0].item()
                p_es = 1.0 - p_de
                tgt_prob = p_de if t_lang == "German" else p_es

                expected_code = "de" if t_lang == "German" else "es"
                correct_tgt = (pred_lang_code == expected_code)

            lid_records.append({
                "safe_id": sid,
                "source_language": s_lang,
                "target_language": t_lang,
                "diagnosis": diag,
                "cohort_tag": tag,
                "predicted_language": pred_lang_code,
                "target_language_probability": round(tgt_prob, 4),
                "correct_target_language": correct_tgt,
                "p_german": round(p_de, 4),
                "p_spanish": round(p_es, 4)
            })

            # 2. ASR Transcription for Content Accuracy
            with torch.no_grad():
                forced_decoder_ids = processor.get_decoder_prompt_ids(language=expected_code, task="transcribe")
                predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids, max_length=448)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

            # Compute WER, CER, WIP
            import string
            def clean_txt(t):
                t = t.lower().translate(str.maketrans("", "", string.punctuation))
                return " ".join(t.split())

            ref_clean = clean_txt(ref_text)
            hyp_clean = clean_txt(transcription)

            val_wer = calc_wer(ref_clean, hyp_clean)
            val_cer = calc_cer(ref_clean, hyp_clean)
            val_wip = calc_wip(ref_clean, hyp_clean)

            asr_transcriptions.append({
                "safe_id": sid,
                "source_language": s_lang,
                "target_language": t_lang,
                "diagnosis": diag,
                "cohort_tag": tag,
                "reference_text": ref_text,
                "asr_transcription": transcription,
                "wer": round(val_wer, 4),
                "cer": round(val_cer, 4),
                "wip": round(val_wip, 4)
            })

            content_records.append({
                "safe_id": sid,
                "source_language": s_lang,
                "target_language": t_lang,
                "diagnosis": diag,
                "cohort_tag": tag,
                "wer": round(val_wer, 4),
                "cer": round(val_cer, 4),
                "wip": round(val_wip, 4)
            })

            if (idx + 1) % 50 == 0 or (idx + 1) == len(df_meta):
                print(f"  Processed {idx + 1}/{len(df_meta)} files for LID & ASR...")

        df_lid = pd.DataFrame(lid_records)
        df_lid.to_csv(lid_csv, index=False)
        df_content = pd.DataFrame(content_records)
        df_content.to_csv(content_csv, index=False)
        df_trans = pd.DataFrame(asr_transcriptions)
        df_trans.to_csv(trans_csv, index=False)

    # Generate LID Report
    lid_rep_path = LID_DIR / "language_id_report.md"
    dev_excl_lid = df_lid[df_lid["cohort_tag"] != "PILOT_DEVELOPMENT"]
    lid_md = [
        "# Target-Language Identification Report — Full 276 Evaluation",
        "",
        "## Executive Summary",
        f"- **Overall Target LID Accuracy**: {df_lid['correct_target_language'].mean()*100:.2f}% ({df_lid['correct_target_language'].sum()}/{len(df_lid)})",
        f"- **Overall Mean Target Probability**: {df_lid['target_language_probability'].mean():.4f}",
        "",
        "## Stratified Breakdown",
        "",
        "| Subgroup | N | Correct LID (%) | Mean Target Prob | Median Target Prob |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]
    for grp_name, sub in [
        ("Spanish -> German", df_lid[df_lid["source_language"] == "Spanish"]),
        ("German -> Spanish", df_lid[df_lid["source_language"] == "German"]),
        ("Healthy Controls (HC)", df_lid[df_lid["diagnosis"] == "HC"]),
        ("Parkinson's Disease (PD)", df_lid[df_lid["diagnosis"] == "PD"]),
        ("Full Cohort (Set A, N=276)", df_lid),
        ("Development-Excluded (Set B, N=264)", dev_excl_lid)
    ]:
        acc = sub["correct_target_language"].mean() * 100
        p_mean = sub["target_language_probability"].mean()
        p_med = sub["target_language_probability"].median()
        lid_md.append(f"| {grp_name} | {len(sub)} | {acc:.2f}% | {p_mean:.4f} | {p_med:.4f} |")

    lid_rep_path.write_text("\n".join(lid_md), encoding="utf-8")
    print(f"Saved Language ID Report to {lid_rep_path}")

    # Generate Content Accuracy Report
    content_rep_path = CONTENT_DIR / "content_accuracy_report.md"
    def get_stats_row(name, s):
        q25, q75 = np.percentile(s, [25, 75])
        p05, p95 = np.percentile(s, [5, 95])
        return f"| {name} | {len(s)} | {s.mean():.4f} | {s.median():.4f} | {s.std():.4f} | {q75-q25:.4f} | {p05:.4f} | {p95:.4f} |"

    content_md = [
        "# True Content Accuracy Report — Full 276 Evaluation",
        "",
        "## Metric Overview Across Cohorts",
        "",
        "### Word Error Rate (WER)",
        "| Stratification | N | Mean | Median | SD | IQR | 5th Pct | 95th Pct |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for grp_name, sub in [
        ("Spanish -> German", df_content[df_content["source_language"] == "Spanish"]),
        ("German -> Spanish", df_content[df_content["source_language"] == "German"]),
        ("HC: Spanish -> German", df_content[(df_content["source_language"] == "Spanish") & (df_content["diagnosis"] == "HC")]),
        ("PD: Spanish -> German", df_content[(df_content["source_language"] == "Spanish") & (df_content["diagnosis"] == "PD")]),
        ("HC: German -> Spanish", df_content[(df_content["source_language"] == "German") & (df_content["diagnosis"] == "HC")]),
        ("PD: German -> Spanish", df_content[(df_content["source_language"] == "German") & (df_content["diagnosis"] == "PD")]),
        ("Full Cohort (Set A)", df_content),
        ("Dev-Excluded (Set B)", df_content[df_content["cohort_tag"] != "PILOT_DEVELOPMENT"]),
        ("Pilot Development Subgroup", df_content[df_content["cohort_tag"] == "PILOT_DEVELOPMENT"]),
        ("Validation 80 Subgroup", df_content[df_content["cohort_tag"] == "VALIDATION_80"]),
        ("Full Cohort Only Subgroup", df_content[df_content["cohort_tag"] == "FULL_COHORT_ONLY"]),
    ]:
        content_md.append(get_stats_row(grp_name, sub["wer"]))

    content_md.extend([
        "",
        "### Word Information Preserved (WIP)",
        "| Stratification | N | Mean | Median | SD | IQR | 5th Pct | 95th Pct |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ])
    for grp_name, sub in [
        ("Spanish -> German", df_content[df_content["source_language"] == "Spanish"]),
        ("German -> Spanish", df_content[df_content["source_language"] == "German"]),
        ("Full Cohort (Set A)", df_content),
        ("Dev-Excluded (Set B)", df_content[df_content["cohort_tag"] != "PILOT_DEVELOPMENT"]),
    ]:
        content_md.append(get_stats_row(grp_name, sub["wip"]))

    content_md.extend([
        "",
        "### Character Error Rate (CER)",
        "| Stratification | N | Mean | Median | SD | IQR | 5th Pct | 95th Pct |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ])
    for grp_name, sub in [
        ("Spanish -> German", df_content[df_content["source_language"] == "Spanish"]),
        ("German -> Spanish", df_content[df_content["source_language"] == "German"]),
        ("Full Cohort (Set A)", df_content),
        ("Dev-Excluded (Set B)", df_content[df_content["cohort_tag"] != "PILOT_DEVELOPMENT"]),
    ]:
        content_md.append(get_stats_row(grp_name, sub["cer"]))

    content_rep_path.write_text("\n".join(content_md), encoding="utf-8")
    print(f"Saved Content Accuracy Report to {content_rep_path}")

    # ========================================================
    # STEP 4: ECAPA-TDNN SPEAKER PRESERVATION (Section 17)
    # ========================================================
    print("\n=== Initializing ECAPA-TDNN for Speaker Biometric Preservation ===")
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(WORKSPACE / "models" / "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cpu"}
    )

    def extract_embedding(filepath):
        y, fs = sf.read(str(filepath))
        if y.ndim > 1:
            y = y.mean(axis=1)
        signal = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        if fs != 16000:
            signal = torchaudio.functional.resample(signal, orig_freq=fs, new_freq=16000)
        with torch.no_grad():
            emb = ecapa.encode_batch(signal).squeeze().cpu().numpy()
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb


    print("Extracting ECAPA-TDNN embeddings for 276 original and 276 generated recordings...")
    orig_embs = {}
    gen_embs = {}

    for idx, r in df_meta.iterrows():
        sid = r["safe_id"]
        orig_p = r["audio_path"]
        gen_p = r["generated_path"]
        orig_embs[sid] = extract_embedding(orig_p)
        gen_embs[sid] = extract_embedding(gen_p)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(df_meta):
            print(f"  Extracted embeddings for {idx + 1}/{len(df_meta)} subjects...")

    # Matched Pair Similarities
    matched_records = []
    for idx, r in df_meta.iterrows():
        sid = r["safe_id"]
        sim = float(np.dot(orig_embs[sid], gen_embs[sid]))
        matched_records.append({
            "safe_id": sid,
            "source_language": r["language"],
            "target_language": r["target_language"],
            "diagnosis": r["diagnosis"],
            "cohort_tag": r["cohort_tag"],
            "matched_speaker_similarity": round(sim, 4),
            "passes_040_gate": sim >= 0.40
        })

    df_spk = pd.DataFrame(matched_records)
    spk_csv = SPK_DIR / "speaker_similarity_276.csv"
    df_spk.to_csv(spk_csv, index=False)
    print(f"Saved matched speaker similarities to {spk_csv}")

    # Negative Different-Speaker Controls (reproducible seed 42)
    print("Generating large reproducible different-speaker negative controls (seed 42)...")
    np.random.seed(42)
    sids = df_meta["safe_id"].tolist()
    ctrl_records = []

    # Generate strictly mismatched cross-subject pairs
    for i, sid_a in enumerate(sids):
        row_a = df_meta[df_meta["safe_id"] == sid_a].iloc[0]
        candidates = [s for s in sids if s != sid_a and df_meta[df_meta["safe_id"]==s].iloc[0]["speaker_id"] != row_a["speaker_id"]]
        sampled_ctrls = np.random.choice(candidates, size=min(40, len(candidates)), replace=False)

        for sid_b in sampled_ctrls:
            row_b = df_meta[df_meta["safe_id"] == sid_b].iloc[0]
            sim = float(np.dot(orig_embs[sid_a], gen_embs[sid_b]))
            ctrl_records.append({
                "source_safe_id": sid_a,
                "different_speaker_safe_id": sid_b,
                "source_language": row_a["language"],
                "target_language": row_a["target_language"],
                "source_diagnosis": row_a["diagnosis"],
                "target_diagnosis": row_b["diagnosis"],
                "different_speaker_similarity": round(sim, 4)
            })

    df_ctrl = pd.DataFrame(ctrl_records)
    ctrl_csv = SPK_DIR / "different_speaker_controls.csv"
    df_ctrl.to_csv(ctrl_csv, index=False)
    print(f"Saved {len(df_ctrl)} different-speaker controls to {ctrl_csv}")

    matched_mean = df_spk["matched_speaker_similarity"].mean()
    ctrl_mean = df_ctrl["different_speaker_similarity"].mean()
    margin = matched_mean - ctrl_mean
    print(f"Speaker Biometric Summary: Matched Mean = {matched_mean:.4f}, Control Mean = {ctrl_mean:.4f}, Discrimination Margin = +{margin:.4f}")

    # Generate Speaker Report
    spk_rep_path = SPK_DIR / "speaker_report.md"
    def spk_stats_row(grp_label, sub_m, sub_c):
        m_vals = sub_m["matched_speaker_similarity"]
        c_vals = sub_c["different_speaker_similarity"]
        m_q05, m_q95 = np.percentile(m_vals, [5, 95])
        c_q05, c_q95 = np.percentile(c_vals, [5, 95])
        delta = m_vals.mean() - c_vals.mean()
        return (f"| {grp_label} | {m_vals.mean():.4f} | {m_vals.median():.4f} | {m_vals.std():.4f} | [{m_q05:.4f}, {m_q95:.4f}] | "
                f"{c_vals.mean():.4f} | {c_vals.median():.4f} | {c_vals.std():.4f} | [{c_q05:.4f}, {c_q95:.4f}] | +{delta:.4f} |")

    spk_md = [
        "# Speaker Biometric Identity Preservation Report — Full 276 Evaluation",
        "",
        "## Metric Overview (ECAPA-TDNN Cosine Similarity)",
        f"- **Matched Pairs Evaluated**: {len(df_spk)}",
        f"- **Different-Speaker Control Pairs (Seed 42)**: {len(df_ctrl)}",
        f"- **Matched Mean Similarity**: {matched_mean:.4f} (Median: {df_spk['matched_speaker_similarity'].median():.4f}, SD: {df_spk['matched_speaker_similarity'].std():.4f})",
        f"- **Control Mean Similarity**: {ctrl_mean:.4f} (Median: {df_ctrl['different_speaker_similarity'].median():.4f}, SD: {df_ctrl['different_speaker_similarity'].std():.4f})",
        f"- **Discrimination Margin**: +{margin:.4f}",
        "",
        "## Stratified Speaker Preservation Breakdown",
        "",
        "| Stratification | Matched Mean | Matched Med | Matched SD | Matched [5th, 95th] | Control Mean | Control Med | Control SD | Control [5th, 95th] | Margin |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for label, s_lang, diag in [
        ("Spanish -> German (Overall)", "Spanish", None),
        ("German -> Spanish (Overall)", "German", None),
        ("Healthy Controls (HC)", None, "HC"),
        ("Parkinson's Disease (PD)", None, "PD"),
        ("Full Cohort (Set A)", None, None)
    ]:
        sub_m = df_spk.copy()
        sub_c = df_ctrl.copy()
        if s_lang:
            sub_m = sub_m[sub_m["source_language"] == s_lang]
            sub_c = sub_c[sub_c["source_language"] == s_lang]
        if diag:
            sub_m = sub_m[sub_m["diagnosis"] == diag]
            sub_c = sub_c[sub_c["source_diagnosis"] == diag]
        spk_md.append(spk_stats_row(label, sub_m, sub_c))

    spk_rep_path.write_text("\n".join(spk_md), encoding="utf-8")
    print(f"Saved Speaker Preservation Report to {spk_rep_path}")

if __name__ == "__main__":
    main()
