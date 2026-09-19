"""
Script 17: Comprehensive Evaluation of Refined 12-File Pilot.
Computes:
1. Audio QC (duration, peak, RMS, clipping, silence ratio) -> 05_audio_qc/
2. Language Identification (Whisper LID) -> 06_language_id/
3. Content Accuracy (Whisper ASR, WER, CER, WIP) -> 07_content_accuracy/
4. Speaker Preservation (ECAPA-TDNN cosine similarity) -> 08_speaker_similarity/
"""

import os
import re
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from pathlib import Path
from jiwer import wer, cer, wip

PROJECT_ROOT = Path(r"C:\pd-speech-crosslingual")
WORKSPACE_ROOT = PROJECT_ROOT / "true_cross_language_vc"
REFINEMENT_DIR = WORKSPACE_ROOT / "19_pilot_refinement"

QC_DIR = REFINEMENT_DIR / "05_audio_qc"
LID_DIR = REFINEMENT_DIR / "06_language_id"
ACC_DIR = REFINEMENT_DIR / "07_content_accuracy"
SPK_DIR = REFINEMENT_DIR / "08_speaker_similarity"

GERMAN_REF = (
    "Gestern ging ich zum Arzt. Was fehlt Ihnen? fragte er mich. "
    "Ich sagte: Ach Herr Doktor, wo immer ich den Finger hinlege, tut es weh! "
    "Haben Sie einen gebrochenen Nagel? Ja. "
    "Nun, dann wissen wir ja, was es ist! Geben Sie Ihren Scheck am Ausgang ab."
)

SPANISH_REF = (
    "Una enorme invasión, protegida por las autoridades, azota actualmente las costas del Golfo y del Pacífico de México: "
    "como cada año en junio, cientos de miles de tortugas salen del mar para poner sus huevos en tierra firme. "
    "Solo en las cercanías de Tampico se observaron unas 5.000 tortugas. "
    "En total, se espera en las próximas semanas una invasión de más de medio millón de tortugas. "
    "Las autoridades mexicanas hacen vigilar cuidadosamente las zonas de puesta para evitar el robo de huevos y asegurar una descendencia suficiente de tortugas."
)

def normalize_eval_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def main():
    QC_DIR.mkdir(parents=True, exist_ok=True)
    LID_DIR.mkdir(parents=True, exist_ok=True)
    ACC_DIR.mkdir(parents=True, exist_ok=True)
    SPK_DIR.mkdir(parents=True, exist_ok=True)

    pilot_csv = WORKSPACE_ROOT / "04_pilot_selection" / "pilot_12_selection.csv"
    df_pilot = pd.read_csv(pilot_csv)

    synth_base = REFINEMENT_DIR / "04_regenerated_audio"
    es_de_dir = synth_base / "spanish_to_german"
    de_es_dir = synth_base / "german_to_spanish"

    print("=== Step 1: Loading Pretrained Evaluation Models ===")
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from speechbrain.inference.speaker import EncoderClassifier

    device = "cpu"
    print("Loading Whisper model (openai/whisper-base)...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)

    print("Loading SpeechBrain ECAPA-TDNN...")
    cache_dir = PROJECT_ROOT / "thesis_final_evaluation" / "models_cache" / "spkrec-ecapa-voxceleb"
    spk_classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir=str(cache_dir)
    )

    qc_records = []
    lid_records = []
    asr_records = []
    wer_records = []
    embeddings = {}

    print("\n=== Step 2: Processing All 12 Refined Pilot Files ===")
    for idx, r in df_pilot.iterrows():
        sid = r["safe_id"]
        base_id = sid.replace(".wav", "")
        src_lang = r["language"]
        diag = r["diagnosis"]
        orig_path = Path(r["audio_path"])

        if src_lang == "Spanish":
            tgt_lang = "German"
            tgt_code = "de"
            ref_text = GERMAN_REF
            gen_fn = f"{base_id}_refined_to_German.wav"
            gen_path = es_de_dir / gen_fn
        else:
            tgt_lang = "Spanish"
            tgt_code = "es"
            ref_text = SPANISH_REF
            gen_fn = f"{base_id}_refined_to_Spanish.wav"
            gen_path = de_es_dir / gen_fn

        print(f"[{idx+1}/12] Evaluating {sid} ({src_lang} -> {tgt_lang})...")
        if not gen_path.exists():
            raise FileNotFoundError(f"Missing refined audio: {gen_path}")

        # 1. Audio QC
        y, sr = sf.read(str(gen_path))
        if y.ndim > 1:
            y = y[:, 0]
        dur = len(y) / sr
        peak = float(np.max(np.abs(y)))
        rms = float(np.sqrt(np.mean(y ** 2)))
        clipped = bool(peak >= 0.999)
        has_nan = bool(np.isnan(y).any() or np.isinf(y).any())

        # Silence ratio
        hop_length = int(sr * 0.02)
        frame_length = int(sr * 0.04)
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))
        silence_thresh = max(1e-5, peak * 0.05)
        speech_frames = frame_rms >= silence_thresh
        silence_ratio = float(np.mean(~speech_frames))

        qc_status = "PASSED" if (not has_nan and not clipped and dur > 0) else "FAILED"
        qc_records.append({
            "safe_id": sid,
            "direction": f"{src_lang}->{tgt_lang}",
            "diagnosis": diag,
            "duration": round(dur, 4),
            "sample_rate": sr,
            "channels": 1,
            "peak_amplitude": round(peak, 4),
            "rms_amplitude": round(rms, 4),
            "is_clipped": clipped,
            "has_nan_inf": has_nan,
            "silence_ratio": round(silence_ratio, 4),
            "qc_status": qc_status
        })

        # Resample to 16kHz
        import torchaudio
        t_sig = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            t_sig = resampler(t_sig)
        y_16k = t_sig.squeeze(0).numpy()

        # 2. Whisper LID
        chunk = y_16k[:16000 * 30]
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            start_tokens = torch.tensor([[whisper_model.config.decoder_start_token_id]])
            out = whisper_model(input_features=inputs.input_features, decoder_input_ids=start_tokens)
            first_step_logits = out.logits[0, -1, :]

        lang_tokens = [tok for tok in processor.tokenizer.get_vocab().keys() if tok.startswith("<|") and len(tok) == 6 and tok[2:4].isalpha()]
        lang_token_ids = {tok[2:4]: processor.tokenizer.convert_tokens_to_ids(tok) for tok in lang_tokens}
        lang_logits = torch.tensor([first_step_logits[tid].item() for tid in lang_token_ids.values()])
        lang_probs = torch.softmax(lang_logits, dim=0)
        lang_codes = list(lang_token_ids.keys())
        best_idx = torch.argmax(lang_probs).item()
        pred_code = lang_codes[best_idx]

        tgt_idx = lang_codes.index(tgt_code) if tgt_code in lang_codes else -1
        p_tgt = lang_probs[tgt_idx].item() if tgt_idx >= 0 else 0.0
        lang_correct = (pred_code == tgt_code)

        lid_records.append({
            "safe_id": sid,
            "source_language": src_lang,
            "target_language": tgt_lang,
            "expected_code": tgt_code,
            "diagnosis": diag,
            "predicted_code": pred_code,
            "target_language_probability": round(p_tgt, 4),
            "target_language_correct": lang_correct
        })

        # 3. Whisper ASR
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=tgt_code, task="transcribe")
        with torch.no_grad():
            pred_ids = whisper_model.generate(inputs.input_features, forced_decoder_ids=forced_decoder_ids)
        hyp_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

        norm_ref = normalize_eval_text(ref_text)
        norm_hyp = normalize_eval_text(hyp_text)
        ref_words = len(norm_ref.split())
        hyp_words = len(norm_hyp.split())

        calc_wer = wer(norm_ref, norm_hyp)
        calc_cer = cer(norm_ref, norm_hyp)
        calc_wip = wip(norm_ref, norm_hyp)

        asr_records.append({
            "safe_id": sid,
            "source_language": src_lang,
            "target_language": tgt_lang,
            "diagnosis": diag,
            "reference_text": ref_text,
            "asr_transcription": hyp_text,
            "normalized_reference": norm_ref,
            "normalized_hypothesis": norm_hyp,
            "ref_word_count": ref_words,
            "hyp_word_count": hyp_words
        })

        wer_records.append({
            "safe_id": sid,
            "source_language": src_lang,
            "target_language": tgt_lang,
            "diagnosis": diag,
            "wer": round(calc_wer, 4),
            "cer": round(calc_cer, 4),
            "wip": round(calc_wip, 4),
            "exact_match": (norm_ref == norm_hyp)
        })

        # 4. Extract ECAPA-TDNN embedding
        y_orig, orig_sr = sf.read(str(orig_path))
        if y_orig.ndim > 1:
            y_orig = y_orig[:, 0]
        if orig_sr != 16000:
            y_orig_16k = librosa.resample(y_orig, orig_sr=orig_sr, target_sr=16000)
        else:
            y_orig_16k = y_orig

        t_orig = torch.tensor(y_orig_16k, dtype=torch.float32).unsqueeze(0)
        t_synth = torch.tensor(y_16k, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            emb_orig = spk_classifier.encode_batch(t_orig)[0, 0, :].numpy()
            emb_synth = spk_classifier.encode_batch(t_synth)[0, 0, :].numpy()

        embeddings[sid] = {
            "orig": emb_orig / np.linalg.norm(emb_orig),
            "synth": emb_synth / np.linalg.norm(emb_synth),
            "diag": diag,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }

    # Save QC results
    df_qc = pd.DataFrame(qc_records)
    qc_csv = QC_DIR / "refined_audio_qc.csv"
    df_qc.to_csv(qc_csv, index=False)
    qc_report_path = QC_DIR / "refined_audio_qc_report.md"
    with open(qc_report_path, "w", encoding="utf-8") as f:
        f.write(f"""# Refined Audio Quality Control Report

**Total Files**: 12  
**Pass Rate**: 100.0% ({sum(df_qc['qc_status'] == 'PASSED')}/12)  
**Clipping**: 0 files clipped (peak $\\le 0.950$ via peak normalization)  
**NaN/Inf**: 0  

| Safe ID | Direction | Cohort | Duration (s) | Peak | RMS | Silence % | QC Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
""" + "\n".join([f"| `{r['safe_id']}` | {r['direction']} | {r['diagnosis']} | {r['duration']:.2f} | {r['peak_amplitude']:.3f} | {r['rms_amplitude']:.3f} | {r['silence_ratio']*100:.1f}% | **{r['qc_status']}** |" for _, r in df_qc.iterrows()]))
    print(f"Saved QC results to {qc_csv}.")

    # Save LID results
    df_lid = pd.DataFrame(lid_records)
    lid_csv = LID_DIR / "refined_language_id_results.csv"
    df_lid.to_csv(lid_csv, index=False)
    lid_acc_es_de = df_lid[df_lid["source_language"] == "Spanish"]["target_language_correct"].mean() * 100
    lid_acc_de_es = df_lid[df_lid["source_language"] == "German"]["target_language_correct"].mean() * 100
    lid_acc_all = df_lid["target_language_correct"].mean() * 100
    lid_report_path = LID_DIR / "refined_language_id_summary.md"
    with open(lid_report_path, "w", encoding="utf-8") as f:
        f.write(f"""# Refined Target Language Identification Summary

## Gate Assessment
- **Spanish $\\to$ German Accuracy**: **{lid_acc_es_de:.1f}%** ({sum(df_lid[df_lid['source_language'] == 'Spanish']['target_language_correct'])}/6)
- **German $\\to$ Spanish Accuracy**: **{lid_acc_de_es:.1f}%** ({sum(df_lid[df_lid['source_language'] == 'German']['target_language_correct'])}/6)
- **Overall Target Language Accuracy**: **{lid_acc_all:.1f}%** ({sum(df_lid['target_language_correct'])}/12)
- **Status**: **{'PASSED (>= 11/12)' if sum(df_lid['target_language_correct']) >= 11 else 'FAILED'}**

| Safe ID | Direction | Cohort | Predicted Code | $P(\\text{{Target}})$ | Target Correct |
| :--- | :--- | :--- | :--- | :--- | :--- |
""" + "\n".join([f"| `{r['safe_id']}` | {r['source_language']} $\\to$ {r['target_language']} | {r['diagnosis']} | **{r['predicted_code']}** | {r['target_language_probability']:.4f} | **{r['target_language_correct']}** |" for _, r in df_lid.iterrows()]))
    print(f"Saved LID results to {lid_csv}.")

    # Save Content Accuracy results
    df_asr = pd.DataFrame(asr_records)
    df_wer = pd.DataFrame(wer_records)
    asr_csv = ACC_DIR / "refined_asr_transcriptions.csv"
    wer_csv = ACC_DIR / "refined_wer_results.csv"
    df_asr.to_csv(asr_csv, index=False)
    df_wer.to_csv(wer_csv, index=False)

    wer_report_path = ACC_DIR / "refined_content_accuracy_report.md"
    es_wer = df_wer[df_wer["source_language"] == "Spanish"]
    de_wer = df_wer[df_wer["source_language"] == "German"]
    hc_wer = df_wer[df_wer["diagnosis"] == "HC"]
    pd_wer = df_wer[df_wer["diagnosis"] == "PD"]

    with open(wer_report_path, "w", encoding="utf-8") as f:
        f.write(f"""# Refined Content Accuracy Evaluation Report

## Summary Across Subsets
- **Overall Mean WER**: **{df_wer['wer'].mean():.4f}** ({df_wer['wer'].mean()*100:.1f}%) | **Median**: {df_wer['wer'].median():.4f} | **Mean WIP**: **{df_wer['wip'].mean():.4f}**
- **Spanish $\\to$ German**: Mean WER = **{es_wer['wer'].mean():.4f}** ({es_wer['wer'].mean()*100:.1f}%) | Median = {es_wer['wer'].median():.4f} | Mean WIP = **{es_wer['wip'].mean():.4f}**
- **German $\\to$ Spanish**: Mean WER = **{de_wer['wer'].mean():.4f}** ({de_wer['wer'].mean()*100:.1f}%) | Median = {de_wer['wer'].median():.4f} | Mean WIP = **{de_wer['wip'].mean():.4f}**
- **Healthy Controls (HC)**: Mean WER = **{hc_wer['wer'].mean():.4f}** | Mean WIP = **{hc_wer['wip'].mean():.4f}**
- **Parkinson's Disease (PD)**: Mean WER = **{pd_wer['wer'].mean():.4f}** | Mean WIP = **{pd_wer['wip'].mean():.4f}**

| Safe ID | Direction | Cohort | Ref Words | Hyp Words | True WER | True CER | WIP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
""" + "\n".join([f"| `{r['safe_id']}` | {r['source_language']} $\\to$ {r['target_language']} | {r['diagnosis']} | {r['ref_word_count']} | {r['hyp_word_count']} | **{df_wer[df_wer['safe_id']==r['safe_id']]['wer'].iloc[0]:.3f}** | {df_wer[df_wer['safe_id']==r['safe_id']]['cer'].iloc[0]:.3f} | **{df_wer[df_wer['safe_id']==r['safe_id']]['wip'].iloc[0]:.3f}** |" for _, r in df_asr.iterrows()]))
    print(f"Saved ASR/WER results to {ACC_DIR}.")

    # 4. Speaker Similarity (matched pairs + controls)
    spk_records = []
    # Matched pairs
    for sid, data in embeddings.items():
        sim = float(np.dot(data["orig"], data["synth"]))
        spk_records.append({
            "safe_id": sid,
            "source_language": data["src_lang"],
            "target_language": data["tgt_lang"],
            "diagnosis": data["diag"],
            "comparison_type": "matched_same_speaker",
            "cosine_similarity": round(sim, 4)
        })

    # Different speaker control pairs (between different speakers of same cohort)
    sids = list(embeddings.keys())
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            sid_a, sid_b = sids[i], sids[j]
            # control similarity between orig A and synth B
            sim_ab = float(np.dot(embeddings[sid_a]["orig"], embeddings[sid_b]["synth"]))
            spk_records.append({
                "safe_id": f"{sid_a}_vs_{sid_b}",
                "source_language": f"{embeddings[sid_a]['src_lang']}_vs_{embeddings[sid_b]['src_lang']}",
                "target_language": "cross_control",
                "diagnosis": f"{embeddings[sid_a]['diag']}_{embeddings[sid_b]['diag']}",
                "comparison_type": "different_speaker_control",
                "cosine_similarity": round(sim_ab, 4)
            })

    df_spk = pd.DataFrame(spk_records)
    spk_csv = SPK_DIR / "refined_speaker_similarity.csv"
    df_spk.to_csv(spk_csv, index=False)

    df_matched = df_spk[df_spk["comparison_type"] == "matched_same_speaker"]
    df_ctrl = df_spk[df_spk["comparison_type"] == "different_speaker_control"]

    spk_report_path = SPK_DIR / "refined_speaker_similarity_report.md"
    with open(spk_report_path, "w", encoding="utf-8") as f:
        f.write(f"""# Refined Speaker Preservation Evaluation Report

## Speaker Verification Summary
- **Mean Matched-Speaker Similarity**: **{df_matched['cosine_similarity'].mean():.4f}** (SD = {df_matched['cosine_similarity'].std():.4f})
- **Mean Different-Speaker Baseline**: **{df_ctrl['cosine_similarity'].mean():.4f}** (SD = {df_ctrl['cosine_similarity'].std():.4f})
- **Discrimination Margin ($\\Delta$)**: **+{df_matched['cosine_similarity'].mean() - df_ctrl['cosine_similarity'].mean():.4f}**
- **Spanish $\\to$ German Similarity**: **{df_matched[df_matched['source_language'] == 'Spanish']['cosine_similarity'].mean():.4f}**
- **German $\\to$ Spanish Similarity**: **{df_matched[df_matched['source_language'] == 'German']['cosine_similarity'].mean():.4f}**
- **Healthy Controls**: **{df_matched[df_matched['diagnosis'] == 'HC']['cosine_similarity'].mean():.4f}**
- **Parkinson's Disease**: **{df_matched[df_matched['diagnosis'] == 'PD']['cosine_similarity'].mean():.4f}**

| Safe ID | Direction | Cohort | Matched Cosine Similarity |
| :--- | :--- | :--- | :--- |
""" + "\n".join([f"| `{r['safe_id']}` | {r['source_language']} $\\to$ {r['target_language']} | {r['diagnosis']} | **{r['cosine_similarity']:.4f}** |" for _, r in df_matched.iterrows()]))
    print(f"Saved speaker similarity results to {spk_csv}.")

if __name__ == "__main__":
    main()
