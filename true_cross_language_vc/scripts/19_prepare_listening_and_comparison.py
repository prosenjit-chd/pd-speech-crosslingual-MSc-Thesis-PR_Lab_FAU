"""
Script 19: Prepare Listening Review Package and Before-vs-After Comparison Figures.
Produces:
- 19_pilot_refinement/10_listening_review/manual_review.csv
- 19_pilot_refinement/10_listening_review/README.md
- 19_pilot_refinement/11_comparison/original_vs_refined.csv
- 19_pilot_refinement/11_comparison/fig1_wer_comparison.png
- 19_pilot_refinement/11_comparison/fig2_wip_comparison.png
- 19_pilot_refinement/11_comparison/fig3_speaker_similarity_comparison.png
- 19_pilot_refinement/11_comparison/fig4_target_language_confidence_comparison.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"C:\pd-speech-crosslingual")
WORKSPACE_ROOT = PROJECT_ROOT / "true_cross_language_vc"
REFINEMENT_DIR = WORKSPACE_ROOT / "19_pilot_refinement"
LISTEN_DIR = REFINEMENT_DIR / "10_listening_review"
COMP_DIR = REFINEMENT_DIR / "11_comparison"

def main():
    LISTEN_DIR.mkdir(parents=True, exist_ok=True)
    COMP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Original Pilot Metrics
    orig_diag_csv = REFINEMENT_DIR / "00_diagnosis" / "original_pilot_diagnosis.csv"
    df_orig = pd.read_csv(orig_diag_csv)

    # 2. Load Refined Pilot Metrics
    ref_qc_csv = REFINEMENT_DIR / "05_audio_qc" / "refined_audio_qc.csv"
    ref_lid_csv = REFINEMENT_DIR / "06_language_id" / "refined_language_id_results.csv"
    ref_wer_csv = REFINEMENT_DIR / "07_content_accuracy" / "refined_wer_results.csv"
    ref_asr_csv = REFINEMENT_DIR / "07_content_accuracy" / "refined_asr_transcriptions.csv"
    ref_spk_csv = REFINEMENT_DIR / "08_speaker_similarity" / "refined_speaker_similarity.csv"

    df_ref_qc = pd.read_csv(ref_qc_csv)
    df_ref_lid = pd.read_csv(ref_lid_csv)
    df_ref_wer = pd.read_csv(ref_wer_csv)
    df_ref_asr = pd.read_csv(ref_asr_csv)
    df_ref_spk = pd.read_csv(ref_spk_csv)
    df_ref_matched_spk = df_ref_spk[df_ref_spk["comparison_type"] == "matched_same_speaker"].copy()

    # Merge refined metrics
    df_ref = pd.merge(df_ref_qc[["safe_id", "direction", "diagnosis", "duration", "qc_status", "is_clipped"]],
                      df_ref_lid[["safe_id", "predicted_code", "target_language_probability", "target_language_correct"]], on="safe_id")
    df_ref = pd.merge(df_ref, df_ref_wer[["safe_id", "wer", "cer", "wip"]], on="safe_id")
    df_ref = pd.merge(df_ref, df_ref_asr[["safe_id", "reference_text", "asr_transcription", "ref_word_count", "hyp_word_count"]], on="safe_id")
    df_ref = pd.merge(df_ref, df_ref_matched_spk[["safe_id", "cosine_similarity"]], on="safe_id")

    # Rename columns for clarity
    df_ref.rename(columns={"cosine_similarity": "speaker_similarity"}, inplace=True)

    print("=== Step 1: Generating Manual Listening Review Package ===")
    pilot_csv = WORKSPACE_ROOT / "04_pilot_selection" / "pilot_12_selection.csv"
    df_pilot = pd.read_csv(pilot_csv)

    listen_rows = []
    # Key representative 4: SP_HC_001, SP_PD_001, DE_HC_001, DE_PD_001
    rep_sids = ["SP_HC_001.wav", "SP_PD_001.wav", "DE_HC_001.wav", "DE_PD_001.wav"]

    for _, r in df_ref.iterrows():
        sid = r["safe_id"]
        base_id = sid.replace(".wav", "")
        diag = r["diagnosis"]
        p_row = df_pilot[df_pilot["safe_id"] == sid].iloc[0]
        o_row = df_orig[df_orig["safe_id"] == sid].iloc[0]

        is_rep = sid in rep_sids
        orig_audio = p_row["audio_path"]
        pilot1_audio = o_row["generated_audio_path"]

        if "Spanish->" in r["direction"]:
            refined_audio = str(REFINEMENT_DIR / "04_regenerated_audio" / "spanish_to_german" / f"{base_id}_refined_to_German.wav")
        else:
            refined_audio = str(REFINEMENT_DIR / "04_regenerated_audio" / "german_to_spanish" / f"{base_id}_refined_to_Spanish.wav")

        listen_rows.append({
            "safe_id": sid,
            "diagnosis": diag,
            "direction": r["direction"],
            "is_primary_representative": is_rep,
            "original_source_audio": orig_audio,
            "first_pilot_audio": pilot1_audio,
            "refined_audio": refined_audio,
            "target_language_correct": "Confirmed" if r["target_language_correct"] else "Unconfirmed",
            "first_pilot_wer": round(float(o_row["WER"]), 3),
            "refined_wer": round(float(r["wer"]), 3),
            "first_pilot_wip": round(float(o_row["WIP"]), 3),
            "refined_wip": round(float(r["wip"]), 3),
            "speaker_similarity": round(float(r["speaker_similarity"]), 4),
            "content_intelligible": "",
            "speaker_resemblance": "",
            "artifacts": "",
            "pathology_characteristics_subjective": "",
            "notes": "Completed sentence-segmented synthesis"
        })

    df_listen = pd.DataFrame(listen_rows)
    listen_csv = LISTEN_DIR / "manual_review.csv"
    df_listen.to_csv(listen_csv, index=False)
    print(f"Saved listening review CSV to {listen_csv}.")

    listen_readme = LISTEN_DIR / "README.md"
    readme_text = """# Manual Listening Review Protocol (3-Way Comparison)

This directory provides the structured listening review protocol for comparing:
1. **Original Source Speech** (unprocessed clinical recording)
2. **First Pilot Conversion** (unsegmented baseline XTTS synthesis)
3. **Refined Conversion** (sentence-segmented text + deterministic clean 10s speaker reference)

## Primary Representative Utterances (4 Subjects)
- **Spanish HC $\\to$ German**: `SP_HC_001.wav`
- **Spanish PD $\\to$ German**: `SP_PD_001.wav`
- **German HC $\\to$ Spanish**: `DE_HC_001.wav`
- **German PD $\\to$ Spanish**: `DE_PD_001.wav`

## Subjective Evaluation Criteria
For human clinical listening audit:
- `content_intelligible`: Scale 1–5 (1 = unintelligible, 5 = perfectly intelligible)
- `speaker_resemblance`: Scale 1–5 (1 = different person, 5 = identical speaker timbre)
- `artifacts`: Note any unnatural prosody, metallic robotic timbre, or vocoder clicks
- `pathology_characteristics_subjective`: Note whether dysarthric articulation (slurring, tremor, breathiness) is audible or regularized.
"""
    with open(listen_readme, "w", encoding="utf-8") as f:
        f.write(readme_text)

    print("\n=== Step 2: Generating Before-vs-After Comparison Table ===")
    comp_records = []

    subsets = [
        ("Spanish->German", df_orig[df_orig["direction"] == "Spanish->German"], df_ref[df_ref["direction"] == "Spanish->German"]),
        ("German->Spanish", df_orig[df_orig["direction"] == "German->Spanish"], df_ref[df_ref["direction"] == "German->Spanish"]),
        ("Overall (12 Files)", df_orig, df_ref)
    ]

    for name, sub_o, sub_r in subsets:
        comp_records.append({
            "subset": name,
            "orig_target_lang_acc": round(sub_o["target_language_correct"].mean() * 100, 1),
            "ref_target_lang_acc": round(sub_r["target_language_correct"].mean() * 100, 1),
            "orig_mean_wer": round(sub_o["WER"].mean(), 4),
            "ref_mean_wer": round(sub_r["wer"].mean(), 4),
            "orig_median_wer": round(sub_o["WER"].median(), 4),
            "ref_median_wer": round(sub_r["wer"].median(), 4),
            "orig_mean_cer": round(sub_o["CER"].mean(), 4),
            "ref_mean_cer": round(sub_r["cer"].mean(), 4),
            "orig_mean_wip": round(sub_o["WIP"].mean(), 4),
            "ref_mean_wip": round(sub_r["wip"].mean(), 4),
            "orig_mean_spk_sim": round(sub_o["speaker_cosine_similarity"].mean(), 4),
            "ref_mean_spk_sim": round(sub_r["speaker_similarity"].mean(), 4),
            "orig_failed_files": int((sub_o["quality_classification"] == "POOR").sum()),
            "ref_failed_files": int((sub_r["qc_status"] != "PASSED").sum()),
            "orig_clipped_files": int(sub_o["is_clipped"].sum()),
            "ref_clipped_files": int(sub_r["is_clipped"].sum())
        })

    df_comp = pd.DataFrame(comp_records)
    comp_csv = COMP_DIR / "original_vs_refined.csv"
    df_comp.to_csv(comp_csv, index=False)
    print(f"Saved before-vs-after comparison CSV to {comp_csv}.")

    print("\n=== Step 3: Generating Comparison Figures ===")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Figure 1: WER Before vs Refined
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df_orig))
    width = 0.35
    ax.bar(x - width/2, df_orig["WER"] * 100, width, label="First Pilot", color="#d9534f", alpha=0.85)
    ax.bar(x + width/2, df_ref["wer"] * 100, width, label="Refined Pilot", color="#5cb85c", alpha=0.85)
    ax.set_ylabel("Word Error Rate (WER %)", fontsize=11)
    ax.set_title("Word Error Rate: First Pilot vs. Refined Pilot (All 12 Files)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([sid.replace(".wav", "") for sid in df_orig["safe_id"]], rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig1_path = COMP_DIR / "fig1_wer_comparison.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"  [OK] Saved {fig1_path.name}")

    # Figure 2: WIP Before vs Refined
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, df_orig["WIP"], width, label="First Pilot", color="#f0ad4e", alpha=0.85)
    ax.bar(x + width/2, df_ref["wip"], width, label="Refined Pilot", color="#0275d8", alpha=0.85)
    ax.set_ylabel("Word Information Preserved (WIP)", fontsize=11)
    ax.set_title("Word Information Preserved (WIP): First Pilot vs. Refined Pilot", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([sid.replace(".wav", "") for sid in df_orig["safe_id"]], rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig2_path = COMP_DIR / "fig2_wip_comparison.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"  [OK] Saved {fig2_path.name}")

    # Figure 3: Speaker Similarity Before vs Refined
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, df_orig["speaker_cosine_similarity"], width, label="First Pilot", color="#6f42c1", alpha=0.85)
    ax.bar(x + width/2, df_ref["speaker_similarity"], width, label="Refined Pilot", color="#20c997", alpha=0.85)
    ax.axhline(0.0798, color="gray", linestyle="--", label="Different-Speaker Control Baseline (0.0798)")
    ax.set_ylabel("ECAPA-TDNN Cosine Similarity", fontsize=11)
    ax.set_title("Cross-Language Speaker Similarity: First Pilot vs. Refined Pilot", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([sid.replace(".wav", "") for sid in df_orig["safe_id"]], rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig3_path = COMP_DIR / "fig3_speaker_similarity_comparison.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"  [OK] Saved {fig3_path.name}")

    # Figure 4: Target Language Confidence Before vs Refined
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, df_orig["target_language_probability"] * 100, width, label="First Pilot", color="#e83e8c", alpha=0.85)
    ax.bar(x + width/2, df_ref["target_language_probability"] * 100, width, label="Refined Pilot", color="#17a2b8", alpha=0.85)
    ax.set_ylabel("Target Language Posterior Probability (%)", fontsize=11)
    ax.set_title("Whisper Target-Language Identification Confidence", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([sid.replace(".wav", "") for sid in df_orig["safe_id"]], rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig4_path = COMP_DIR / "fig4_target_language_confidence_comparison.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f"  [OK] Saved {fig4_path.name}")

if __name__ == "__main__":
    main()
