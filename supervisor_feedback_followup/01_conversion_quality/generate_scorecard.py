import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Directories
    base_dir = r"C:\pd-speech-crosslingual"
    out_dir = os.path.join(base_dir, "supervisor_feedback_followup", "01_conversion_quality")
    fig_dir = os.path.join(base_dir, "supervisor_feedback_followup", "04_figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    # Scorecard rows matching requested schema
    scorecard_data = [
        {
            "Evaluation dimension": "Speaker preservation",
            "Metric": "ECAPA cosine similarity",
            "Overall": "0.9648",
            "Spanish\u2192German": "0.9599",
            "German\u2192Spanish": "0.9675",
            "Interpretation": "Very high speaker identity retention; German\u2192Spanish slightly outperforms Spanish\u2192German."
        },
        {
            "Evaluation dimension": "Spectral change",
            "Metric": "MCD-DTW (dB)",
            "Overall": "5.4381",
            "Spanish\u2192German": "4.2387",
            "German\u2192Spanish": "6.1196",
            "Interpretation": "Successful spectral envelope adaptation; Spanish\u2192German shows less cepstral distortion."
        },
        {
            "Evaluation dimension": "Spectral change",
            "Metric": "LSD (dB)",
            "Overall": "17.2291",
            "Spanish\u2192German": "17.6560",
            "German\u2192Spanish": "16.9865",
            "Interpretation": "Moderate overall log-spectral distance change; German\u2192Spanish has lower global distortion."
        },
        {
            "Evaluation dimension": "Spectral change",
            "Metric": "Spectral convergence",
            "Overall": "0.7291",
            "Spanish\u2192German": "1.1228",
            "German\u2192Spanish": "0.5054",
            "Interpretation": "Spanish\u2192German exhibits larger deviation in reconstruction convergence than German\u2192Spanish."
        },
        {
            "Evaluation dimension": "Content consistency",
            "Metric": "WER proxy",
            "Overall": "0.2178",
            "Spanish\u2192German": "0.2588",
            "German\u2192Spanish": "0.1945",
            "Interpretation": "Moderate word-level degradation; German\u2192Spanish preserves verbal content better."
        },
        {
            "Evaluation dimension": "Content consistency",
            "Metric": "CER proxy",
            "Overall": "0.1035",
            "Spanish\u2192German": "0.1079",
            "German\u2192Spanish": "0.1010",
            "Interpretation": "Low character-level degradation (~10% error), very consistent across directions."
        },
        {
            "Evaluation dimension": "Content consistency",
            "Metric": "WIP",
            "Overall": "0.6924",
            "Spanish\u2192German": "0.6208",
            "German\u2192Spanish": "0.7331",
            "Interpretation": "Good information preservation; German\u2192Spanish retains more lexical information."
        },
        {
            "Evaluation dimension": "Target-domain movement",
            "Metric": "% acoustic features moving target-ward",
            "Overall": "83.7%",
            "Spanish\u2192German": "86.0%",
            "German\u2192Spanish": "81.4%",
            "Interpretation": "Robust acoustic envelope shift toward the target domain."
        },
        {
            "Evaluation dimension": "Pathology preservation",
            "Metric": "pathology sign preservation",
            "Overall": "69.2%",
            "Spanish\u2192German": "75.0%",
            "German\u2192Spanish": "69.2%",
            "Interpretation": "Moderate preservation of pathological separation signs."
        },
        {
            "Evaluation dimension": "Clinical utility",
            "Metric": "cross-language UAR original\u2192converted",
            "Overall": "0.6021 \u2192 0.5663",
            "Spanish\u2192German": "0.5992 \u2192 0.5575",
            "German\u2192Spanish": "0.6050 \u2192 0.5750",
            "Interpretation": "Slight decrease in zero-shot cross-language diagnostic generalization; utility is slightly degraded."
        }
    ]
    
    # Save CSV
    df = pd.DataFrame(scorecard_data)
    csv_path = os.path.join(out_dir, "conversion_quality_scorecard.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Saved CSV scorecard to {csv_path}")
    
    # Save Markdown table
    md_path = os.path.join(out_dir, "conversion_quality_scorecard.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Conversion Quality Scorecard\n\n")
        f.write("| Evaluation dimension | Metric | Overall | Spanish\u2192German | German\u2192Spanish | Interpretation |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for row in scorecard_data:
            f.write(f"| {row['Evaluation dimension']} | {row['Metric']} | {row['Overall']} | {row['Spanish\u2192German']} | {row['German\u2192Spanish']} | {row['Interpretation']} |\n")
    print(f"Saved MD scorecard to {md_path}")
    
    # Generate Interpretation Document
    interpretation_md = """# Scientific Interpretation of Conversion Quality vs Pathology Preservation

This report presents a clinical and acoustic evaluation of the voice conversion pipeline, distinguishing between speaker adaptation, linguistic preservation, target domain alignment, and pathology preservation.

## Key Scientific Findings

1. **Conversion Quality vs. Target-Domain Movement**:
   - **Target-domain movement** is high, with **86.0% (Spanish\u2192German)** and **81.4% (German\u2192Spanish)** of acoustic features shifting closer to the target domain.
   - However, **conversion quality** (measured by ASR content consistency and speaker similarity) behaves asymmetrically. German\u2192Spanish preserves content better (WER proxy 0.1945 vs. 0.2588) and speaker similarity (0.9675 vs. 0.9599), despite having higher cepstral distortion (MCD-DTW 6.12 dB vs. 4.24 dB).
   - This proves that **larger acoustic shifts (target-ward movement) do not automatically guarantee better conversion quality or speech naturalness**.

2. **Pathology Preservation vs. Diagnostic Generalization (Clinical Utility)**:
   - Voice conversion retains pathological signs moderately (**75.0% Spanish**, **69.2% German** sign preservation).
   - Yet, cross-language clinical diagnostic utility drops slightly (from an original baseline of **0.6021 UAR to 0.5663 UAR**).
   - This suggests that global voice conversion attenuates or distorts subtle, fine-grained dysarthric features, indicating that **voice conversion is not a silver bullet for zero-shot clinical generalization**.

3. **Core Conclusion**:
   $$\\text{conversion quality} \\neq \\text{target-domain movement} \\neq \\text{pathology preservation} \\neq \\text{classification improvement}$$
   Each of these represents a distinct dimension of the transformation:
   - *Target-domain movement* measures macro-acoustic shift.
   - *Conversion quality* measures micro-acoustic naturalness, speaker preservation, and content preservation.
   - *Pathology preservation* checks the survival of clinical disease markers.
   - *Clinical utility / Classification* checks the zero-shot generalization of diagnostic systems.
"""
    
    interp_path = os.path.join(out_dir, "conversion_quality_interpretation.md")
    with open(interp_path, "w", encoding="utf-8") as f:
        f.write(interpretation_md)
    print(f"Saved Interpretation report to {interp_path}")
    
    # Generate Publication Figure
    # We will plot normalized version of 5 key metrics:
    # 1. Speaker Verification (Cosine Similarity)
    # 2. Content Retention (1 - WER Proxy)
    # 3. Feature Movement Rate (Fraction)
    # 4. Pathology Sign Preservation (Fraction)
    # 5. Converted Diagnostic UAR
    
    metrics = ["Speaker Preservation\n(Cosine Sim)", "Content Retention\n(1 - WER)", "Feature Movement\n(Rate closer)", "Pathology Retention\n(Sign pres rate)", "Clinical Utility\n(Converted UAR)"]
    sp_vals = [0.9599, 1 - 0.2588, 0.860, 0.750, 0.5575]
    ge_vals = [0.9675, 1 - 0.1945, 0.814, 0.692, 0.5750]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Set seaborn style for clean academic presentation
    sns.set_theme(style="whitegrid")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, sp_vals, width, label="Spanish \u2192 German-domain", color="#3b5998", alpha=0.85, edgecolor="black", linewidth=1.2)
    rects2 = ax.bar(x + width/2, ge_vals, width, label="German \u2192 Spanish-domain", color="#e06666", alpha=0.85, edgecolor="black", linewidth=1.2)
    
    ax.set_ylabel("Normalized Value / Rate", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Voice Conversion Scorecard - Subgroup Dimensional Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, facecolor="white", edgecolor="gray", loc="upper right")
    
    # Attach labels above bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight="bold")
            
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "conversion_quality_summary.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved Scorecard Plot to {fig_path}")

if __name__ == "__main__":
    main()
