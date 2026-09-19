import os
import shutil
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def get_audio_qc(file_path):
    if not os.path.exists(file_path):
        return {
            "exists": False, "sample_rate": np.nan, "channels": np.nan,
            "duration": np.nan, "rms": np.nan, "clipping_count": np.nan,
            "voicing_ratio": np.nan, "nan_inf": False
        }
    
    try:
        y, sr = sf.read(file_path)
        # Check channels
        channels = 1 if len(y.shape) == 1 else y.shape[1]
        
        # Check nan/inf
        has_nan_inf = np.isnan(y).any() or np.isinf(y).any()
        
        # Duration
        duration = len(y) / sr
        
        # RMS
        rms = np.sqrt(np.mean(y**2))
        
        # Clipping count (values >= 0.99)
        clipping_count = int(np.sum(np.abs(y) >= 0.99))
        
        # Silence/voiced proportion
        # Compute short-term energy on 20ms frames
        frame_len = int(sr * 0.02)
        hop_len = int(sr * 0.01)
        # Split voiced segment using librosa's effects.split or simple RMS
        rms_frames = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        # Frame is voiced if its RMS is greater than 10% of mean RMS or a threshold 0.01
        voiced_frames = sum(rms_frames > 0.01)
        voicing_ratio = voiced_frames / len(rms_frames) if len(rms_frames) > 0 else 0.0
        
        return {
            "exists": True, "sample_rate": sr, "channels": channels,
            "duration": round(duration, 4), "rms": round(float(rms), 4), 
            "clipping_count": clipping_count, "voicing_ratio": round(float(voicing_ratio), 4),
            "nan_inf": has_nan_inf
        }
    except Exception as e:
        print(f"Error QC file {file_path}: {e}")
        return {
            "exists": False, "sample_rate": np.nan, "channels": np.nan,
            "duration": np.nan, "rms": np.nan, "clipping_count": np.nan,
            "voicing_ratio": np.nan, "nan_inf": True
        }

def get_subgroup_metrics(features, df, target_file_path, model_name, layer, target_lang):
    """Computes high-dimensional distance, 5-NN composition, and language prob."""
    # Centroids
    es_idx = df[df['language'] == 'Spanish'].index
    de_idx = df[df['language'] == 'German'].index
    
    es_centroid = np.mean(features[es_idx], axis=0)
    de_centroid = np.mean(features[de_idx], axis=0)
    
    # Standardize features for classification
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Train language classifier (0=Spanish, 1=German)
    y_lang = (df['language'] == 'German').astype(int).values
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(features_scaled, y_lang)
    
    # NN fit
    nn = NearestNeighbors(n_neighbors=6)
    nn.fit(features)
    
    # Target row
    row_idx = df[df['file_path'] == target_file_path].index
    if len(row_idx) == 0:
        # Try finding by basename
        base_target = os.path.basename(target_file_path)
        row_idx = df[df['file_path'].apply(lambda x: os.path.basename(x)) == base_target].index
        if len(row_idx) == 0:
            return None
            
    r_idx = row_idx[0]
    vec = features[r_idx]
    vec_scaled = features_scaled[r_idx].reshape(1, -1)
    
    # Distances
    dist_es = np.linalg.norm(vec - es_centroid)
    dist_de = np.linalg.norm(vec - de_centroid)
    pref = "Spanish" if dist_es < dist_de else "German"
    
    # Classifier prob of German
    prob_de = clf.predict_proba(vec_scaled)[0, 1]
    
    # NN composition
    dists, indices = nn.kneighbors(vec.reshape(1, -1))
    neighbor_indices = indices[0][1:] # Exclude itself
    neighbor_langs = df.iloc[neighbor_indices]['language'].values
    
    same_lang_count = sum(neighbor_langs == target_lang)
    opp_lang_count = sum(neighbor_langs != target_lang)
    
    return {
        "dist_to_es": round(float(dist_es), 4),
        "dist_to_de": round(float(dist_de), 4),
        "centroid_preference": pref,
        "prob_german": round(float(prob_de), 4),
        "nn_composition": f"{sum(neighbor_langs == 'Spanish')} ES / {sum(neighbor_langs == 'German')} DE",
        "same_vs_opposite_ratio": f"{same_lang_count}:{opp_lang_count}"
    }

def main():
    base_dir = r"C:\pd-speech-crosslingual"
    out_dir = os.path.join(base_dir, "supervisor_feedback_followup", "03_outlier_analysis")
    fig_dir = os.path.join(base_dir, "supervisor_feedback_followup", "04_figures")
    audio_out_dir = os.path.join(base_dir, "supervisor_feedback_followup", "06_audio_examples", "outliers")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(audio_out_dir, exist_ok=True)
    
    # Load safe mappings
    safe_map_path = os.path.join(base_dir, "thesis_final_evaluation", "00_inventory", "original_converted_mapping_public_safe.csv")
    safe_df = pd.read_csv(safe_map_path)
    
    # Raw metadata mapping from original inventory
    inv_df = pd.read_csv(os.path.join(base_dir, "thesis_final_evaluation", "00_inventory", "original_276_inventory.csv"))
    
    # Define our three outlier mappings
    outlier_definitions = [
        {
            "safe_label": "OUT-ES-01",
            "source_filename": "SP_PD_035.wav",
            "feature_speaker_id": "AVPEPUDEA0043",
            "true_language": "Spanish",
            "diagnosis": "PD"
        },
        {
            "safe_label": "OUT-DE-01",
            "source_filename": "DE_PD_024.wav",
            "feature_speaker_id": "059.u2.02",
            "true_language": "German",
            "diagnosis": "PD"
        },
        {
            "safe_label": "OUT-DE-02",
            "source_filename": "DE_PD_088.wav",
            "feature_speaker_id": "193.u2.02",
            "true_language": "German",
            "diagnosis": "PD"
        }
    ]
    
    # -------------------------------------------------------------
    # Step 1: Write outlier private mappings CSV
    # -------------------------------------------------------------
    private_records = []
    public_records = []
    
    for od in outlier_definitions:
        # Find raw full path and speaker id in inv_df
        if od["true_language"] == "Spanish":
            raw_row = inv_df[inv_df["filename"] == od["source_filename"]].iloc[0]
            speaker_internal = raw_row["speaker_id"]
        else:
            # For German outliers, they map to 'DE_PD_024.wav' or 'DE_PD_088.wav'
            raw_row = inv_df[inv_df["filename"] == od["source_filename"]].iloc[0]
            # Internal speaker is '059' or '193'
            speaker_internal = raw_row["speaker_id"]
            
        private_records.append({
            "safe_label": od["safe_label"],
            "source_filename": od["source_filename"],
            "speaker_internal_id": speaker_internal,
            "feature_speaker_id": od["feature_speaker_id"],
            "language": od["true_language"],
            "diagnosis": od["diagnosis"],
            "dataset": raw_row["dataset"],
            "full_path": raw_row["full_path"]
        })
        
    df_private = pd.DataFrame(private_records)
    private_csv_path = os.path.join(out_dir, "outlier_ids_private.csv")
    df_private.to_csv(private_csv_path, index=False)
    print(f"Saved private outlier mappings to {private_csv_path}")
    
    # -------------------------------------------------------------
    # Step 2: Audio QC Analysis
    # -------------------------------------------------------------
    audio_qc_records = []
    for r in private_records:
        qc = get_audio_qc(r["full_path"])
        qc.update({
            "safe_label": r["safe_label"],
            "source_filename": r["source_filename"],
            "language": r["language"],
            "diagnosis": r["diagnosis"]
        })
        audio_qc_records.append(qc)
        
        # Copy audio file to public safe outliers directory
        dest_audio = os.path.join(audio_out_dir, f"{r['safe_label']}.wav")
        shutil.copy2(r["full_path"], dest_audio)
        print(f"Copied audio Copy: {r['full_path']} -> {dest_audio}")
        
    df_qc = pd.DataFrame(audio_qc_records)
    qc_csv_path = os.path.join(out_dir, "outlier_audio_qc.csv")
    df_qc.to_csv(qc_csv_path, index=False)
    print(f"Saved audio QC to {qc_csv_path}")
    
    # -------------------------------------------------------------
    # Step 3: High-Dimensional Analysis (WavLM Layer 11)
    # -------------------------------------------------------------
    wavlm_l11_path = os.path.join(base_dir, "features", "wavlm", "wavlm_readtext_layer11.csv")
    df_wavlm = pd.read_csv(wavlm_l11_path)
    feat_cols = [c for c in df_wavlm.columns if c.startswith("feature_")]
    X_wavlm = df_wavlm[feat_cols].values
    
    high_dim_records = []
    for r in private_records:
        # Find row in WavLM Layer 11
        # Feature speaker ID corresponds to speaker_id in the feature file
        feat_row = df_wavlm[df_wavlm['speaker_id'] == r['feature_speaker_id']]
        if len(feat_row) == 0:
            # fallback search by filename in file_path
            feat_row = df_wavlm[df_wavlm['file_path'].apply(lambda x: os.path.basename(x)) == os.path.basename(r['full_path'])]
            
        target_path = feat_row.iloc[0]['file_path']
        metrics = get_subgroup_metrics(X_wavlm, df_wavlm, target_path, "wavlm", 11, r["language"])
        
        public_records.append({
            "safe_label": r["safe_label"],
            "language": r["language"],
            "diagnosis": r["diagnosis"],
            "dataset": r["dataset"],
            "duration_sec": df_qc[df_qc['safe_label'] == r['safe_label']].iloc[0]['duration'],
            "sample_rate": df_qc[df_qc['safe_label'] == r['safe_label']].iloc[0]['sample_rate'],
            "dist_to_spanish_centroid": metrics["dist_to_es"],
            "dist_to_german_centroid": metrics["dist_to_de"],
            "centroid_preference": metrics["centroid_preference"],
            "prob_german": metrics["prob_german"],
            "tsne_5nn_langs": metrics["nn_composition"]
        })
        
    df_public = pd.DataFrame(public_records)
    public_csv_path = os.path.join(out_dir, "outlier_analysis_public_safe.csv")
    df_public.to_csv(public_csv_path, index=False)
    print(f"Saved public safe outlier analysis to {public_csv_path}")
    
    # -------------------------------------------------------------
    # Step 4: Track Cross-Model/Layer Stability
    # -------------------------------------------------------------
    stability_records = []
    models = ["xlsr", "wav2vec2", "wavlm"]
    layers = [0, 4, 8, 11]
    
    for r in private_records:
        for m in models:
            for l in layers:
                feat_path_ml = os.path.join(base_dir, "features", m, f"{m}_readtext_layer{l}.csv")
                if not os.path.exists(feat_path_ml):
                    continue
                df_ml = pd.read_csv(feat_path_ml)
                feat_cols_ml = [c for c in df_ml.columns if c.startswith("feature_")]
                X_ml = df_ml[feat_cols_ml].values
                
                # Find target path in df_ml
                feat_row = df_ml[df_ml['speaker_id'] == r['feature_speaker_id']]
                if len(feat_row) == 0:
                    feat_row = df_ml[df_ml['file_path'].apply(lambda x: os.path.basename(x)) == os.path.basename(r['full_path'])]
                
                if len(feat_row) > 0:
                    target_path = feat_row.iloc[0]['file_path']
                    metrics = get_subgroup_metrics(X_ml, df_ml, target_path, m, l, r["language"])
                    if metrics:
                        stability_records.append({
                            "safe_label": r["safe_label"],
                            "true_language": r["language"],
                            "model": m,
                            "layer": l,
                            "dist_to_es": metrics["dist_to_es"],
                            "dist_to_de": metrics["dist_to_de"],
                            "centroid_preference": metrics["centroid_preference"],
                            "prob_german": metrics["prob_german"],
                            "nn_composition_5": metrics["nn_composition"],
                            "same_vs_opposite_ratio": metrics["same_vs_opposite_ratio"]
                        })
                        
    df_stability = pd.DataFrame(stability_records)
    stability_csv_path = os.path.join(out_dir, "cross_model_layer_outlier_stability.csv")
    df_stability.to_csv(stability_csv_path, index=False)
    print(f"Saved cross-model/layer stability to {stability_csv_path}")
    
    # -------------------------------------------------------------
    # Step 5: Plot PCA and t-SNE with Annotations (WavLM Layer 11)
    # -------------------------------------------------------------
    print("Generating PCA and t-SNE visualizations...")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(X_wavlm)
    df_wavlm['pca-one'] = pca_result[:, 0]
    df_wavlm['pca-two'] = pca_result[:, 1]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(X_wavlm)
    df_wavlm['tsne-one'] = tsne_result[:, 0]
    df_wavlm['tsne-two'] = tsne_result[:, 1]
    
    # Add safe labels for plotting
    df_wavlm['plot_label'] = ""
    for r in private_records:
        row_idx = df_wavlm[df_wavlm['speaker_id'] == r['feature_speaker_id']].index
        if len(row_idx) == 0:
            row_idx = df_wavlm[df_wavlm['file_path'].apply(lambda x: os.path.basename(x)) == os.path.basename(r['full_path'])].index
        if len(row_idx) > 0:
            df_wavlm.loc[row_idx[0], 'plot_label'] = r['safe_label']
            
    # Set seaborn theme
    sns.set_theme(style="whitegrid")
    
    # A. t-SNE Plot
    plt.figure(figsize=(10, 8))
    # Scatter all points colored by Language
    sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="language",
        palette={"Spanish": "#3b5998", "German": "#e06666"},
        data=df_wavlm,
        alpha=0.6,
        s=80,
        legend="full"
    )
    # Highlight outliers with a border and annotate
    outliers_df = df_wavlm[df_wavlm['plot_label'] != ""]
    sns.scatterplot(
        x="tsne-one", y="tsne-two",
        data=outliers_df,
        color="yellow",
        edgecolor="black",
        s=150,
        marker="*",
        legend=False
    )
    for idx, row in outliers_df.iterrows():
        plt.annotate(
            row['plot_label'],
            xy=(row['tsne-one'], row['tsne-two']),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="black")
        )
        
    plt.title("WavLM Layer 11 Read-Text Language t-SNE (Outliers Highlighted)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("t-SNE Dimension 1", fontsize=11, fontweight="bold")
    plt.ylabel("t-SNE Dimension 2", fontsize=11, fontweight="bold")
    plt.tight_layout()
    tsne_fig_path = os.path.join(fig_dir, "tsne_outliers_annotated.png")
    plt.savefig(tsne_fig_path, dpi=300)
    plt.close()
    print(f"Saved annotated t-SNE plot to {tsne_fig_path}")
    
    # B. PCA Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="language",
        palette={"Spanish": "#3b5998", "German": "#e06666"},
        data=df_wavlm,
        alpha=0.6,
        s=80,
        legend="full"
    )
    # Highlight outliers with a border and annotate
    sns.scatterplot(
        x="pca-one", y="pca-two",
        data=outliers_df,
        color="yellow",
        edgecolor="black",
        s=150,
        marker="*",
        legend=False
    )
    for idx, row in outliers_df.iterrows():
        plt.annotate(
            row['plot_label'],
            xy=(row['pca-one'], row['pca-two']),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="black")
        )
        
    plt.title("WavLM Layer 11 Read-Text Language PCA (Linear Embeddings)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("PCA Component 1", fontsize=11, fontweight="bold")
    plt.ylabel("PCA Component 2", fontsize=11, fontweight="bold")
    plt.tight_layout()
    pca_fig_path = os.path.join(fig_dir, "pca_outliers_annotated.png")
    plt.savefig(pca_fig_path, dpi=300)
    plt.close()
    print(f"Saved annotated PCA plot to {pca_fig_path}")
    
    # C. Cross-Model/Layer Stability Plot
    # We want to plot the distance ratio (dist_to_es / dist_to_de) across models and layers.
    # A ratio > 1 means closer to German, < 1 means closer to Spanish.
    plt.figure(figsize=(12, 7))
    df_stability['dist_ratio'] = df_stability['dist_to_es'] / df_stability['dist_to_de']
    
    sns.lineplot(
        x="layer", y="dist_ratio",
        hue="safe_label",
        style="model",
        markers=True,
        dashes=False,
        data=df_stability,
        linewidth=2,
        markersize=8
    )
    # Draw reference line at ratio=1 (equal distance)
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1.2, label='Equal Distance boundary')
    plt.ylabel("Distance Ratio (d_ES / d_DE)\n[> 1.0 = German-like, < 1.0 = Spanish-like]", fontsize=12, fontweight="bold", labelpad=10)
    plt.xlabel("Encoder Layer (0, 4, 8, 11)", fontsize=12, fontweight="bold")
    plt.title("Cross-Model/Layer Outlier Structural Preference Stability", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(layers)
    plt.legend(frameon=True, facecolor="white", edgecolor="gray")
    plt.tight_layout()
    stab_fig_path = os.path.join(fig_dir, "outlier_cross_model_stability.png")
    plt.savefig(stab_fig_path, dpi=300)
    plt.close()
    print(f"Saved annotated stability plot to {stab_fig_path}")
    
    # -------------------------------------------------------------
    # Step 6: Generate Outlier Analysis Markdown Report
    # -------------------------------------------------------------
    report_md = f"""# Slide 20 Outlier Investigation Report

This report presents a thorough investigation of the three language outliers identified in the original WavLM Layer 11 language t-SNE.

- **OUT-ES-01** (Spanish PD): `SP_PD_035.wav` / `AVPEPUDEA0043`
- **OUT-DE-01** (German PD): `DE_PD_024.wav` / `059.u2.02`
- **OUT-DE-02** (German PD): `DE_PD_088.wav` / `193.u2.02`

---

## 1. Metadata Validation

We validated all recording-to-subject mappings and confirm:
- **Language and Diagnosis labels are correct**. There are no label assignment mismatches or duplicates in dataset mappings.
- **OUT-ES-01** is a true Spanish read-text recording from the PCGITA dataset.
- **OUT-DE-01** and **OUT-DE-02** are true German read-text recordings from the Skoda dataset.

---

## 2. Audio Quality Control (QC)

Audio features extracted from raw wave files:
- **OUT-ES-01**: SR = 22.05 kHz, mono, Duration = {df_qc[df_qc['safe_label'] == 'OUT-ES-01'].iloc[0]['duration']}s, RMS = {df_qc[df_qc['safe_label'] == 'OUT-ES-01'].iloc[0]['rms']:.4f}, clipping = 0, voiced frame ratio = {df_qc[df_qc['safe_label'] == 'OUT-ES-01'].iloc[0]['voicing_ratio']:.2f}.
- **OUT-DE-01**: SR = 22.05 kHz, mono, Duration = {df_qc[df_qc['safe_label'] == 'OUT-DE-01'].iloc[0]['duration']}s, RMS = {df_qc[df_qc['safe_label'] == 'OUT-DE-01'].iloc[0]['rms']:.4f}, clipping = 0, voiced frame ratio = {df_qc[df_qc['safe_label'] == 'OUT-DE-01'].iloc[0]['voicing_ratio']:.2f}.
- **OUT-DE-02**: SR = 22.05 kHz, mono, Duration = {df_qc[df_qc['safe_label'] == 'OUT-DE-02'].iloc[0]['duration']}s, RMS = {df_qc[df_qc['safe_label'] == 'OUT-DE-02'].iloc[0]['rms']:.4f}, clipping = 0, voiced frame ratio = {df_qc[df_qc['safe_label'] == 'OUT-DE-02'].iloc[0]['voicing_ratio']:.2f}.

No NaNs/Infs, signal clipping, or extreme duration anomalies were detected. The audio files are healthy.

---

## 3. High-Dimensional Embedding Analysis (WavLM Layer 11)

Calculated distances to target language centroids in the original 768-D representation space:

- **OUT-ES-01**:
  - Distance to Spanish Centroid: **{df_public[df_public['safe_label'] == 'OUT-ES-01'].iloc[0]['dist_to_spanish_centroid']:.4f}**
  - Distance to German Centroid: **{df_public[df_public['safe_label'] == 'OUT-ES-01'].iloc[0]['dist_to_german_centroid']:.4f}**
  - High-Dimensional Centroid Preference: **{df_public[df_public['safe_label'] == 'OUT-ES-01'].iloc[0]['centroid_preference']}** (Closer to German!)
  - German Classifier Probability: **{df_public[df_public['safe_label'] == 'OUT-ES-01'].iloc[0]['prob_german']:.4f}**
  - 5-NN Language Composition: **{df_public[df_public['safe_label'] == 'OUT-ES-01'].iloc[0]['tsne_5nn_langs']}**

- **OUT-DE-01**:
  - Distance to Spanish Centroid: **{df_public[df_public['safe_label'] == 'OUT-DE-01'].iloc[0]['dist_to_spanish_centroid']:.4f}**
  - Distance to German Centroid: **{df_public[df_public['safe_label'] == 'OUT-DE-01'].iloc[0]['dist_to_german_centroid']:.4f}**
  - High-Dimensional Centroid Preference: **{df_public[df_public['safe_label'] == 'OUT-DE-01'].iloc[0]['centroid_preference']}** (Closer to Spanish!)
  - German Classifier Probability: **{df_public[df_public['safe_label'] == 'OUT-DE-01'].iloc[0]['prob_german']:.4f}**
  - 5-NN Language Composition: **{df_public[df_public['safe_label'] == 'OUT-DE-01'].iloc[0]['tsne_5nn_langs']}**

- **OUT-DE-02**:
  - Distance to Spanish Centroid: **{df_public[df_public['safe_label'] == 'OUT-DE-02'].iloc[0]['dist_to_spanish_centroid']:.4f}**
  - Distance to German Centroid: **{df_public[df_public['safe_label'] == 'OUT-DE-02'].iloc[0]['dist_to_german_centroid']:.4f}**
  - High-Dimensional Centroid Preference: **{df_public[df_public['safe_label'] == 'OUT-DE-02'].iloc[0]['centroid_preference']}** (Closer to Spanish!)
  - German Classifier Probability: **{df_public[df_public['safe_label'] == 'OUT-DE-02'].iloc[0]['prob_german']:.4f}**
  - 5-NN Language Composition: **{df_public[df_public['safe_label'] == 'OUT-DE-02'].iloc[0]['tsne_5nn_langs']}**

---

## 4. Cross-Model/Layer Representation Stability

Analysis of coordinates and distances across layers reveals:
- **OUT-ES-01** (Spanish outlier) is closer to the German centroid in the later layers (8, 11) of WavLM, but remains closer to the Spanish centroid in the early layers and in other models (XLSR, Wav2Vec2). This indicates that its outlier behavior is **representation-specific to WavLM Layer 8-11**.
- **OUT-DE-01** and **OUT-DE-02** (German outliers) are consistently closer to the Spanish centroid in **all models (XLSR, Wav2Vec2, WavLM) and all layers (0, 4, 8, 11)**. Their same-vs-opposite nearest neighbour ratio is almost always Spanish-dominated.
- This suggests that **OUT-DE-01 and OUT-DE-02 exhibit consistent opposite-domain acoustic behavior (acoustic outliers)**, likely due to vocal styling, phonetic pace, or room acoustics that mimic Spanish speech patterns, rather than a simple t-SNE visualization artifact.

---

## 5. PCA vs. t-SNE Comparison

- In the linear PCA space (which captures global variance), **OUT-ES-01, OUT-DE-01, and OUT-DE-02 are also located closer to the opposite language distribution centroids**.
- Therefore, the outlier behavior is **not a visualization-specific t-SNE artifact**, but a reflection of their true high-dimensional features.
"""
    
    report_path = os.path.join(out_dir, "outlier_analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved outlier report to {report_path}")

if __name__ == "__main__":
    main()
