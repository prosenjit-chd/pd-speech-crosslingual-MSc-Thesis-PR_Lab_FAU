import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

base_dir = "/mnt/c/pd-speech-crosslingual/23_final_supervisor_evaluation"
paired_csv = os.path.join(base_dir, "02_dataset_index", "paired_original_converted_index.csv")
seeds_yaml = os.path.join(base_dir, "01_config", "random_seeds.yaml")

with open(seeds_yaml, 'r') as f:
    seeds = yaml.safe_load(f)

RANDOM_STATE = seeds.get('tsne', 42)

def compute_pca_tsne():
    df_meta = pd.read_csv(paired_csv)
    
    models = ["xlsr"]
    layers = [11]
    
    for model in models:
        for layer in layers:
            print(f"Running PCA/t-SNE for {model} layer {layer}")
            orig_file = os.path.join(base_dir, "04_embeddings", model, f"{model}_layer{layer}_original.csv")
            conv_file = os.path.join(base_dir, "04_embeddings", model, f"{model}_layer{layer}_converted.csv")
            
            df_orig = pd.read_csv(orig_file)
            df_conv = pd.read_csv(conv_file)
            
            # Reorder to match meta
            merged_orig = pd.merge(df_meta, df_orig, on='sample_id')
            merged_conv = pd.merge(df_meta, df_conv, on='sample_id')
            
            feat_cols = [c for c in merged_orig.columns if c.startswith('feature_')]
            
            X_orig = merged_orig[feat_cols].values
            X_conv = merged_conv[feat_cols].values
            
            X_combined = np.vstack([X_orig, X_conv])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_combined)
            
            pca = PCA(n_components=2, random_state=RANDOM_STATE)
            X_pca = pca.fit_transform(X_scaled)
            
            tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
            X_tsne = tsne.fit_transform(X_scaled)
            
            n = len(X_orig)
            
            # Combine back into a dataframe for plotting
            plot_df = pd.DataFrame({
                'sample_id': list(merged_orig['sample_id']) + list(merged_conv['sample_id']),
                'diagnosis': list(merged_orig['diagnosis']) + list(merged_conv['diagnosis']),
                'speech_condition': ['Original'] * n + ['Converted'] * n,
                'source_language': list(merged_orig['source_language']) + list(merged_conv['source_language']),
                'target_language': list(merged_orig['target_language']) + list(merged_conv['target_language']),
                'pca_1': X_pca[:, 0],
                'pca_2': X_pca[:, 1],
                'tsne_1': X_tsne[:, 0],
                'tsne_2': X_tsne[:, 1]
            })
            
            # Define Acoustic Language
            def get_acoustic(row):
                if row['speech_condition'] == 'Original':
                    return row['source_language']
                else:
                    return row['target_language']
            plot_df['acoustic_language'] = plot_df.apply(get_acoustic, axis=1)
            
            coord_dir = os.path.join(base_dir, "07_pca_tsne", "coordinates")
            os.makedirs(coord_dir, exist_ok=True)
            plot_df.to_csv(os.path.join(coord_dir, f"{model}_layer{layer}_coordinates.csv"), index=False)
            
            # Plotting
            fig_dir = os.path.join(base_dir, "07_pca_tsne", "figures")
            os.makedirs(fig_dir, exist_ok=True)
            
            # A. language/domain PCA
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=plot_df, x='pca_1', y='pca_2', hue='acoustic_language', style='speech_condition', alpha=0.7)
            plt.title(f"{model.upper()} Layer {layer} - Acoustic Language PCA")
            plt.savefig(os.path.join(fig_dir, "A_language_domain_PCA.png"), dpi=300)
            plt.close()
            
            # B. language/domain t-SNE
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=plot_df, x='tsne_1', y='tsne_2', hue='acoustic_language', style='speech_condition', alpha=0.7)
            plt.title(f"{model.upper()} Layer {layer} - Acoustic Language t-SNE")
            plt.savefig(os.path.join(fig_dir, "B_language_domain_tSNE.png"), dpi=300)
            plt.close()
            
            # C. Original vs Converted PCA
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=plot_df, x='pca_1', y='pca_2', hue='speech_condition', alpha=0.7)
            plt.title("Original vs Converted PCA")
            plt.savefig(os.path.join(fig_dir, "C_original_vs_converted_PCA.png"), dpi=300)
            plt.close()

            # D. Original vs Converted t-SNE
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=plot_df, x='tsne_1', y='tsne_2', hue='speech_condition', alpha=0.7)
            plt.title("Original vs Converted t-SNE")
            plt.savefig(os.path.join(fig_dir, "D_original_vs_converted_tSNE.png"), dpi=300)
            plt.close()
            
            # G. PD/HC coloured PCA
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=plot_df, x='pca_1', y='pca_2', hue='diagnosis', style='speech_condition', alpha=0.7)
            plt.title("PD vs HC PCA")
            plt.savefig(os.path.join(fig_dir, "G_pd_hc_PCA.png"), dpi=300)
            plt.close()
            
            # H. PD/HC coloured t-SNE
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=plot_df, x='tsne_1', y='tsne_2', hue='diagnosis', style='speech_condition', alpha=0.7)
            plt.title("PD vs HC t-SNE")
            plt.savefig(os.path.join(fig_dir, "H_pd_hc_tSNE.png"), dpi=300)
            plt.close()
            
            print(f"Finished PCA/t-SNE for {model} layer {layer}")

if __name__ == "__main__":
    compute_pca_tsne()
