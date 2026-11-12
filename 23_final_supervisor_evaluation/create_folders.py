import os

base_dir = r"C:\pd-speech-crosslingual\23_final_supervisor_evaluation"

folders = [
    "00_audit",
    "01_config",
    "02_dataset_index",
    "03_preprocessing/scripts", "03_preprocessing/logs", "03_preprocessing/qc",
    "04_embeddings/scripts", "04_embeddings/xlsr", "04_embeddings/wav2vec2", "04_embeddings/wavlm", "04_embeddings/manifests", "04_embeddings/logs",
    "05_classification/scripts", "05_classification/configs", "05_classification/predictions", "05_classification/fold_results", "05_classification/summaries", "05_classification/statistics", "05_classification/logs",
    "06_layer_analysis/tables", "06_layer_analysis/figures", "06_layer_analysis/reports",
    "07_pca_tsne/scripts", "07_pca_tsne/coordinates", "07_pca_tsne/figures", "07_pca_tsne/quantitative_metrics", "07_pca_tsne/report",
    "08_speaker_similarity_ecapa/scripts", "08_speaker_similarity_ecapa/embeddings", "08_speaker_similarity_ecapa/results", "08_speaker_similarity_ecapa/figures", "08_speaker_similarity_ecapa/report",
    "09_visqol/installation", "09_visqol/scripts", "09_visqol/pair_lists", "09_visqol/results", "09_visqol/figures", "09_visqol/report",
    "10_mos_listening_test/scripts", "10_mos_listening_test/selected_samples", "10_mos_listening_test/anonymized_manifest", "10_mos_listening_test/rating_templates", "10_mos_listening_test/listening_test", "10_mos_listening_test/ratings", "10_mos_listening_test/results", "10_mos_listening_test/report",
    "11_integrated_quality/tables", "11_integrated_quality/figures", "11_integrated_quality/report",
    "12_master_comparison/tables", "12_master_comparison/figures", "12_master_comparison/statistics", "12_master_comparison/report",
    "13_thesis_ready/figures", "13_thesis_ready/tables", "13_thesis_ready/latex", "13_thesis_ready/captions", "13_thesis_ready/methods", "13_thesis_ready/results", "13_thesis_ready/discussion", "13_thesis_ready/limitations",
    "14_supervisor_package/figures", "14_supervisor_package/tables", "14_supervisor_package/executive_summary",
    "15_reproducibility"
]

for f in folders:
    os.makedirs(os.path.join(base_dir, f), exist_ok=True)
