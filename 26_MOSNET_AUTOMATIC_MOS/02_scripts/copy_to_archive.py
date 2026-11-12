import os
import shutil

src_base = r"C:\pd-speech-crosslingual\26_MOSNET_AUTOMATIC_MOS"
dst = r"C:\pd-speech-crosslingual\25_MASTER_THESIS_TECHNICAL_ARCHIVE\12_VISQOL_AND_AUDIO_QUALITY\MOSNET"
os.makedirs(dst, exist_ok=True)

files = [
    (r"06_comparison_tables\MOSNET_SUMMARY_BY_CONDITION.csv", "MOSNET_SUMMARY_BY_CONDITION.csv"),
    (r"06_comparison_tables\MOSNET_PAIRED_COMPARISON.csv", "MOSNET_PAIRED_COMPARISON.csv"),
    (r"06_comparison_tables\MOSNET_PAIRED_STATISTICS.csv", "MOSNET_PAIRED_STATISTICS.csv"),
    (r"03_original_speech_results\MOSNET_ORIGINAL_N276.csv", "MOSNET_ORIGINAL_N276.csv"),
    (r"04_converted_speech_results\MOSNET_XTTS_N276.csv", "MOSNET_XTTS_N276.csv"),
    (r"05_hifigan_reconstruction_results\MOSNET_HIFIGAN_N276.csv", "MOSNET_HIFIGAN_N276.csv"),
    (r"07_figures\F_MOSNET_CONDITION_COMPARISON.png", "F_MOSNET_CONDITION_COMPARISON.png"),
    (r"07_figures\F_MOSNET_PAIRED_ORIGINAL_VS_XTTS.png", "F_MOSNET_PAIRED_ORIGINAL_VS_XTTS.png"),
    (r"07_figures\F_MOSNET_HIFIGAN_COMPARISON.png", "F_MOSNET_HIFIGAN_COMPARISON.png"),
    (r"08_thesis_ready\MOSNET_METHOD.md", "MOSNET_METHOD.md"),
    (r"08_thesis_ready\MOSNET_RESULTS.md", "MOSNET_RESULTS.md"),
    (r"08_thesis_ready\MOSNET_VS_VISQOL_INTERPRETATION.md", "MOSNET_VS_VISQOL_INTERPRETATION.md"),
    (r"09_audit\MOSNET_FINAL_AUDIT.md", "MOSNET_FINAL_AUDIT.md"),
    (r"09_audit\MOSNET_MODEL_VALIDATION.md", "MOSNET_MODEL_VALIDATION.md")
]

for s, d in files:
    sp = os.path.join(src_base, s)
    dp = os.path.join(dst, d)
    shutil.copy2(sp, dp)
    print(f"Copied: {d} ({os.path.getsize(dp):,} bytes)")

print("All files successfully copied to master technical archive.")
