import csv
import os

idx_file = "C:/pd-speech-crosslingual/23_final_supervisor_evaluation/13_thesis_ready/FIGURE_INDEX.csv"

rows_to_add = [
    ["Fig076", "visqol_samecontent_vs_crosslanguage.png", "Same-Content vs Cross-Language ViSQOL", "visqol_reconstruction_per_pair_OFFICIAL.csv", "Compare matched vs mismatched phonetic content evaluation", "Same-content MOS-LQO is high (~4.26) confirming vocoder integrity; mismatched is low (~1.18) confirming it functions only as a similarity score.", "V"],
    ["Fig077", "visqol_reconstruction_by_cohort.png", "Same-Content Reconstruction ViSQOL by Cohort", "visqol_reconstruction_per_pair_OFFICIAL.csv", "Assess vocoder stability across diseases and languages", "Vocoder integrity remains extremely high across all four subgroups.", "V"],
    ["Fig078", "visqol_crosslanguage_by_direction.png", "Exploratory Cross-Language ViSQOL by Direction", "visqol_crosslanguage_per_pair_OFFICIAL.csv", "Show similarity shifts depending on conversion direction", "GE->SP shows slightly higher similarity than SP->GE, matching classification patterns.", "V"],
    ["Fig079", "visqol_crosslanguage_by_direction_and_pathology.png", "Exploratory Cross-Language ViSQOL by Pathology", "visqol_crosslanguage_per_pair_OFFICIAL.csv", "Assess difference in VC processing of healthy vs PD", "Both groups exhibit similar massive drops in structural similarity when phonetic content mismatches.", "V"],
    ["Fig080", "visqol_crosslanguage_overall.png", "Overall Exploratory Cross-Language ViSQOL Distribution", "visqol_crosslanguage_per_pair_OFFICIAL.csv", "Show total distribution of exploratory scores", "Score distribution centers around 1.15.", "V"]
]

with open(idx_file, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for row in rows_to_add:
        writer.writerow(row)
print("Done appending")
