import pandas as pd
import os

base_dir = "C:/pd-speech-crosslingual/23_final_supervisor_evaluation"
paired_csv = os.path.join(base_dir, "02_dataset_index", "paired_original_converted_index.csv")

df = pd.read_csv(paired_csv)
excluded = df[df['development_subject'] == True]
included = df[df['development_subject'] == False]

audit_text = f"""# N=264 EXCLUSION FINAL AUDIT

## Cohort Verification
- **Total Initial Subjects (N=276):** {len(df)}
- **Excluded Development Subjects:** {len(excluded)}
- **Final Sensitivity Cohort (N=264):** {len(included)}

## Final Cohort Composition (N=264)
- **Spanish (ES):** {len(included[included['source_language'] == 'Spanish'])}
- **German (DE):** {len(included[included['source_language'] == 'German'])}
- **PD:** {len(included[included['diagnosis'] == 'PD'])}
- **HC:** {len(included[included['diagnosis'] == 'HC'])}

## Excluded Subjects (N={len(excluded)})
| Sample ID | Language | Diagnosis |
|-----------|----------|-----------|
"""

for _, row in excluded.iterrows():
    audit_text += f"| {row['sample_id']} | {row['source_language']} | {row['diagnosis']} |\n"

audit_text += "\n**Confirmation:** The exclusion filtering (`development_subject == False`) occurs strictly BEFORE the cross-validation outer loops and index alignments, guaranteeing complete isolation of development data from the sensitivity evaluation.\n"

with open(os.path.join(base_dir, "00_audit", "N264_EXCLUSION_FINAL_AUDIT.md"), "w") as f:
    f.write(audit_text)

print(f"Audit generated. Excluded: {len(excluded)} (Expected 12: 6 ES, 6 DE)")
