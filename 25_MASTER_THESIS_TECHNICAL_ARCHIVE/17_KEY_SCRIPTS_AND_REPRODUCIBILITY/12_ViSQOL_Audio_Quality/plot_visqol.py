import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = "C:/pd-speech-crosslingual/23_final_supervisor_evaluation"
results_dir = os.path.join(base_dir, "09_visqol", "results")
figures_dir = os.path.join(base_dir, "09_visqol", "figures")
os.makedirs(figures_dir, exist_ok=True)

df_res = pd.read_csv(os.path.join(results_dir, "visqol_reconstruction_per_pair_OFFICIAL.csv"))
df_cross = pd.read_csv(os.path.join(results_dir, "visqol_crosslanguage_per_pair_OFFICIAL.csv"))

# Figure 1: Same-Content vs Cross-Language
plt.figure(figsize=(10, 6))
df_res_plot = df_res.copy()
df_res_plot['Setting'] = 'Same-Content Reconstruction\n(Matched Linguistic Content)'
df_cross_plot = df_cross.copy()
df_cross_plot['Setting'] = 'Exploratory Cross-Language\n(Mismatched Linguistic Content)'

df_plot = pd.concat([
    df_res_plot[['Setting', 'visqol_mos_lqo']],
    df_cross_plot[['Setting', 'visqol_mos_lqo']]
])
df_plot['visqol_mos_lqo'] = pd.to_numeric(df_plot['visqol_mos_lqo'])

sns.violinplot(data=df_plot, x='Setting', y='visqol_mos_lqo', hue='Setting', inner='quartile', palette='Set2', legend=False)
plt.title('ViSQOL MOS-LQO: Conventional vs Exploratory Application', fontsize=14)
plt.ylabel('ViSQOL MOS-LQO Score')
plt.xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "visqol_samecontent_vs_crosslanguage.png"), dpi=300)
plt.close()

# Figure 2: Same-Content Reconstruction Distribution (by language and pathology)
plt.figure(figsize=(10, 6))
df_res_plot['Group'] = df_res_plot['source_language'] + " " + df_res_plot['pd_hc']
df_res_plot['visqol_mos_lqo'] = pd.to_numeric(df_res_plot['visqol_mos_lqo'])
sns.boxplot(data=df_res_plot, x='Group', y='visqol_mos_lqo', hue='Group', palette='viridis', legend=False)
plt.title('Same-Content Reconstruction ViSQOL by Cohort', fontsize=14)
plt.ylabel('ViSQOL MOS-LQO Score')
plt.xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "visqol_reconstruction_by_cohort.png"), dpi=300)
plt.close()

# Figure 3: Cross-language ViSQOL distribution (by direction)
plt.figure(figsize=(8, 6))
df_cross_plot['visqol_mos_lqo'] = pd.to_numeric(df_cross_plot['visqol_mos_lqo'])
sns.boxplot(data=df_cross_plot, x='direction', y='visqol_mos_lqo', hue='direction', palette='mako', legend=False)
plt.title('Exploratory Cross-Language ViSQOL by Direction', fontsize=14)
plt.ylabel('ViSQOL MOS-LQO Score')
plt.xlabel('Conversion Direction')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "visqol_crosslanguage_by_direction.png"), dpi=300)
plt.close()

# Figure 4: Cross-language ViSQOL distribution (by direction and pathology)
plt.figure(figsize=(10, 6))
df_cross_plot['visqol_mos_lqo'] = pd.to_numeric(df_cross_plot['visqol_mos_lqo'])
sns.boxplot(data=df_cross_plot, x='direction', y='visqol_mos_lqo', hue='pd_hc', palette='magma')
plt.title('Exploratory Cross-Language ViSQOL by Direction and Pathology', fontsize=14)
plt.ylabel('ViSQOL MOS-LQO Score')
plt.xlabel('Conversion Direction')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "visqol_crosslanguage_by_direction_and_pathology.png"), dpi=300)
plt.close()

# Figure 5: Overall Cross-language ViSQOL distribution
plt.figure(figsize=(6, 6))
sns.histplot(data=df_cross_plot, x='visqol_mos_lqo', kde=True, color='purple', bins=30)
plt.title('Overall Exploratory Cross-Language ViSQOL Distribution', fontsize=14)
plt.xlabel('ViSQOL MOS-LQO Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "visqol_crosslanguage_overall.png"), dpi=300)
plt.close()

print("Figures generated!")
