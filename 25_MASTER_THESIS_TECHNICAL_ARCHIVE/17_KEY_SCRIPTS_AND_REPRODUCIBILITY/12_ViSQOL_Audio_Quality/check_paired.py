import pandas as pd
import os

base_dir = "C:/pd-speech-crosslingual/23_final_supervisor_evaluation"
paired_csv = os.path.join(base_dir, "02_dataset_index", "paired_original_converted_index.csv")

df = pd.read_csv(paired_csv)
print(df.columns)
print(df.head(2).to_dict(orient='records'))
