import pandas as pd
import soundfile as sf
import os

input_manifest_path = r"C:\pd-speech-crosslingual\true_cross_language_vc\22_full_276_evaluation\06_synthesis\synthesis_manifest_276.csv"
inventory_path = r"C:\pd-speech-crosslingual\true_cross_language_vc\22_full_276_evaluation\01_inventory\full_276_inventory.csv"
output_csv = r"C:\pd-speech-crosslingual\23_final_supervisor_evaluation\02_dataset_index\paired_original_converted_index.csv"
class_balance_csv = r"C:\pd-speech-crosslingual\23_final_supervisor_evaluation\02_dataset_index\class_balance.csv"

manifest = pd.read_csv(input_manifest_path)
inventory = pd.read_csv(inventory_path)

# Merge inventory to get speaker_id and dataset
merged = pd.merge(manifest, inventory[['safe_id', 'speaker_id', 'dataset']], on='safe_id', how='left')

records = []
for idx, row in merged.iterrows():
    orig_path = row['source_path']
    conv_path = row['output_path']
    
    # get sample rate and duration
    try:
        orig_info = sf.info(orig_path)
        orig_sr = orig_info.samplerate
        orig_dur = orig_info.duration
    except Exception as e:
        orig_sr = None
        orig_dur = None
        print(f"Error reading {orig_path}: {e}")
        
    try:
        conv_info = sf.info(conv_path)
        conv_sr = conv_info.samplerate
        conv_dur = conv_info.duration
    except Exception as e:
        conv_sr = None
        conv_dur = None
        print(f"Error reading {conv_path}: {e}")

    conversion_direction = f"{row['source_language'][:2].upper()}_to_{row['target_language'][:2].upper()}"

    records.append({
        'sample_id': row['safe_id'],
        'speaker_id': row['speaker_id'],
        'source_language': row['source_language'],
        'target_language': row['target_language'],
        'diagnosis': row['diagnosis'],
        'original_audio_path': orig_path,
        'converted_audio_path': conv_path,
        'conversion_direction': conversion_direction,
        'dataset': row['dataset'],
        'development_subject': True if row['cohort_tag'] == 'PILOT_DEVELOPMENT' else False,
        'original_sample_rate': orig_sr,
        'converted_sample_rate': conv_sr,
        'duration_original': orig_dur,
        'duration_converted': conv_dur
    })

df_paired = pd.DataFrame(records)
df_paired.to_csv(output_csv, index=False)
print(f"Saved paired index to {output_csv}")

# Class balance table
summary = df_paired.groupby(['source_language', 'target_language', 'conversion_direction', 'diagnosis']).size().reset_index(name='count')
summary.to_csv(class_balance_csv, index=False)
print(f"Saved class balance to {class_balance_csv}")
