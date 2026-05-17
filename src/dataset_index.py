import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def normalize_label(label_str):
    """
    Normalizes dataset labels to standard 'PD' and 'HC'.
    Handles variations like 'pd', 'hc', 'control', 'ips'.
    """
    label_str = label_str.lower()
    if label_str in ['pd', 'ips']:
        return 'PD'
    elif label_str in ['hc', 'control']:
        return 'HC'
    return None

def infer_speaker_id(filename, language):
    """
    Attempts to infer the speaker ID from the filename.
    Extracts the prefix before any underscores or hyphens.
    """
    base_name = os.path.splitext(filename)[0]
    
    # Split by common delimiters and take the first part
    for delimiter in ['_', '-']:
        if delimiter in base_name:
            return base_name.split(delimiter)[0]
            
    return base_name

def create_dataset_index(input_dir, target_task="readtext"):
    """
    Scans the input directory and creates a dataset index dataframe.
    """
    records = []
    
    # Expected structure: input/{Language}/{Task}/... OR input/{Language}/...
    for language in ['Spanish', 'German']:
        lang_dir = os.path.join(input_dir, language)
        if not os.path.exists(lang_dir):
            logger.warning(f"Language directory not found: {lang_dir}")
            continue
            
        # Walk through the language directory looking for audio files
        for root, _, files in os.walk(lang_dir):
            # Check if the current directory path contains the target task
            # Or if the files themselves might belong to the task
            if target_task.lower() not in root.lower():
                # Some datasets might not have a task folder, but task in filename
                pass
            
            for file in files:
                if not file.lower().endswith(('.wav', '.mp3', '.flac')):
                    continue
                    
                # To be strict about the readtext task, we check if the path or filename contains 'readtext'
                if target_task.lower() not in root.lower() and target_task.lower() not in file.lower():
                    continue

                file_path = os.path.join(root, file)
                
                # Try to infer label from filename or parent folder
                label = None
                parent_dir = os.path.basename(root)
                grandparent_dir = os.path.basename(os.path.dirname(root))
                
                # Check filename
                for l in ['PD', 'HC', 'control', 'ips']:
                    if l.lower() in file.lower().split('_') or l.lower() in file.lower().split('-'):
                        label = normalize_label(l)
                        break
                
                # Check parent dirs if not found
                if not label:
                    for l in ['PD', 'HC', 'control', 'ips']:
                        if l.lower() in parent_dir.lower() or l.lower() in grandparent_dir.lower():
                            label = normalize_label(l)
                            break
                            
                if not label:
                    logger.warning(f"Could not infer label for {file_path}. Skipping.")
                    continue
                
                speaker_id = infer_speaker_id(file, language)
                dataset_name = "PCGITA" if language == "Spanish" else "German_Sabine_Skoda"
                
                records.append({
                    "file_path": file_path,
                    "speaker_id": speaker_id,
                    "language": language,
                    "task": target_task,
                    "label": label,
                    "dataset": dataset_name
                })
                
    df = pd.DataFrame(records)
    if len(df) == 0:
        logger.error("Dataset index is empty! Please check your input folder structure.")
    else:
        logger.info(f"Found {len(df)} {target_task} recordings.")
        
    return df
