import os
import pandas as pd
import numpy as np
import shutil

base_dir = r"C:\pd-speech-crosslingual\23_final_supervisor_evaluation"
paired_csv = os.path.join(base_dir, "02_dataset_index", "paired_original_converted_index.csv")
mos_dir = os.path.join(base_dir, "10_mos_listening_test")

def prepare_mos():
    df = pd.read_csv(paired_csv)
    
    # We want ~40 samples: 20 ES->DE, 20 DE->ES (half PD, half HC)
    rng = np.random.default_rng(42)
    
    groups = df.groupby(['conversion_direction', 'diagnosis'])
    sampled = groups.apply(lambda x: x.sample(10, random_state=rng)).reset_index(drop=True)
    
    # Shuffle
    sampled = sampled.sample(frac=1, random_state=rng).reset_index(drop=True)
    
    # Assign anonymous IDs
    sampled['anonymous_id'] = [f"A{str(i).zfill(3)}" for i in range(1, len(sampled)+1)]
    
    # Save manifest private
    sampled.to_csv(os.path.join(mos_dir, "anonymized_manifest", "mos_sample_manifest_private.csv"), index=False)
    
    # Save manifest blinded
    blinded = sampled[['anonymous_id']]
    blinded.to_csv(os.path.join(mos_dir, "anonymized_manifest", "mos_sample_manifest_blinded.csv"), index=False)
    
    # Copy samples to selected_samples
    samples_dir = os.path.join(mos_dir, "selected_samples")
    os.makedirs(samples_dir, exist_ok=True)
    for _, row in sampled.iterrows():
        ext = os.path.splitext(row['converted_audio_path'])[1]
        dest = os.path.join(samples_dir, f"{row['anonymous_id']}{ext}")
        shutil.copy2(row['converted_audio_path'], dest)
        
    # Create rating template
    template_rows = []
    # Assumes 5 listeners for template
    for listener in range(1, 6):
        for _, row in sampled.iterrows():
            template_rows.append({
                'rater_id': f"Listener_{listener}",
                'sample_id': row['anonymous_id'],
                'rating': '',
                'comment_optional': ''
            })
    pd.DataFrame(template_rows).to_csv(os.path.join(mos_dir, "rating_templates", "mos_rating_sheet.csv"), index=False)
    
    # Initial status file
    with open(os.path.join(mos_dir, "results", "mos_overall.csv"), "w") as f:
        f.write("status,value\nPENDING HUMAN RATINGS,NaN\n")

    # Simple HTML interface
    html = """<!DOCTYPE html>
<html>
<head><title>MOS Listening Test</title></head>
<body>
    <h1>Speech Quality Evaluation</h1>
    <p>Rate the overall naturalness / speech quality of this audio (1=Bad, 5=Excellent).</p>
    <!-- Placeholder for JS dynamically loading the audio files -->
    <div id="test-container"></div>
    <script>
        const samples = """ + str(sampled['anonymous_id'].tolist()) + """;
        let container = document.getElementById('test-container');
        samples.forEach(id => {
            let div = document.createElement('div');
            div.innerHTML = `<h3>Sample ${id}</h3><audio controls src="../selected_samples/${id}.wav"></audio>
            <br>Rating: <input type="number" min="1" max="5">`;
            container.appendChild(div);
        });
    </script>
</body>
</html>"""
    with open(os.path.join(mos_dir, "listening_test", "index.html"), "w") as f:
        f.write(html)
        
    print("MOS preparation completed.")

if __name__ == "__main__":
    prepare_mos()
