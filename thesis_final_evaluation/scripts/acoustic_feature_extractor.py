"""
Core Acoustic Feature Extraction Module
=======================================
Implements standardized, reproducible, interpretable acoustic feature extraction for read-text speech.
Uses librosa, numpy, scipy, and soundfile.
"""

import numpy as np
import librosa
import soundfile as sf
import warnings

def extract_features_from_audio(file_path, target_sr=22050):
    """
    Extracts interpretable acoustic features from a given WAV audio file.
    Returns: (features_dict, failure_dict)
    """
    failures = {}
    features = {}
    
    try:
        y, sr = sf.read(file_path)
    except Exception as e:
        return None, {"error": f"Failed to read audio: {str(e)}"}
        
    if y.ndim > 1:
        y = np.mean(y, axis=1)
        
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        
    total_duration = len(y) / sr
    features["duration_total_sec"] = float(total_duration)
    
    # 1. Energy / Intensity (RMS)
    try:
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
        rms_cv = float(rms_std / (rms_mean + 1e-9))
        features["rms_mean"] = rms_mean
        features["rms_std"] = rms_std
        features["rms_cv"] = rms_cv
    except Exception as e:
        failures["energy"] = str(e)
        features["rms_mean"] = np.nan
        features["rms_std"] = np.nan
        features["rms_cv"] = np.nan

    # 2. F0 / Prosody (pyin)
    try:
        fmin = 50
        fmax = 500
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            fill_na=np.nan
        )
        voiced_f0 = f0[~np.isnan(f0)]
        voiced_count = len(voiced_f0)
        total_frames = len(f0)
        
        if voiced_count > 5:
            f0_mean = float(np.mean(voiced_f0))
            f0_median = float(np.median(voiced_f0))
            f0_std = float(np.std(voiced_f0))
            f0_cv = float(f0_std / (f0_mean + 1e-9))
            f0_p05 = float(np.percentile(voiced_f0, 5))
            f0_p95 = float(np.percentile(voiced_f0, 95))
            f0_range_robust = float(f0_p95 - f0_p05)
            voiced_fraction = float(voiced_count / total_frames)
            
            # Voiced vs Unvoiced Duration and Pauses
            # A pause is defined as contiguous unvoiced segment > 150 ms
            frame_dur_sec = hop_length / sr
            voiced_duration = float(voiced_count * frame_dur_sec)
            unvoiced_duration = float(total_duration - voiced_duration)
            
            # Find contiguous unvoiced chunks
            is_unvoiced = np.isnan(f0)
            diffs = np.diff(np.concatenate(([0], is_unvoiced.view(np.int8), [0])))
            pause_starts = np.where(diffs == 1)[0]
            pause_ends = np.where(diffs == -1)[0]
            pause_durations = (pause_ends - pause_starts) * frame_dur_sec
            
            # Min threshold 150ms for speech pause
            valid_pauses = pause_durations[pause_durations >= 0.150]
            num_pauses = int(len(valid_pauses))
            mean_pause_dur = float(np.mean(valid_pauses)) if num_pauses > 0 else 0.0
            pause_ratio = float(unvoiced_duration / (total_duration + 1e-9))
            
            # Voiced segments per second (Speaking rate proxy)
            is_voiced = ~np.isnan(f0)
            v_diffs = np.diff(np.concatenate(([0], is_voiced.view(np.int8), [0])))
            v_starts = np.where(v_diffs == 1)[0]
            num_voiced_segments = int(len(v_starts))
            voiced_segment_rate = float(num_voiced_segments / (total_duration + 1e-9))
            
            features["f0_mean_hz"] = f0_mean
            features["f0_median_hz"] = f0_median
            features["f0_std_hz"] = f0_std
            features["f0_cv"] = f0_cv
            features["f0_range_robust_hz"] = f0_range_robust
            features["voiced_fraction"] = voiced_fraction
            features["duration_voiced_sec"] = voiced_duration
            features["duration_unvoiced_sec"] = unvoiced_duration
            features["num_pauses"] = num_pauses
            features["mean_pause_duration_sec"] = mean_pause_dur
            features["pause_ratio"] = pause_ratio
            features["voiced_segment_rate_hz"] = voiced_segment_rate
            
            # Local perturbation proxies (Jitter / Shimmer proxies on contiguous voiced chunks)
            # Cycle-to-cycle relative perturbation on contiguous voiced frames
            f0_diffs = np.abs(np.diff(voiced_f0))
            jitter_local_proxy = float(np.mean(f0_diffs) / (f0_mean + 1e-9))
            features["jitter_local_proxy"] = jitter_local_proxy
            
        else:
            failures["f0"] = "Insufficient voiced frames detected (< 5 frames)"
            features["f0_mean_hz"] = np.nan
            features["f0_median_hz"] = np.nan
            features["f0_std_hz"] = np.nan
            features["f0_cv"] = np.nan
            features["f0_range_robust_hz"] = np.nan
            features["voiced_fraction"] = 0.0
            features["duration_voiced_sec"] = 0.0
            features["duration_unvoiced_sec"] = total_duration
            features["num_pauses"] = 0
            features["mean_pause_duration_sec"] = 0.0
            features["pause_ratio"] = 1.0
            features["voiced_segment_rate_hz"] = 0.0
            features["jitter_local_proxy"] = np.nan
            
    except Exception as e:
        failures["f0"] = str(e)
        features["f0_mean_hz"] = np.nan
        features["f0_median_hz"] = np.nan
        features["f0_std_hz"] = np.nan
        features["f0_cv"] = np.nan
        features["f0_range_robust_hz"] = np.nan
        features["voiced_fraction"] = np.nan
        features["duration_voiced_sec"] = np.nan
        features["duration_unvoiced_sec"] = np.nan
        features["num_pauses"] = np.nan
        features["mean_pause_duration_sec"] = np.nan
        features["pause_ratio"] = np.nan
        features["voiced_segment_rate_hz"] = np.nan
        features["jitter_local_proxy"] = np.nan

    # 3. Spectral Features
    try:
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, roll_percent=0.85)[0]
        spec_flat = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
        
        features["spectral_centroid_mean"] = float(np.mean(spec_cent))
        features["spectral_centroid_std"] = float(np.std(spec_cent))
        features["spectral_bandwidth_mean"] = float(np.mean(spec_bw))
        features["spectral_bandwidth_std"] = float(np.std(spec_bw))
        features["spectral_rolloff_85_mean"] = float(np.mean(spec_roll))
        features["spectral_rolloff_85_std"] = float(np.std(spec_roll))
        features["spectral_flatness_mean"] = float(np.mean(spec_flat))
        features["spectral_flatness_std"] = float(np.std(spec_flat))
    except Exception as e:
        failures["spectral"] = str(e)
        for k in ["spectral_centroid_mean", "spectral_centroid_std", "spectral_bandwidth_mean",
                  "spectral_bandwidth_std", "spectral_rolloff_85_mean", "spectral_rolloff_85_std",
                  "spectral_flatness_mean", "spectral_flatness_std"]:
            features[k] = np.nan

    # 4. Harmonic-to-Noise Ratio (HNR proxy via Harmonic-Percussive / Residual separation)
    try:
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_energy = np.sum(y_harm ** 2) + 1e-12
        noise_energy = np.sum(y_perc ** 2) + 1e-12
        hnr_db = float(10.0 * np.log10(harm_energy / noise_energy))
        features["hnr_hpss_db"] = hnr_db
    except Exception as e:
        failures["hnr"] = str(e)
        features["hnr_hpss_db"] = np.nan

    # 5. Cepstral Features (MFCC 1 to 13 mean and std)
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        for i in range(13):
            features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
            features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
    except Exception as e:
        failures["mfcc"] = str(e)
        for i in range(13):
            features[f"mfcc_{i+1}_mean"] = np.nan
            features[f"mfcc_{i+1}_std"] = np.nan

    return features, failures
