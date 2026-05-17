import librosa
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_and_preprocess_audio(file_path, target_sr=16000):
    """
    Loads an audio file safely, converts to mono, resamples, and normalizes.
    
    Args:
        file_path (str): Path to audio file.
        target_sr (int): Target sampling rate (default 16000).
        
    Returns:
        np.ndarray: Preprocessed waveform.
    """
    try:
        # librosa automatically converts to mono if mono=True (default)
        waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        # Normalize audio (zero mean, unit variance) - basic normalization
        # Some models prefer normalized audio; it generally helps robustness
        if len(waveform) > 0:
            waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
            
        return waveform
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {e}")
        return None
