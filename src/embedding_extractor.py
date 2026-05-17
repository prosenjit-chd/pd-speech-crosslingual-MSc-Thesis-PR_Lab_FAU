import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Model
import logging
import numpy as np

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, model_name="facebook/wav2vec2-large-xlsr-53", device="cpu"):
        self.device = device
        self.model_name = model_name
        logger.info(f"Loading processor and model: {model_name}")
        
        try:
            self.processor = AutoFeatureExtractor.from_pretrained(model_name)
            # output_hidden_states=True is critical to get layer-wise embeddings
            self.model = Wav2Vec2Model.from_pretrained(model_name, output_hidden_states=True).to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}. Error: {e}")
            raise e

    def extract_features(self, waveform, target_layers=[0, 4, 8, 11]):
        """
        Extracts layer-wise mean-pooled hidden states from the model.
        
        Args:
            waveform (np.ndarray): 1D audio waveform (16 kHz).
            target_layers (list): List of layer indices to extract.
            
        Returns:
            dict: Dictionary mapping layer index to a 1D numpy array (feature vector).
        """
        if waveform is None or len(waveform) == 0:
            return None
            
        try:
            inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")
            input_values = inputs.input_values.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_values)
                
            # hidden_states is a tuple of length (num_layers + 1)
            # layer 0 is the output of the CNN feature extractor, layer 1-N are transformer layers
            hidden_states = outputs.hidden_states
            
            features = {}
            for layer in target_layers:
                if layer < len(hidden_states):
                    # shape: (batch_size, sequence_length, hidden_size)
                    layer_output = hidden_states[layer]
                    # Mean pooling over time (sequence_length dimension)
                    pooled_output = torch.mean(layer_output, dim=1).squeeze(0)
                    features[layer] = pooled_output.cpu().numpy()
                else:
                    logger.warning(f"Layer {layer} out of bounds for model with {len(hidden_states)} states.")
                    
            return features
            
        except Exception as e:
            logger.error(f"Error during feature extraction: {e}")
            return None
