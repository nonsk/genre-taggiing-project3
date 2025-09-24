import numpy as np
import time
from .model_architecture import build_cnn_rnn_model, compile_model
from .config import TARGET_PARAMETERS, NUM_TAGS, TARGET_INFERENCE_MS

class MusicAutoTagger:
    def __init__(self):
        self.model = build_cnn_rnn_model()
        self.model = compile_model(self.model)
        self.tag_names = self._get_magnatagatune_tags()
        self._display_specifications()
        
    def _get_magnatagatune_tags(self):
        return [
            'guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 
            'rock', 'fast', 'piano', 'ambient', 'beat', 'violin', 'vocal', 'synth',
            'female', 'indian', 'opera', 'male', 'singing', 'vocals', 'no vocals',
            'harpsichord', 'loud', 'quiet', 'flute', 'woman', 'male vocal', 'no vocal',
            'pop', 'soft', 'sitar', 'solo', 'man', 'classic', 'choir', 'voice',
            'new age', 'dance', 'male voice', 'female vocal', 'beats', 'harp',
            'cello', 'no voice', 'weird', 'country', 'metal', 'female voice', 'choral'
        ]
    
    def _display_specifications(self):
        total_params = self.model.count_params()
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        print("=" * 70)
        print("MusicNet: Deep Learning Music Auto-Tagging System")
        print("=" * 70)
        print(f"Architecture: 4-layer CNN + 2-layer GRU")
        print(f"Parameters: {total_params:,}")
        print(f"Model size: {model_size_mb:.2f} MB")
        print(f"Tags supported: {NUM_TAGS} (MagnaTagATune)")
        print(f"Sampling rate: 12 kHz")
        print(f"Optimizer: Adam")
        print(f"Target inference: {TARGET_INFERENCE_MS} ms per prediction")
        print("=" * 70)
        
        if abs(total_params - TARGET_PARAMETERS) < 5000:
            print("Parameter target achieved successfully")
        else:
            print(f"Parameter deviation: {total_params - TARGET_PARAMETERS}")
    
    def predict_with_timing(self, melspectrograms):
        if len(melspectrograms.shape) == 2:
            melspectrograms = melspectrograms[np.newaxis, :]
        
        start_time = time.time()
        predictions = self.model.predict(melspectrograms, batch_size=1, verbose=0)
        inference_time = (time.time() - start_time) * 1000
        
        return predictions, inference_time
    
    def predict_tags(self, melspectrograms, top_k=10):
        predictions, inference_time = self.predict_with_timing(melspectrograms)
        
        results = []
        for prediction in predictions:
            top_indices = np.argsort(prediction)[::-1][:top_k]
            top_tags = [(self.tag_names[i], prediction[i]) for i in top_indices]
            results.append(top_tags)
        
        return results, inference_time
    
    def get_model_specifications(self):
        params = self.model.count_params()
        return {
            'architecture': '4-layer CNN + 2-layer GRU',
            'parameters': params,
            'model_size_mb': params * 4 / (1024 * 1024),
            'target_inference_ms': TARGET_INFERENCE_MS,
            'sampling_rate_khz': 12,
            'num_tags': NUM_TAGS,
            'optimizer': 'Adam',
            'framework': 'TensorFlow'
        }