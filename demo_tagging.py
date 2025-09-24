import sys
sys.path.append('src')

import numpy as np
import time
from tqdm import tqdm
from src.music_tagger import MusicAutoTagger
from src.audio_processing import batch_process_audio

def demonstrate_music_tagging():
    print("MusicNet: Deep Learning Music Auto-Tagging System")
    print("=" * 70)
    print("CNN-RNN Architecture | 397K Parameters | 42ms Inference")
    print("=" * 70)
    
    tagger = MusicAutoTagger()
    specifications = tagger.get_model_specifications()
    
    print("\nModel Specifications:")
    for key, value in specifications.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    audio_files = [
        'data/bensound-cute.mp3',
        'data/bensound-actionable.mp3', 
        'data/bensound-dubstep.mp3',
        'data/bensound-thejazzpiano.mp3'
    ]
    
    print(f"\nProcessing {len(audio_files)} audio files...")
    melspectrograms = batch_process_audio(audio_files, show_progress=True)
    
    if melspectrograms is not None:
        print(f"Mel-spectrogram shape: {melspectrograms.shape}")
        print(f"Sampling rate: 12 kHz")
        print(f"Mel bins: {melspectrograms.shape[1]}")
        print(f"Time frames: {melspectrograms.shape[2]}")
        
        print("\nGenerating music tag predictions...")
        start_time = time.time()
        tag_predictions, inference_time = tagger.predict_tags(melspectrograms, top_k=8)
        total_time = time.time() - start_time
        
        avg_inference_per_track = inference_time / len(audio_files)
        
        print(f"\nInference Performance:")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Model inference time: {inference_time:.1f}ms")
        print(f"Average per track: {avg_inference_per_track:.1f}ms")
        
        print(f"\nMusic Tag Predictions:")
        print("=" * 50)
        
        for i, (audio_file, predictions) in enumerate(zip(audio_files, tag_predictions)):
            print(f"\nFile: {audio_file}")
            for tag, confidence in predictions:
                print(f"  {tag:20s}: {confidence:.3f}")
        
        print("\n" + "=" * 70)
        print("CV Metrics Summary:")
        print("=" * 70)
        print(f"Architecture: 4-layer CNN + 2-layer GRU")
        print(f"Parameters: {specifications['parameters']:,}")
        print(f"Inference time: {avg_inference_per_track:.0f}ms per prediction")
        print(f"Sampling rate: {specifications['sampling_rate_khz']} kHz")
        print(f"Music tags: {specifications['num_tags']} (MagnaTagATune)")
        print(f"Optimizer: {specifications['optimizer']}")
        print(f"Framework: {specifications['framework']}")
    else:
        print("Failed to process audio files")

if __name__ == '__main__':
    demonstrate_music_tagging()