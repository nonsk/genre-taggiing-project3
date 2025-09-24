import sys
sys.path.append('src')

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from src.music_tagger import MusicAutoTagger
from src.data_loader import load_magnatagatune_dataset
from src.audio_processing import compute_melspectrogram

ANNOTATIONS_FILE = './magnatagatune_dataset/annotations_final.csv'
AUDIO_DIR = './magnatagatune_dataset/mp3'

def evaluate_auc_performance():
    print("MagnaTagATune Dataset Evaluation")
    print("=" * 70)

    try:
        audio_paths, true_labels = load_magnatagatune_dataset(ANNOTATIONS_FILE, AUDIO_DIR)
        print(f"Loaded {len(audio_paths)} audio files for evaluation")
    except FileNotFoundError:
        print("ERROR: MagnaTagATune dataset not found")
        print("Please download dataset and update file paths")
        return None

    print("\nInitializing CNN-RNN model with 397K parameters...")
    tagger = MusicAutoTagger()
    
    print("\nProcessing audio files and generating predictions...")
    predicted_labels = []
    inference_times = []
    
    for audio_path in tqdm(audio_paths, desc="Evaluating model"):
        melspectrogram = compute_melspectrogram(audio_path)
        if melspectrogram is not None:
            melspec_batch = melspectrogram[np.newaxis, :]
            prediction, inference_time = tagger.predict_with_timing(melspec_batch)
            predicted_labels.append(prediction[0])
            inference_times.append(inference_time)
        
        # Print progress every 100 files, or if it's the first file
        if len(predicted_labels) % 100 == 0 or len(predicted_labels) == 1:
            print(f"Processed {len(predicted_labels)} files - Avg inference: {np.mean(inference_times):.1f}ms")

    predicted_labels = np.array(predicted_labels)
    
    print("\nCalculating ROC AUC Score...")
    try:
        auc_score = roc_auc_score(true_labels[:len(predicted_labels)], predicted_labels, average='macro')
        avg_inference_time = np.mean(inference_times)
        
        print("\n" + "=" * 70)
        print("PERFORMANCE EVALUATION RESULTS")
        print("=" * 70)
        print(f"ROC AUC Score (Macro Average): {auc_score:.3f}")
        print(f"Average inference time: {avg_inference_time:.1f} ms")
        print(f"Model parameters: {tagger.model.count_params():,}")
        print(f"Architecture: 4-layer CNN + 2-layer GRU")
        print(f"Dataset: MagnaTagATune ({len(predicted_labels)} samples evaluated)")
        
        if auc_score >= 0.890:
            print(f"AUC target achieved: {auc_score:.1%}")
        else:
            print(f"AUC below target: {auc_score:.1%}")
            
        if avg_inference_time <= 45:
            print(f"Inference speed target achieved: {avg_inference_time:.1f}ms")
        else:
            print(f"Inference speed above target: {avg_inference_time:.1f}ms")
            
        return {
            'auc_score': auc_score,
            'avg_inference_ms': avg_inference_time,
            'samples_evaluated': len(predicted_labels)
        }

    except Exception as e:
        print(f"AUC calculation failed: {e}")
        return None

if __name__ == '__main__':
    results = evaluate_auc_performance()