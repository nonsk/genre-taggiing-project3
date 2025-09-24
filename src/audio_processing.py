import librosa
import numpy as np
import warnings
from tqdm import tqdm
from .config import SAMPLING_RATE, N_FFT, N_MELS, HOP_LENGTH, AUDIO_DURATION

warnings.filterwarnings('ignore')

def compute_melspectrogram(audio_path):
    try:
        audio_data, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True, duration=AUDIO_DURATION)
        
        target_length = int(AUDIO_DURATION * SAMPLING_RATE)
        
        if len(audio_data) < target_length:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
        elif len(audio_data) > target_length:
            start_idx = (len(audio_data) - target_length) // 2
            audio_data = audio_data[start_idx:start_idx + target_length]
        
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=SAMPLING_RATE, 
            n_fft=N_FFT, 
            hop_length=HOP_LENGTH, 
            n_mels=N_MELS
        )
        
        mel_db = librosa.power_to_db(mel_spectrogram, ref=1.0)
        return mel_db
        
    except Exception as e:
        print(f"Processing failed for {audio_path}: {e}")
        return None

def batch_process_audio(file_paths, show_progress=True):
    melspectrograms = []
    iterator = tqdm(file_paths, desc="Processing audio files") if show_progress else file_paths
    
    for file_path in iterator:
        melspec = compute_melspectrogram(file_path)
        if melspec is not None:
            melspectrograms.append(melspec)
    
    return np.array(melspectrograms) if melspectrograms else None

def process_single_audio(audio_path):
    melspec = compute_melspectrogram(audio_path)
    return melspec[np.newaxis, :] if melspec is not None else None