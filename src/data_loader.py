import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from .audio_processing import compute_melspectrogram

def load_magnatagatune_dataset(annotations_file, audio_directory):
    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")
    if not os.path.exists(audio_directory):
        raise FileNotFoundError(f"Audio directory not found: {audio_directory}")
        
    print("Loading MagnaTagATune annotations...")
    dataframe = pd.read_csv(annotations_file, sep='\t', lineterminator='\r')
    tag_columns = dataframe.columns[2:-1]

    for tag in tqdm(tag_columns, desc="Sanitizing tag labels"):
        dataframe[tag] = pd.to_numeric(dataframe[tag], errors='coerce').fillna(0).astype(int)
    
    dataframe['mp3_path'] = dataframe['mp3_path'].astype(str)
    dataframe['audio_path'] = dataframe['mp3_path'].apply(lambda x: os.path.join(audio_directory, x))
    print(dataframe['audio_path'].iloc[0])
    dataframe = dataframe[dataframe['audio_path'].apply(os.path.isfile)].copy()
    
    audio_paths = dataframe['audio_path'].tolist()
    labels = dataframe[tag_columns].values
    
    print(f"Dataset loaded: {len(audio_paths)} audio files with {len(tag_columns)} tags")
    return audio_paths, labels

def create_data_batches(audio_paths, labels, batch_size=32, show_progress=True):
    batched_data = []
    batched_labels = []
    
    iterator = tqdm(range(0, len(audio_paths), batch_size), desc="Creating batches") if show_progress else range(0, len(audio_paths), batch_size)
    
    for i in iterator:
        batch_paths = audio_paths[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]
        
        batch_melspecs = []
        valid_labels = []
        
        for path, label in zip(batch_paths, batch_labels):
            melspec = compute_melspectrogram(path)
            if melspec is not None:
                batch_melspecs.append(melspec)
                valid_labels.append(label)
        
        if batch_melspecs:
            batched_data.append(np.array(batch_melspecs))
            batched_labels.append(np.array(valid_labels))
    
    return batched_data, batched_labels