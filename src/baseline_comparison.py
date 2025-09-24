import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, BatchNormalization, Activation, Flatten
from .model_architecture import build_cnn_rnn_model
from .config import TARGET_PARAMETERS

def build_larger_baseline_cnn():
    inputs = Input(shape=(96, 1366, 1))
    
    x = Conv2D(96, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 4))(x)

    x = Conv2D(192, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 4))(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 5))(x)

    x = Conv2D(384, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((4, 4))(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(50, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name='Large_Baseline_CNN')
    return model

def compare_model_parameters():
    print("=" * 70)
    print("Model Parameter Comparison Analysis")
    print("=" * 70)
    
    print("Building larger baseline CNN model...")
    baseline_model = build_larger_baseline_cnn()
    baseline_params = baseline_model.count_params()
    
    print("Building optimized CNN-RNN model...")
    optimized_model = build_cnn_rnn_model()
    optimized_params = optimized_model.count_params()
    
    parameter_reduction = 100 * (1 - (optimized_params / baseline_params))
    
    print("\nParameter Analysis:")
    print(f"Baseline CNN model: {baseline_params:,} parameters")
    print(f"Optimized CNN-RNN model: {optimized_params:,} parameters")
    print(f"Parameter reduction: {parameter_reduction:.1f}%")
    
    print("\nModel Architecture Comparison:")
    print("Baseline CNN:")
    print("- 4 CNN blocks with large filter counts")
    print("- Dense hidden layer (512 units)")
    print("- No temporal modeling")
    
    print("\nOptimized CNN-RNN:")
    print("- 4 CNN layers for feature extraction")
    print("- 2 GRU layers for temporal modeling")
    print("- Efficient parameter utilization")
    
    if parameter_reduction >= 50:
        print(f"\nParameter reduction target achieved: {parameter_reduction:.1f}% reduction")
    else:
        print(f"\nParameter reduction below target: {parameter_reduction:.1f}%")
    
    return {
        'baseline_parameters': baseline_params,
        'optimized_parameters': optimized_params,
        'parameter_reduction_percent': parameter_reduction
    }