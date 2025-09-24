import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Lambda
from tensorflow.keras.layers import BatchNormalization, Dropout, Reshape, GRU, Activation
from .config import NUM_TAGS, TARGET_PARAMETERS

def build_cnn_rnn_model():
    inputs = Input(shape=(96, 1366), name='melspectrogram_input')
    x = Lambda(lambda tensor: tf.expand_dims(tensor, axis=-1))(inputs)
    
    x = Conv2D(44, (3, 3), padding='same', name='cnn_layer_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 4))(x)
    x = Dropout(0.1)(x)
    
    x = Conv2D(76, (3, 3), padding='same', name='cnn_layer_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 4))(x)
    x = Dropout(0.15)(x)
    
    x = Conv2D(108, (3, 3), padding='same', name='cnn_layer_3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 4))(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(128, (3, 3), padding='same', name='cnn_layer_4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 4))(x)
    x = Dropout(0.25)(x)

    x = Reshape((5, 512))(x)
    
    x = GRU(86, return_sequences=True, name='gru_layer_1')(x)
    x = Dropout(0.3)(x)
    
    x = GRU(32, return_sequences=False, name='gru_layer_2')(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(NUM_TAGS, activation='sigmoid', name='output_tags')(x)
    
    model = Model(inputs, outputs, name='CNN_RNN_MusicTagger')
    return model

def build_baseline_cnn():
    inputs = Input(shape=(96, 1366, 1))
    
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 4))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 4))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 5))(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((4, 4))(x)
    
    x = tf.keras.layers.Flatten()(x)
    outputs = Dense(NUM_TAGS, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name='Baseline_CNN')
    return model

def compile_model(model, learning_rate=0.001):
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['binary_accuracy', 'precision', 'recall']
    )
    
    return model