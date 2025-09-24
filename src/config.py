import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = '1'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def setup_tensorflow():
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

setup_tensorflow()

SAMPLING_RATE = 12000
N_FFT = 512
N_MELS = 96
HOP_LENGTH = 256
AUDIO_DURATION = 29.12
TARGET_PARAMETERS = 397000
NUM_TAGS = 50
TARGET_INFERENCE_MS = 42