import random
import numpy as np
import tensorflow as tf

def set_seed(seed):
    random.seed(seed)  # For Python random
    np.random.seed(seed)  # For NumPy
    tf.random.set_seed(seed)  # For TensorFlow/Keras