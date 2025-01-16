"""
RACNN (Residual Attention CNN) for Near-field Channel Estimation in 6G
This module implements a CNN architecture combining residual connections and attention mechanisms
for improved channel estimation performance.
"""

from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Subtract, Attention
from keras.models import Model

def residual_block_with_attention(x, filters, kernel_size=10):
    """
    Creates a residual block with an attention mechanism.
    
    The block consists of:
    1. Two convolutional layers with batch normalization
    2. An attention mechanism to focus on important features
    3. A residual connection to help with gradient flow
    
    Args:
        x (tensor): Input tensor to the block
        filters (int): Number of filters in convolutional layers
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 10.
    
    Returns:
        tensor: Output tensor after processing through the residual attention block
    """
    # Store input for residual connection
    shortcut = x
    
    # First convolutional block
    x = Conv2D(filters, kernel_size, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)  # Normalize to stabilize training
    
    # Second convolutional block
    x = Conv2D(filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x)
    
    # Self-attention mechanism
    # Helps the network focus on important features in the feature map
    attn = Attention()([x, x])  # Self-attention where query and key are the same
    x = Add()([x, attn])  # Combine attention features with original features
    
    # Residual connection and final activation
    x = Add()([x, shortcut])  # Add input to output (residual connection)
    x = Activation("relu")(x)  # Non-linear activation
    
    return x

def RACNN(input_shape, output_dim, num_residual_blocks=7, num_filters=64, kernel_size=3):
    """
    Builds the complete RACNN model for channel estimation.
    
    Architecture:
    1. Initial convolution layer
    2. Multiple residual blocks with attention
    3. Final output layer with residual connection to input
    
    Args:
        input_shape (tuple): Shape of input tensor (height, width, channels)
        output_dim (int): Number of output channels
        num_residual_blocks (int, optional): Number of residual blocks. Defaults to 5.
        num_filters (int, optional): Number of filters in conv layers. Defaults to 64.
        kernel_size (int, optional): Size of convolutional kernels. Defaults to 3.
    
    Returns:
        Model: Compiled Keras model ready for training
    """
    # Input layer
    inp = Input(shape=input_shape)
    x = inp
    
    # Initial feature extraction
    x = Conv2D(num_filters, kernel_size, padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    
    # Stack of residual blocks with attention
    for _ in range(num_residual_blocks):
        x = residual_block_with_attention(x, num_filters)
    
    # Final channel estimation
    x = Conv2D(output_dim, kernel_size, padding="same", activation="linear")(x)
    # Residual learning: predict the difference between input and true channel
    output = Subtract()([inp, x])
    
    # Create and return model
    model = Model(inputs=inp, outputs=output)
    return model