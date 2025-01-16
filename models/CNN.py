from keras.layers import Input, Conv2D, BatchNormalization, Subtract
from keras.models import Model

def CNN(input_dim, output_dim, n_layers=5, num_filters=64, kernel_size=3):
    """
    Build basic CNN model for channel estimation
    
    Args:
        input_dim: Shape of input data (height, width, channels)
        output_dim: Number of output channels
        n_layers: Number of convolutional layers
        num_filters: Number of filters in Conv2D layers
        kernel_size: Size of kernel in Conv2D layers (K x K)
        
    Returns:
        model: Compiled Keras model
    """
    # Input layer
    inp = Input(shape=input_dim)
    
    # First Conv layer
    xn = Conv2D(filters=num_filters, 
                kernel_size=(kernel_size, kernel_size), 
                padding='Same', 
                activation='relu')(inp)
    xn = BatchNormalization()(xn)
    
    # Middle Conv layers
    for _ in range(n_layers):
        xn = Conv2D(filters=num_filters, 
                    kernel_size=(kernel_size, kernel_size), 
                    padding='Same', 
                    activation='relu')(xn)
        xn = BatchNormalization()(xn)
    
    # Output layer
    xn = Conv2D(filters=output_dim, 
                kernel_size=(kernel_size, kernel_size), 
                padding='Same', 
                activation='linear')(xn)
    x1 = Subtract()([inp, xn])
    
    model = Model(inputs=inp, outputs=x1)
    return model