import tensorflow as tf
import numpy as np
from keras.models import load_model

import warnings
# Suppress ignore warnings
warnings.filterwarnings("ignore")

from config import get_args  # Import the argument loader
from gen_data.noisy_channel import gen_noisy_channel
from utils.common import set_seed

if __name__ == '__main__':
    args = get_args()
    
    seed = args.seed if args.use_seed else None

    # Set global seed if needed
    if args.use_seed:
        set_seed(args.seed)

    # Check if GPU is available and set the device
    if tf.config.list_physical_devices('GPU'):
        device = f"GPU:{args.gpu}"
    else:
        device = "CPU:0"
    print(f'Using device: {device}')

    # Set the device for TensorFlow
    tf.config.set_visible_devices(tf.config.list_physical_devices(device.split(':')[0])[int(device.split(':')[1])], device.split(':')[0])

    # Load the model
    model = load_model(args.ckpt, compile=True)
    model.summary()

    # Load datasets
    snr_count = int((args.SNR_max - args.SNR_min) / args.SNR_step)

    for snr in range(args.SNR_min, args.SNR_max + args.SNR_step, args.SNR_step):
        H_noisy_in_test, H_true_out_test, _, _, data_num_test = gen_noisy_channel(data_path=args.test_data_path, SNR_range=[snr], Nx=args.Nx, Ny=args.Ny)
        
        # Model prediction
        decoded_channel = model.predict(H_noisy_in_test, batch_size=args.batch_size)
        
        # Calculate NMSE
        nmse = np.zeros((data_num_test, 1))
        for n in range(data_num_test):
            MSE = ((H_true_out_test[n,:,:,:] - decoded_channel[n,:,:,:])**2).sum()
            norm_real = ((H_true_out_test[n,:,:,:])**2).sum()
            nmse[n] = MSE/norm_real
        
        avg_nmse = nmse.sum() / data_num_test
        print(f"SNR: {snr} dB, NMSE: {avg_nmse}")
