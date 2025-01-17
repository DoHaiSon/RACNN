import os, sys, io
import tensorflow as tf

import warnings
# Suppress ignore warnings
warnings.filterwarnings("ignore")

from utils.logger import Logger
from config import get_args  # Import the argument loader
from gen_data.noisy_channel import gen_noisy_channel
from utils.common import set_seed

from models.RACNN import RACNN
from models.CNN import CNN

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

# Function to select the network based on the provided argument
def select_model(net, input_shape, output_dim=2, num_blocks=7, num_filters=64, kernel_size=3):
    if net == 'RACNN':
        return RACNN(input_shape=input_shape, output_dim=output_dim, 
                     num_residual_blocks=num_blocks, num_filters=num_filters, kernel_size=kernel_size)
    elif net == 'CNN':
        return CNN(input_dim=input_shape, output_dim=output_dim,
                   n_layers=num_blocks, num_filters=num_filters, kernel_size=kernel_size)  
    else:
        raise ValueError(f"Unknown network type: {net}")

if __name__ == '__main__':
    args = get_args()
    
    seed = args.seed if args.use_seed else None

    # Set global seed if needed
    if args.use_seed:
        set_seed(args.seed)

    # Configure the logger
    writer = Logger(enable_logging=args.enable_logging, default_log_dir=args.default_log_dir)

    # Print the argument summary
    print("\n===== Training Configuration =====")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    
    # Format the argument summary for TensorBoard with markdown-style table
    arg_summary = "| **Parameter** | **Value** |\n|---|---|\n"
    for key, value in vars(args).items():
        if key == 'ckpt' or key == 'SNR_min' or key == 'SNR_max' or key == 'SNR_step':
            continue
        arg_summary += f"| {key} | {value} |\n"

    # Check if GPU is available and set the device
    if tf.config.list_physical_devices('GPU'):
        device = f"GPU:{args.gpu}"
    else:
        device = "CPU:0"
    print(f'Using device: {device}')
    # Set the device for TensorFlow
    tf.config.set_visible_devices(tf.config.list_physical_devices(device.split(':')[0])[int(device.split(':')[1])], device.split(':')[0])

    # Load datasets
    H_noisy_in, H_true_out, Mean_H_noisy_in, Mean_H_true_out, _ = gen_noisy_channel(data_path=args.train_data_path, SNR_range=args.SNR_range, Nx=args.Nx, Ny=args.Ny)
    arg_summary += f"| Mean H_noisy_in | {Mean_H_noisy_in:.6f} |\n"
    arg_summary += f"| Mean H_true_out | {Mean_H_true_out:.6f} |\n"

    ## Initialize the model
    input_shape = (args.Nx, args.Ny, 2)
    output_dim = 2
    num_blocks = args.num_blocks
    model = select_model(net=args.net, input_shape=input_shape, output_dim=output_dim, 
                         num_blocks=num_blocks, num_filters=args.num_filters, kernel_size=args.kernel_size)

    # Check if we should load a checkpoint
    if args.load_ckpt is not None:
        if os.path.isfile(args.load_ckpt):
            model = tf.keras.models.load_model(args.load_ckpt)
            print(f'Loaded checkpoint from {args.load_ckpt}')
        else:
            raise ValueError(f"Checkpoint file not found: {args.load_ckpt}")

    # Calculate the total number of parameters in the model
    model_summary_str = io.StringIO()
    model.summary(print_fn=lambda x: model_summary_str.write(x + '\n'))
    model_summary_str = model_summary_str.getvalue()
    # ----------------------------------------------------------------------------
    
    tensorboard_callback = TensorBoard(
        log_dir=args.default_log_dir,
        histogram_freq=1,
        update_freq='epoch',  # Log every epoch (or set your preferred frequency)
        write_graph=False,  # Optional: disable graph visualization
        write_images=False  # Optional: disable image summaries
    )
    
    # Convert SNR_range to a string
    snr_range_str = '_'.join(map(str, args.SNR_range))

    # Create the checkpoint name
    ckpt_name = f'{args.net}_net_{args.num_blocks}_blocks_{args.N}_N_{snr_range_str}_SNR.keras'
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
    print('Model checkpoint location:', ckpt_path)
    arg_summary += f"| Checkpoint Name | {ckpt_path} |\n"
    arg_summary += f"| Model Summary | {model_summary_str} |\n"
    
    # Log the argument summary to TensorBoard
    writer.add_text('Training Configuration', arg_summary, step=0)

    adam = Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='mse')

    checkpoint = ModelCheckpoint(ckpt_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint, tensorboard_callback]

    # Train the model
    history_callback = model.fit(
        x=H_noisy_in,
        y=H_true_out,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        callbacks=callbacks_list,
        verbose=1,
        shuffle=True,
        validation_split=args.validation_split
    )

    writer.close()