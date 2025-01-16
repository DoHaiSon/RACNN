import os
import numpy as np
import scipy.io as sio

def gen_noisy_channel(data_path, SNR_range, Nx, Ny):

    PROJECT_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

    snr_count_train = len(SNR_range)

    raw_data        = sio.loadmat(os.path.join(PROJECT_dir, data_path))
    channel         = raw_data['Channel_mat_total']
    data_num_train  = int(raw_data['num_Channel'][0][0])

    H_noisy_in = np.zeros((data_num_train * snr_count_train, Nx, Ny, 2), dtype=float)
    H_true_out = np.zeros((data_num_train * snr_count_train, Nx, Ny, 2), dtype=float)

    for snr in SNR_range:
        print('Generate noisy channel at snr = ', snr)
        P = 10**(snr / 10.0)
        count = 0
        for i in range(data_num_train):
            h = channel[i]
            H = np.reshape(h, (Nx, Ny))
            H_true_out[data_num_train * count + i, :, :, 0] = np.real(H)
            H_true_out[data_num_train * count + i, :, :, 1] = np.imag(H)
            noise = 1 / np.sqrt(2) * np.random.randn(Nx, Ny) + 1j * 1 / np.sqrt(2) * np.random.randn(Nx, Ny)
            H_with_noisy = H + 1 / np.sqrt(P) * noise
            H_noisy_in[data_num_train * count + i, :, :, 0] = np.real(H_with_noisy)
            H_noisy_in[data_num_train * count + i, :, :, 1] = np.imag(H_with_noisy)
        count += 1
    Mean_H_noisy_in = ((H_noisy_in)**2).mean()
    Mean_H_true_out = ((H_true_out)**2).mean()

    return H_noisy_in, H_true_out, Mean_H_noisy_in, Mean_H_true_out