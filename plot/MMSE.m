% MIMO Channel Estimation using Simplified MMSE Algorithm
% Signal model: y = Hx + n
% Author: DoHaiSon
% Date: 2025-01-19

% Clear all variables and close all figures
clear all;
close all;
clc;

% System parameters
Nt = 1;                     % Number of transmit antennas
Nr = 256;                   % Number of receive antennas
Np = 1;                     % Number of pilot symbols
H_matrix = load('../data/data_10_f_10_n_10000_samples_256_N_10_numsta_200_fading.mat');
num_experiments = H_matrix.num_Channel;      % Number of Monte Carlo experiments
H_matrix = H_matrix.Channel_mat_total;
SNR_dB = -10:5:20;          % SNR range in dB

% Initialize NMSE storage
NMSE = zeros(1, length(SNR_dB));

% Main loop for different SNR values
for snr_idx = 1:length(SNR_dB)
    
    % Convert SNR from dB to linear scale
    SNR = 10^(SNR_dB(snr_idx)/10);
    sigma_n2 = 1/SNR;  % Noise variance (sigma_n^2)
    
    % Initialize error storage for current SNR
    error_sum = 0;
    
    % Monte Carlo experiments
    for exp = 1:num_experiments
        
        % Generate pilot symbols (QPSK modulation)
        pilot_symbols = (1/sqrt(2)) * (sign(randn(Nt, Np)) + 1i*sign(randn(Nt, Np)));
        
        % Extract true channel matrix
        H_true = H_matrix(exp, :).';
        
        % Generate received signal without noise
        Y_clean = H_true * pilot_symbols;
        
        % Add noise
        Y = awgn(Y_clean, SNR_dB(snr_idx));
        
        % Simplified MMSE channel estimation
        H_est = Y * pilot_symbols' * pinv(pilot_symbols * pilot_symbols' + sigma_n2 * eye(Nt));
        
        % Calculate and accumulate NMSE for this experiment
        error_sum = error_sum + (norm(H_true - H_est, 'fro')^2) / (norm(H_true, 'fro')^2);
    end
    
    % Calculate average NMSE for current SNR
    NMSE(snr_idx) = error_sum / num_experiments;
end

% Plot results
figure;
semilogy(SNR_dB, NMSE, 'ro-', 'LineWidth', 1.5);
grid on;
xlabel('SNR (dB)');
ylabel('NMSE');

% Save results
save('./assets/NMSE_MMSE.mat', 'NMSE');
