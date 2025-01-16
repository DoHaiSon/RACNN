clear all;
clc;

% Parameters
N = 256; % Number of BS antennas
lamda = 0.01; % Wavelength
Anten_dis = lamda / 2; % Antenna spacing
Anten_dis_norm = 0.5; % Normalized antenna spacing
L_f = 10; % Maximum number of paths for far-field part
L_n = 10; % Maximum number of paths for near-field part
num_sta = 10; % Number of stations
num_ffading = 200; % Number of fading realizations
num_train = num_sta * num_ffading; % Total number of training samples
num_Channel = 100000; % Total number of channels to generate
Channel_mat_total = zeros(num_train, N); % Initialize total channel matrix

% Generate channels
for j = 1:num_Channel / num_train
    Channel_mat = zeros(num_train, N); % Initialize channel matrix
    Channel_near_mat = zeros(num_train, N); % Initialize near-field channel matrix
    Channel_far_mat = zeros(num_train, N); % Initialize far-field channel matrix
    
    for i = 1:num_sta
        Lf = randi([1, L_f]); % Random number of far-field paths
        Ln = randi([1, L_n]); % Random number of near-field paths
        Cp_far = 1; % Channel average power of far part
        Cp_near = 1; % Channel average power of near part
        theta_far = pi * unifrnd(-1/2, 1/2, 1, Lf); % AoA at BS of far part
        theta_near = pi * unifrnd(-1/2, 1/2, 1, Ln); % AoA at BS of near part
        r_near = 70 * rand(1, Ln) + 10; % Distance for near-field part
        
        for n = 1:num_ffading
            % Generate far-field channel
            A_far = 1/sqrt(N) * exp((-1j * 2 * pi * Anten_dis / lamda * [0:N-1]') * sin(theta_far));
            f_far = (normrnd(0, Cp_far/sqrt(2), Lf, 1) + 1j * normrnd(0, Cp_far/sqrt(2), Lf, 1));
            h_far = A_far * f_far;
            
            % Generate near-field channel
            delta = (2 * [1:N]' - N - 1) / 2;
            B_near = zeros(N, Ln);
            for l = 1:Ln
                r_dis = sqrt(r_near(l)^2 + delta.^2 * Anten_dis^2 - 2 * r_near(l) * delta * Anten_dis * sin(theta_near(l)));
                B_near(:, l) = 1/sqrt(N) * exp(-1j * 2 * pi / lamda * (r_dis - r_near(l)));
            end
            f_near = (normrnd(0, Cp_near/sqrt(2), Ln, 1) + 1j * normrnd(0, Cp_near/sqrt(2), Ln, 1));
            h_near = B_near * f_near;
            
            % Hybrid channel
            h_hyb = sqrt(N / (Lf + Ln)) * (h_far + h_near);
            Channel_mat((i-1) * num_ffading + n, :) = h_hyb.';
            Channel_far_mat((i-1) * num_ffading + n, :) = sqrt(N / (Lf + Ln)) * h_far.';
            Channel_near_mat((i-1) * num_ffading + n, :) = sqrt(N / (Lf + Ln)) * h_near.';
        end
    end
    
    if j == 1
        Channel_mat_total = Channel_mat;
    else
        Channel_mat_total = [Channel_mat_total; Channel_mat];
    end
end

% Get the absolute path of the current script
currentScriptPath = mfilename('fullpath');
[currentScriptDir, ~, ~] = fileparts(currentScriptPath);
% Define the output directory
outputDir = fullfile(currentScriptDir, '..', 'data');
if ~exist(outputDir, 'dir')
    mkdir(outputDir)
end

% Save the generated data
pathName = fullfile(outputDir, sprintf('data_%d_f_%d_n_%d_samples_%d_N_%d_numsta_%d_fading.mat', L_f, L_n, num_Channel, N, num_sta, num_ffading));
save(pathName, 'Channel_mat_total', 'num_Channel');