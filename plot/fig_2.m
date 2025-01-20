% Data
x = 0:4:20;

CNN_7_64_3_10_15 = [1.0380768892362895
0.3783703559916995
0.12212936645951938
0.03742235142072346
0.015230018976023675
0.008453012442370554];

RACNN_5_64_4_10_15 = [0.9216867010187981
0.3100383075859617
0.08645405361864493
0.021925338001454516
0.008532896721387339
0.0048748365497651575];

LS = load('assets/NMSE_LS.mat');
LS = LS.NMSE;

MMSE = load('assets/NMSE_MMSE.mat');
MMSE = MMSE.NMSE;

semilogy(x, LS, '-', 'color', '#77AC30', 'DisplayName', 'LS', 'LineWidth', 1.5);
hold on;
semilogy(x, MMSE, '-<', 'color', '#4DBEEE', 'DisplayName', 'MMSE', 'LineWidth', 1.5);
semilogy(x, CNN_7_64_3_10_15, '-*', 'color', '#7E2F8E', 'DisplayName', 'XLCNet', 'LineWidth', 1.5);
semilogy(x, RACNN_5_64_4_10_15, '-d', 'color', '#D95319', 'DisplayName', 'Proposed RACNN', 'LineWidth', 1.5); 

ylim([min(RACNN_5_64_4_10_15)/2 max(LS)*2]);

% Labels and Title
grid minor;
ylabel('NMSE', 'FontSize', 14, 'Interpreter', 'latex');
xlabel('SNR (dB)', 'FontSize', 14, 'Interpreter', 'latex');
legend('Interpreter', 'latex', 'FontSize', 14, 'Edgecolor', 'white');
hAx=gca;                              % get the axes handle
hAx.XTickLabel=hAx.XTickLabel;        % overwrite the existing tick labels with present values
set(gcf, 'color', 'w');
set(gca, 'XTick', 0:4:20, 'XTickLabel', 0:4:20);
set(gca, 'FontName', 'Times', 'fontsize', 12);