clc
clear all

% Define the file paths
CNN_train = 'assets/run-run_CNN_20250116-212101_train-tag-epoch_loss.csv';
CNN_valid = 'assets/run-run_CNN_20250116-212101_validation-tag-epoch_loss.csv';
RACNN_train = 'assets/run-run_RACNN_20250118-030454_train-tag-epoch_loss.csv';
RACNN_valid = 'assets/run-run_RACNN_20250118-030454_validation-tag-epoch_loss.csv';

% Read the CSV files into tables
CNN_train = readtable(CNN_train);
CNN_valid = readtable(CNN_valid);
RACNN_train = readtable(RACNN_train);
RACNN_valid = readtable(RACNN_valid);

% Extract the loss values
CNN_train_loss = CNN_train.Value;
CNN_valid_loss = CNN_valid.Value;
RACNN_train_loss = RACNN_train.Value;
RACNN_valid_loss = RACNN_valid.Value;

% Display the loss values
semilogy(1:200, CNN_train_loss, '-*', 'color', '#EDB120', 'DisplayName', 'XLCNet: training', 'LineWidth', 1.5, 'MarkerIndices', 1:10:200);
hold on;
semilogy(1:200, CNN_valid_loss, '--*', 'color', '#7E2F8E', 'DisplayName', 'XLCNet: validation', 'LineWidth', 1.5, 'MarkerIndices', 1:10:200);
semilogy(1:200, RACNN_train_loss, '-d', 'color', '#0072BD', 'DisplayName', 'RACNN: training', 'LineWidth', 1.5, 'MarkerIndices', 1:10:200);
semilogy(1:200, RACNN_valid_loss, '--d', 'color', '#D95319', 'DisplayName', 'RACNN: validation', 'LineWidth', 1.5, 'MarkerIndices', 1:10:200);
ylim([10^-8 30]);

legend('Interpreter', 'latex', 'FontSize', 14, 'Edgecolor', 'white', 'NumColumns', 2);

grid minor;
xlabel('Iteration', 'FontSize', 14, 'Interpreter','latex');
ylabel('MSE', 'FontSize', 14, 'Interpreter','latex');
hAx=gca;                              % get the axes handle
hAx.XTickLabel=hAx.XTickLabel;        % overwrite the existing tick labels with present values
set(gcf,'color','w');
ax = get(gca,'XTickLabel');
xticks('auto');
set(gca,'XTickLabel',ax,'FontName','Times','fontsize',12);