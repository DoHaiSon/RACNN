% Data
x = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0];

CNN_15 = [10.94293924763387
3.40819437311494
1.037589642705364
0.2870081191902069
0.06635475457740174
0.018951154025429257
0.008919843998854227];

CNN_10_15 = [11.0147647821397
3.39865787528828
1.03357048029116
0.292520854853871
0.0710349775301537
0.0203378657115113
0.00923080996148221];

RACNN_10_15 = [10.897715735765283
3.3762121639637424
1.0289418154194352
0.29196742679392207
0.07595158163028762
0.023631914328914684
0.010500888255704511];


% Define available line styles and markers
line_styles = {'-', '--', ':', '-.'};
markers = {'', 'o', '+', '*', '.', 'x', 's', 'd', '^', 'v', '>', '<', 'p', 'h'};

% Function to get random line style and marker
get_random_style = @() strcat(line_styles{randi(numel(line_styles))}, markers{randi(numel(markers))});

semilogy(x, CNN_15, get_random_style(), 'DisplayName', 'CNN-15', 'LineWidth', 1.5);
hold on;
semilogy(x, CNN_10_15, get_random_style(), 'DisplayName', 'CNN-10-15', 'LineWidth', 1.5); 
semilogy(x, RACNN_10_15, get_random_style(), 'DisplayName', 'RACNN-10:5:15', 'LineWidth', 1.5); 

% Labels and Title
xlabel('SNR (dB)');
ylabel('NMSE');
title('Logarithmic Plot of Data');

% Additional plot settings
legend('show'); % Display the legend
grid on;
set(gca, 'FontSize', 12); % Set font size for better readability
