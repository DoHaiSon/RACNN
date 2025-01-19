% Data
x = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0];

CNN_7_64_3_10_15 = [11.002236649217588
3.410245567870639
1.0363549248374513
0.2894068902869407
0.06674470455047488
0.01842139177390176
0.008460230053075753];

RACNN_5_64_4_10_15 = [10.325361662427092
3.1637872051127576
0.9199750798440295
0.2302889683155809
0.04283032819019456
0.010281928381504891
0.004877664172946971];

LS = load('assets/NMSE_LS.mat');
LS = LS.NMSE;

MMSE = load('assets/NMSE_MMSE.mat');
MMSE = MMSE.NMSE;

semilogy(x, LS, '-', 'color', '#77AC30', 'DisplayName', 'LS', 'LineWidth', 1.5);
hold on;
semilogy(x, MMSE, '-<', 'color', '#4DBEEE', 'DisplayName', 'MMSE', 'LineWidth', 1.5);
semilogy(x, CNN_7_64_3_10_15, '-*', 'color', '#7E2F8E', 'DisplayName', 'XLCNet', 'LineWidth', 1.5);
semilogy(x, RACNN_5_64_4_10_15, '-d', 'color', '#D95319', 'DisplayName', 'Proposed RACNN', 'LineWidth', 1.5); 

% Labels and Title
grid minor;
ylabel('NMSE', 'FontSize', 14, 'Interpreter','latex');
xlabel('SNR (dB)', 'FontSize', 14, 'Interpreter','latex');
legend('Interpreter', 'latex', 'FontSize', 14, 'Edgecolor', 'white');
hAx=gca;                              % get the axes handle
hAx.XTickLabel=hAx.XTickLabel;        % overwrite the existing tick labels with present values
set(gcf,'color','w');
ax = get(gca,'XTickLabel');
xticks('auto');
set(gca,'XTickLabel',ax,'FontName','Times','fontsize',12);