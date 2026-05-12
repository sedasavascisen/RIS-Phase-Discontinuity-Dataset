
% baseline_fairness_K_analysis.m  —  v2

clear; close all; clc;

%% ──────────────────────────────────────────────
K_vals  = [5, 20, 50];

% CD Phase-aware RMSE [°]
RMSE_CD = [3.50, 3.49, 3.20];

% CD  [ms/örnek]
TIME_CD = [0.15, 0.60, 1.50];

CNN_RMSE = 2.77;    % CNN-LSTM RMSE [°]
CNN_time = 0.08;    % CNN-LSTM  [ms]
URLLC_ms = 1.0;     % 6G URLLC  [ms]

%% ───────────────────────────────────────────────────
clr_rmse  = [0.000, 0.447, 0.741];   
clr_time  = [0.850, 0.100, 0.100];   
clr_cnn_r = [0.000, 0.447, 0.741];   
clr_urllc = [0.20,  0.20,  0.20 ];   
clr_cnn_t = [0.850, 0.325, 0.098];   

%% ──────────────────────────────────────────
fig = figure('Color', 'white', 'Position', [100, 100, 820, 540]);
ax1 = axes(fig);

%% ───RMSE─────────────────────────────
yyaxis(ax1, 'left');

h1 = plot(ax1, K_vals, RMSE_CD, ...
    '-sb', 'LineWidth', 2.5, 'MarkerSize', 10, ...
    'MarkerFaceColor', clr_rmse, ...
    'DisplayName', 'CD (K) RMSE');
hold(ax1, 'on');

% CNN-LSTM RMSE 
h2 = yline(ax1, CNN_RMSE, '--', ...
    'LineWidth', 2.0, 'Color', clr_cnn_r, ...
    'DisplayName', sprintf('CNN-LSTM RMSE (%.2f°)', CNN_RMSE));

% Y limit
ylim(ax1, [2.40, 4.20]);
ylabel(ax1, 'Phase-Aware RMSE [°]', ...
    'FontName', 'Times New Roman', 'FontSize', 13, ...
    'FontWeight', 'bold', 'Color', clr_rmse);
ax1.YColor = clr_rmse;

%% ───────────────────────────
yyaxis(ax1, 'right');

h3 = plot(ax1, K_vals, TIME_CD, ...
    '-or', 'LineWidth', 2.5, 'MarkerSize', 10, ...
    'MarkerFaceColor', clr_time, ...
    'DisplayName', 'CD (K) Latency');

% URLLC 1 ms limit 
h4 = yline(ax1, URLLC_ms, '--', ...
    'LineWidth', 2.0, 'Color', clr_urllc, ...
    'DisplayName', '6G URLLC Limit (1 ms)');

% CNN-LSTM Latency
h5 = yline(ax1, CNN_time, ':', ...
    'LineWidth', 2.2, 'Color', clr_cnn_t, ...
    'DisplayName', sprintf('CNN-LSTM Latency (%.2f ms)', CNN_time));

% Y limit: -0.5 → 14

ylim(ax1, [-0.5, 14]);
ylabel(ax1, 'Inference Latency [ms]', ...
    'FontName', 'Times New Roman', 'FontSize', 13, ...
    'FontWeight', 'bold', 'Color', clr_time);
ax1.YColor = clr_time;

%% ─── K=20 URLLC HandleVisibility──────────────────────────────
yyaxis(ax1, 'right');
x_shade = [20, 50, 50, 20];
y_shade = [URLLC_ms, URLLC_ms, 14, 14];
fill(ax1, x_shade, y_shade, [1.0 0.88 0.88], ...
    'FaceAlpha', 0.25, 'EdgeColor', 'none', ...
    'HandleVisibility', 'off');

%% ────────────
x_lbl = 51.5;   

% CNN-LSTM RMSE 
yyaxis(ax1, 'right');
text(ax1, x_lbl, 2.60, ...
    sprintf('\\leftarrow CNN-LSTM\nRMSE = %.2f°', CNN_RMSE), ...
    'FontName', 'Times New Roman', 'FontSize', 12, ...
    'Color', clr_cnn_r, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'Clipping', 'off');

% URLLC 1 ms 
text(ax1, x_lbl, 1.15, ...
    '\leftarrow URLLC  1 ms', ...
    'FontName', 'Times New Roman', 'FontSize', 12, ...
    'Color', clr_urllc, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'Clipping', 'off');

% CNN-LSTM Latency
text(ax1, x_lbl, 0.35, ...
    sprintf('\\leftarrow CNN-LSTM\nLatency = %.2f ms', CNN_time), ...
    'FontName', 'Times New Roman', 'FontSize', 12, ...
    'Color', clr_cnn_t, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'left', 'Clipping', 'off');

%% ─────────────────────────────────
yyaxis(ax1, 'right');
time_labels = {'0.15 ms', '0.60 ms', '1.50 ms'};
time_offsets = [0.25, 0.28, 0.30];

for i = 1:length(K_vals)
    flag = '';
    if TIME_CD(i) >= URLLC_ms, flag = '  x'; end
    text(ax1, K_vals(i) + 0.8, TIME_CD(i) + time_offsets(i), ...
        [time_labels{i} flag], ...
        'FontName', 'Times New Roman', 'FontSize', 12, ...
        'Color', clr_time, 'FontWeight', 'bold');
end

% RMSE 
yyaxis(ax1, 'left');
rmse_labels = {'3.50°', '3.49°', '3.20°'};
for i = 1:length(K_vals)
    text(ax1, K_vals(i) + 0.8, RMSE_CD(i) + 0.025, ...
        rmse_labels{i}, ...
        'FontName', 'Times New Roman', 'FontSize', 12, ...
        'Color', clr_rmse, 'FontWeight', 'bold');
end

%% ─── URLLC Exceeded ────────────────────────────
yyaxis(ax1, 'right');
text(ax1, 33, 5.5, 'K \geq 20: URLLC Exceeded', ...
    'FontName', 'Times New Roman', 'FontSize', 13, ...
    'Color', [0.75 0.10 0.10], 'FontAngle', 'italic', ...
    'FontWeight', 'bold');

%% ─── ──────────────────────────────
yyaxis(ax1, 'left');
xlabel(ax1, 'Iteration Count (K) for Baseline Algorithms', ...
    'FontName', 'Times New Roman', 'FontSize', 13, 'FontWeight', 'bold');
title(ax1, 'Baseline Fairness: CD RMSE and Latency vs. Iteration Count K', ...
    'FontName', 'Times New Roman', 'FontSize', 13, 'FontWeight', 'bold');

xlim(ax1, [3, 52]);
xticks(ax1, K_vals);
xticklabels(ax1, {'K = 5', 'K = 20', 'K = 50'});

%% ───────────────────────────────────────────
grid(ax1, 'on');
box(ax1, 'on');
ax1.GridAlpha      = 0.25;
ax1.GridLineStyle  = ':';
ax1.LineWidth      = 1.2;
set(ax1, 'FontName', 'Times New Roman', 'FontSize', 13);

% Dikey çizgi K=20'de (URLLC sınırı)
xline(ax1, 20, '-.k', 'LineWidth', 1.5, 'HandleVisibility', 'off');

%% ───────────────────────────────────────────────────
lgd = legend(ax1, [h1, h2, h3, h4, h5], ...
    'CD (K) — Phase-Aware RMSE [°]', ...
    sprintf('CNN-LSTM RMSE (%.2f°)', CNN_RMSE), ...
    'CD (K) — Inference Latency [ms]', ...
    '6G URLLC Limit (1 ms)', ...
    sprintf('CNN-LSTM Latency (%.2f ms)', CNN_time), ...
    'Location', 'northeast', ...
    'Box', 'on');
set(lgd, 'FontName', 'Times New Roman', 'FontSize', 12);

%% ───────────────────────────────────────────────────
exportgraphics(fig, 'Figure_Fairness_K.eps', 'ContentType', 'vector');
exportgraphics(fig, 'Figure_Fairness_K.jpg', 'Resolution', 300);
exportgraphics(fig, 'Figure_Fairness_K.png', 'Resolution', 600);

%% ──────────────────────────────────────────
fprintf('\n');
fprintf('=======================================================\n');
fprintf(' TABLO: CD Baseline Fairness — K = {5, 20, 50}\n');
fprintf('=======================================================\n');
fprintf('%-6s %12s %14s %10s\n', 'K', 'RMSE [°]', 'Latency [ms]', 'URLLC?');
fprintf('-------------------------------------------------------\n');
fprintf('%-6s %12.2f %14.3f %10s  <- onerilen\n', ...
    'CNN', CNN_RMSE, CNN_time, 'OK');
fprintf('-------------------------------------------------------\n');
for i = 1:length(K_vals)
    flag = 'OK';
    if TIME_CD(i) >= URLLC_ms, flag = 'ASILI x'; end
    fprintf('K=%-4d %12.2f %14.3f %10s\n', ...
        K_vals(i), RMSE_CD(i), TIME_CD(i), flag);
end
fprintf('=======================================================\n');
fprintf('\nAna Mesaj (Hakem 3 Z5):\n');
fprintf('  K=50 CD RMSE = 3.20 > CNN-LSTM 2.77  (hala ustun)\n');
fprintf('  K=50 Latency = 1.50 ms > URLLC 1 ms  (kullanilamaz)\n');
fprintf('=======================================================\n');
fprintf('\nKaydedildi:\n');
fprintf('  Figure_Fairness_K.eps\n');
fprintf('  Figure_Fairness_K.jpg (300 dpi)\n');
fprintf('  Figure_Fairness_K.png (600 dpi)\n');
