
% FIGURE 10: Scalability Analysis (N vs Latency)
% =========================================================

clear; close all; clc;

%% ────────────────────────────────────────────────────
N_vals = [32, 64, 128, 256];

% CNN-LSTM  O(N)    — N=32  0.08 ms,
time_cnn         = [0.08,  0.16,  0.32,  0.64 ];

% CD K=5    O(K·N²) — N=32  0.76 ms, 
time_cd_k5       = [0.76,  3.04, 12.16, 48.64 ];

% Transformer O(N²) — N=32  1.14 ms, 
time_transformer = [1.14,  4.56, 18.24, 72.96 ];

URLLC_limit = 1.0;   % 6G URLLC (ms)

%% ──────────────────────────────────────────────────
clr_cnn   = [0.000, 0.447, 0.741];
clr_cd    = [0.929, 0.694, 0.125];
clr_tr    = [0.850, 0.325, 0.098];
clr_urllc = [0.20,  0.20,  0.20 ];

%% ────────────────────────────────────────────────────
fig = figure('Color', 'white', 'Position', [100, 100, 720, 520]);
ax  = axes(fig);

%% ──────────────────────────────────────────────
h1 = semilogy(ax, N_vals, time_cnn, ...
    '-o', 'LineWidth', 2.5, 'MarkerSize', 9, ...
    'Color', clr_cnn, 'MarkerFaceColor', clr_cnn, ...
    'DisplayName', 'CNN-LSTM  [O(N)]');
hold(ax, 'on');

h2 = semilogy(ax, N_vals, time_cd_k5, ...
    '-^', 'LineWidth', 2.5, 'MarkerSize', 9, ...
    'Color', clr_cd, 'MarkerFaceColor', clr_cd, ...
    'DisplayName', 'CD (K=5)  [O(K \cdot N^{2})]');

h3 = semilogy(ax, N_vals, time_transformer, ...
    '-s', 'LineWidth', 2.5, 'MarkerSize', 9, ...
    'Color', clr_tr, 'MarkerFaceColor', clr_tr, ...
    'DisplayName', 'Transformer  [O(N^{2})]');

h4 = yline(ax, URLLC_limit, '--', ...
    'LineWidth', 2.2, 'Color', clr_urllc, ...
    'DisplayName', '6G URLLC Limit (1 ms)');

%% ─── URLLC─────────────────────────
x_fill = [N_vals(1), N_vals(end), N_vals(end), N_vals(1)];
y_fill = [URLLC_limit, URLLC_limit, 0.03, 0.03];
fill(ax, x_fill, y_fill, [0.85 1.0 0.85], ...
    'FaceAlpha', 0.18, 'EdgeColor', 'none', ...
    'HandleVisibility', 'off');

%% ─── CNN-LSTM ────────────────────
cnn_labels = {'0.08 ms', '0.16 ms', '0.32 ms', '0.64 ms'};
for i = 1:length(N_vals)
    text(ax, N_vals(i) + 6, time_cnn(i) * 1.45, ...
        cnn_labels{i}, ...
        'FontName', 'Times New Roman', 'FontSize', 12, ...
        'Color', clr_cnn, 'FontWeight', 'bold');
end

%% ───────────────
% N=64
text(ax, 64 + 6, time_cd_k5(2) * 1.55, ...
    '3.04 ms  x', ...
    'FontName', 'Times New Roman', 'FontSize', 12, ...
    'Color', clr_cd, 'FontWeight', 'bold');

% N=128
text(ax, 128 + 6, time_cd_k5(3) / 1.8, ...
    '12.2 ms  x', ...
    'FontName', 'Times New Roman', 'FontSize', 12, ...
    'Color', clr_cd, 'FontWeight', 'bold');

% N=256 
text(ax, 256 - 62, time_cd_k5(4) * 1.6, ...
    '48.6 ms  x', ...
    'FontName', 'Times New Roman', 'FontSize', 12, ...
    'Color', clr_cd, 'FontWeight', 'bold');

%% ───────────────────────────
% N=256
text(ax, 256 - 62, time_transformer(4) / 1.7, ...
    '73.0 ms  x', ...
    'FontName', 'Times New Roman', 'FontSize', 12, ...
    'Color', clr_tr, 'FontWeight', 'bold');

%% ───────────────────────────────────────────

text(ax, 34, 0.38, '\leftarrow URLLC Safe Zone', ...
    'FontName', 'Times New Roman', 'FontSize', 13, ...
    'Color', [0.05 0.45 0.05], 'FontAngle', 'italic');


text(ax, 34, 8.0, '\uparrow Exceeds URLLC Limit', ...
    'FontName', 'Times New Roman', 'FontSize', 13, ...
    'Color', [0.75 0.10 0.10], 'FontAngle', 'italic');

%% ───────────────────────────────────────
xlim(ax, [20, 290]);
ylim(ax, [0.05, 150]);

xticks(ax, N_vals);
xticklabels(ax, {'32', '64', '128', '256'});

xlabel(ax, 'Number of RIS Elements (N)', ...
    'FontName', 'Times New Roman', 'FontSize', 13, 'FontWeight', 'bold');
ylabel(ax, 'Inference Latency [ms] (Log Scale)', ...
    'FontName', 'Times New Roman', 'FontSize', 13, 'FontWeight', 'bold');
title(ax, 'Scalability Analysis: Inference Latency vs. N', ...
    'FontName', 'Times New Roman', 'FontSize', 13, 'FontWeight', 'bold');

%% ────────────────────────────────────────────
grid(ax, 'on');
box(ax, 'on');
ax.GridAlpha      = 0.25;
ax.GridLineStyle  = ':';
ax.LineWidth      = 1.2;
set(ax, 'FontName', 'Times New Roman', 'FontSize', 13);

%% ──────────────────────────────────
xline(ax,  64, ':', 'Color', [0.65 0.65 0.65], 'LineWidth', 1.1, ...
    'HandleVisibility', 'off');
xline(ax, 128, ':', 'Color', [0.65 0.65 0.65], 'LineWidth', 1.1, ...
    'HandleVisibility', 'off');

%% ─────────────────────────────
lgd = legend(ax, [h1, h2, h3, h4], ...
    'CNN-LSTM  [O(N)]', ...
    'CD (K=5)  [O(K \cdot N^{2})]', ...
    'Transformer  [O(N^{2})]', ...
    '6G URLLC Limit (1 ms)', ...
    'Location', 'northwest', ...
    'Box', 'on');
set(lgd, 'FontName', 'Times New Roman', 'FontSize', 12);

%% ──────────────────────────────────────────────────
exportgraphics(fig, 'Figure_10_Scalability.eps', 'ContentType', 'vector');
exportgraphics(fig, 'Figure_10_Scalability.jpg', 'Resolution', 300);
exportgraphics(fig, 'Figure_10_Scalability.png', 'Resolution', 600);

%% ──────────────────────────────────────────
fprintf('\n');
fprintf('==========================================================\n');
fprintf('  TABLO: Ölçeklenebilirlik — Çıkarım Süresi (ms/örnek)\n');
fprintf('==========================================================\n');
fprintf('%-5s %12s %14s %16s\n','N','CNN-LSTM','CD (K=5)','Transformer');
fprintf('----------------------------------------------------------\n');
for i = 1:length(N_vals)
    cd_f = ''; tr_f = '';
    if time_cd_k5(i)       > URLLC_limit, cd_f = ' [URLLC ASILI]'; end
    if time_transformer(i) > URLLC_limit, tr_f = ' [URLLC ASILI]'; end
    fprintf('N=%-6d %8.2f ms %10.2f ms%s %12.2f ms%s\n', ...
        N_vals(i), time_cnn(i), ...
        time_cd_k5(i),      cd_f, ...
        time_transformer(i), tr_f);
end
fprintf('==========================================================\n');
fprintf('CNN-LSTM: O(N)    — tum N degerlerinde URLLC ici  OK\n');
fprintf('CD K=5  : O(K*N2) — N>=64 URLLC disi             FAIL\n');
fprintf('Transf. : O(N2)   — N=32 dahil URLLC disi         FAIL\n');
fprintf('==========================================================\n');
fprintf('\nKaydedildi:\n');
fprintf('  Figure_10_Scalability.eps\n');
fprintf('  Figure_10_Scalability.jpg (300 dpi)\n');
fprintf('  Figure_10_Scalability.png (600 dpi)\n');
