% plot_results.m
% Visualizes CNN-LSTM test results after running theta_NN_multioutput.m.
% Expects cnn_lstm_test_results.mat in the current folder.

clear; close all; clc;

R = load('cnn_lstm_test_results.mat');

THETA_pred = R.THETA_pred;
THETA_true = R.THETA_true;
phase_err  = R.phase_err;
rmse_deg   = R.rmse_deg;
t_per_sample = R.t_per_sample;

[nTest, N] = size(THETA_true);

fprintf('=== CNN-LSTM Test Results ===\n');
fprintf('RMSE       : %.2f deg\n', rmse_deg);
fprintf('Inference  : %.3f ms/sample\n', t_per_sample);

for tau = [1 3 5 10]
    sr = mean(abs(phase_err(:)) < tau*pi/180);
    fprintf('SR@%2d deg  : %.1f%%\n', tau, sr*100);
end

%% 1) per-element RMSE bar
rmse_elem = sqrt(mean(phase_err.^2, 1)) * (180/pi);

figure('Color', 'w', 'Position', [50 50 750 320]);
bar(rmse_elem, 'FaceColor', [0.20 0.50 0.80]);
xlabel('RIS element index');
ylabel('Phase-aware RMSE [°]');
title(sprintf('Per-element RMSE  (mean = %.2f°)', rmse_deg));
yline(rmse_deg, '--r', 'LineWidth', 1.5, 'Label', 'Mean');
grid on;

%% 2) error histogram
figure('Color', 'w', 'Position', [50 420 500 320]);
histogram(phase_err(:) * 180/pi, 50, 'FaceColor', [0.20 0.50 0.80], 'EdgeColor', 'none');
xlabel('Phase error [°]');
ylabel('Count');
title('Phase error distribution (all test samples)');
xline( rmse_deg, '--r', 'LineWidth', 1.5);
xline(-rmse_deg, '--r', 'LineWidth', 1.5);
grid on;

%% 3) example predictions (first 5 samples)
figure('Color', 'w', 'Position', [600 50 650 420]);
for i = 1:5
    plot(THETA_true(i,:)*180/pi,  '-b', 'LineWidth', 1.2);  hold on;
    plot(THETA_pred(i,:)*180/pi,  '-r', 'LineWidth', 1.2);
end
legend('True', 'Predicted');
xlabel('RIS element index');
ylabel('Phase [°]');
title(sprintf('Phase predictions — 5 test samples  (RMSE=%.2f°)', rmse_deg));
grid on;

%% 4) boxplot of error per element
figure('Color', 'w', 'Position', [600 500 750 360]);
boxplot(phase_err * 180/pi, 'Labels', 1:N);
xlabel('RIS element index');
ylabel('Phase error [°]');
title('Per-element error distribution (boxplot)');
grid on;
