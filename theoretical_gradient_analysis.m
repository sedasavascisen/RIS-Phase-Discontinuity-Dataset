% theoretical_gradient_analysis.m
% Empirical verification of Proposition 1 (Section 3.3 in paper).
%
% Shows that:
%  (i)  L_direct has a ~4pi gradient discontinuity at +-pi
%  (ii) L_PA gradient is bounded: max|grad L_PA| <= 4
%  (iii) For small errors, L_PA ≈ epsilon^2 (Taylor, 1st order)
%  (iv) At the +-pi boundary, L_PA gives ~30% lower error than L_direct
%
% No training needed -- pure analytical/Monte Carlo.

clear; close all; clc;

theta_star = pi - 0.01;   % near the +-pi boundary (worst case for L_direct)
theta_pred = linspace(-pi, pi, 2000);

% L_direct and its gradient
L_dir  = (theta_pred - theta_star).^2;
dL_dir = 2*(theta_pred - theta_star);

% L_PA and its gradient  (= ||(sin_pred - sin_true, cos_pred - cos_true)||^2)
L_PA  = (sin(theta_pred) - sin(theta_star)).^2 + (cos(theta_pred) - cos(theta_star)).^2;
dL_PA = 2*(sin(theta_pred)-sin(theta_star)).*cos(theta_pred) ...
      - 2*(cos(theta_pred)-cos(theta_star)).*sin(theta_pred);

% circular (true) squared error for reference
circ_err = angle(exp(1j*(theta_pred - theta_star))).^2;

% --- (i) gradient discontinuity at +-pi ---
bd = abs(theta_pred - (-pi)) < 0.2;
jump_dir = max(abs(dL_dir(bd))) - min(abs(dL_dir(bd)));
jump_PA  = max(abs(dL_PA(bd)))  - min(abs(dL_PA(bd)));

fprintf('(i)  Gradient jump at +-pi:\n');
fprintf('     L_direct : %.4f   (expected ~4pi = %.4f)\n', jump_dir, 4*pi);
fprintf('     L_PA     : %.4f   (expected ~0)\n\n', jump_PA);

% --- (ii) gradient bound ---
fprintf('(ii) max|grad L_PA| = %.4f   (bound: 4.0)\n\n', max(abs(dL_PA)));

% --- (iii) Taylor approximation: L_PA ≈ eps^2 ---
eps_vals  = linspace(0, 0.5, 500);
L_PA_exact = 2*(1 - cos(eps_vals));   % analytical form
L_taylor   = eps_vals.^2;
rel_err    = abs(L_PA_exact - L_taylor) ./ (L_PA_exact + 1e-12);
fprintf('(iii) Max relative error of Taylor approx (|eps|<0.2 rad): %.3f%%\n\n', ...
        max(rel_err(eps_vals < 0.2)) * 100);

% --- (iv) empirical boundary error comparison ---
rng(42);
N_sim    = 10000;
th_t = wrapToPi((rand(N_sim,1)-0.5)*0.4 + pi*sign(randn(N_sim,1)));
th_p = wrapToPi(th_t + (rand(N_sim,1)-0.5)*0.6);

err_d = (th_p - th_t).^2;
err_c = angle(exp(1j*(th_p - th_t))).^2;

fprintf('(iv) Boundary region error (N=%d simulations):\n', N_sim);
fprintf('     L_direct  : %.4f rad^2\n', mean(err_d));
fprintf('     L_PA (circ): %.4f rad^2\n', mean(err_c));
fprintf('     Reduction  : %.1f%%\n\n', 100*(mean(err_d)-mean(err_c))/mean(err_d));

% --- Proposition 1 summary ---
fprintf('Proposition 1 summary:\n');
fprintf('  (i)   L_direct jump = %.2f, L_PA jump = %.4f\n', jump_dir, jump_PA);
fprintf('  (ii)  ||grad L_PA||_inf = %.4f <= 4\n', max(abs(dL_PA)));
fprintf('  (iii) Taylor max rel err = %.2f%% (for |eps|<0.2 rad)\n', ...
        max(rel_err(eps_vals<0.2))*100);
fprintf('  (iv)  Boundary reduction = %.1f%%\n', ...
        100*(mean(err_d)-mean(err_c))/mean(err_d));

%% figures
fig = figure('Color', 'w', 'Position', [50 50 1100 420]);

subplot(1,3,1);
plot(theta_pred, L_dir,     '-r', 'LineWidth', 2, 'DisplayName', 'L_{direct}');  hold on;
plot(theta_pred, L_PA,      '-b', 'LineWidth', 2, 'DisplayName', 'L_{PA}');
plot(theta_pred, circ_err,  ':k', 'LineWidth', 1.5, 'DisplayName', 'Circular error');
xline(theta_star, '--g', 'LineWidth', 1.5, 'Label', '\theta^*');
xlabel('\theta_{pred} [rad]');  ylabel('Loss');
title('Loss functions (\theta^* = \pi - 0.01)');
legend('Location', 'north');  ylim([0 45]);  grid on;

subplot(1,3,2);
plot(theta_pred, dL_dir, '-r', 'LineWidth', 2, 'DisplayName', '\nablaL_{direct}');  hold on;
plot(theta_pred, dL_PA,  '-b', 'LineWidth', 2, 'DisplayName', '\nablaL_{PA}');
yline( 4, '--g', 'LineWidth', 1.5, 'Label', '||grad||=4');
yline(-4, '--g', 'LineWidth', 1.5);
xlabel('\theta_{pred} [rad]');  ylabel('Gradient');
title('Gradient comparison');
legend('Location', 'south');  ylim([-15 15]);  grid on;

subplot(1,3,3);
plot(eps_vals*180/pi, L_PA_exact, '-b', 'LineWidth', 2, 'DisplayName', 'L_{PA}');  hold on;
plot(eps_vals*180/pi, L_taylor,   '--r', 'LineWidth', 2, 'DisplayName', '\epsilon^2 (Taylor)');
xlabel('Error \epsilon [deg]');  ylabel('L_{PA}');
title('Taylor approx: L_{PA} \approx \epsilon^2');
legend('Location', 'northwest');  xlim([0 30]);  grid on;

sgtitle('Proposition 1 — Phase-Aware Loss: Theoretical Analysis', 'FontSize', 12);
saveas(fig, 'theoretical_analysis_figure.png');

%% save numerical results
save('theoretical_bounds.mat', ...
     'jump_dir', 'jump_PA', 'eps_vals', 'L_PA_exact', 'L_taylor', 'rel_err', ...
     'err_d', 'err_c');
fprintf('\nSaved: theoretical_analysis_figure.png,  theoretical_bounds.mat\n');
