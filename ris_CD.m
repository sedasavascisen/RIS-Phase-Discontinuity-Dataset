% ================================================================
% ris_CD.m  –  Coordinate Descent (CD) Phase Optimisation for RIS
% ================================================================
% KULLANIM:
%   [Theta_all, h_eff_est, rmse_deg, t_ms] = ris_CD(F, G, h_d, maxIter)
%
% GİRİŞLER:
%   F        : [numSamples x N]  BS-RIS kanal vektörü
%   G        : [numSamples x N]  RIS-UE kanal vektörü
%   h_d      : [numSamples x 1]  Doğrudan BS-UE kanalı
%   maxIter  : iterasyon sayısı  (default: 5)
%
% ÇIKIŞLAR:
%   Theta_all  : [numSamples x N] optimum faz konfigürasyonu (complex)
%   h_eff_est  : [numSamples x 1] efektif kanal kestirimi
%   rmse_deg   : skaler – Phase-aware RMSE (derece cinsinden)
%   t_ms       : skaler – ortalama çıkarım süresi (ms/örnek)
%
% NOT: Bu dosya makalede bildirilen "CD, K=5" karşılaştırmasını
%      tam olarak yeniden üretmektedir.  K=20 ve K=50 için
%      baseline_fairness_K_analysis.m betiğini çalıştırın.
% ================================================================

function [Theta_all, h_eff_est, rmse_deg, t_ms] = ris_CD(F, G, h_d, maxIter)

if nargin < 4 || isempty(maxIter)
    maxIter = 5;
end

[numSamples, N] = size(F);
Theta_all  = ones(numSamples, N);   % complex faz matrisi
h_eff_est  = zeros(numSamples, 1);

t_total = 0;

for s = 1:numSamples
    Fs = F(s,:);
    Gs = G(s,:);
    hd = h_d(s);

    % Başlangıç: tüm fazlar = 1 (theta_n = 0)
    Theta = ones(1, N);

    t0 = tic;
    for it = 1:maxIter
        % Her element için sırayla optimal faz güncellesi
        for n = 1:N
            % n. eleman dışındaki tüm elemanların katkısı
            idx = true(1, N);
            idx(n) = false;
            H_minus_n = hd + sum(Gs(idx) .* (Theta(idx) .* Fs(idx)));

            % n. eleman için optimal faz (kapalı-form)
            theta_n_opt = angle(H_minus_n) - angle(Gs(n) * Fs(n));
            theta_n_opt = mod(theta_n_opt + pi, 2*pi) - pi;  % wrapToPi
            Theta(n) = exp(1j * theta_n_opt);
        end
    end
    t_total = t_total + toc(t0);

    Theta_all(s,:) = Theta;
    h_eff_est(s)   = hd + sum(Gs .* (Theta .* Fs));
end

t_ms = (t_total / numSamples) * 1e3;   % ms/örnek

% ---- Optimal faz etiketi (kapalı-form referans) ----
THETA_true = -angle(G) - angle(F);     % θ* = -∠(G·F)
THETA_pred = angle(Theta_all);

% Phase-aware RMSE (rad → derece)
phase_err  = angle(exp(1j * (THETA_pred - THETA_true)));
rmse_rad   = sqrt(mean(phase_err(:).^2));
rmse_deg   = rmse_rad * (180/pi);

fprintf('CD (K=%d) | Phase-aware RMSE = %.2f°  |  avg time = %.3f ms/sample\n', ...
        maxIter, rmse_deg, t_ms);
end
