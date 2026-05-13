% RIS_dataset_large_2D.m
% Generates the channel dataset used to train and test the CNN-LSTM model.
%
% Setup: 2D RIS panel (8x4 = 32 elements), Rayleigh fading + scatterers,
%        closed-form optimal phase labels (Eq. 8 in paper).
%
% Outputs (saved to RIS_dataset_large_2D.mat):
%   F, G   : BS-RIS and RIS-UE channels [numSamples x N]
%   h_d    : direct BS-UE channel       [numSamples x 1]
%   THETA  : optimal phase labels       [numSamples x N]
%   TX_pos, RX_pos, elemPos : geometry info
%
% Run time ~10 min on a standard laptop (single core).
% For N=64 or N=128, set M and K below and re-run.

clear; close all; clc;
rng(42);

%% parameters
numSamples = 40000;
M = 8;   % RIS x-dimension
K = 4;   % RIS y-dimension
N = M*K; % 32 elements total

fc = 3e9;
c  = 3e8;
lambda = c/fc;
kwave  = 2*pi/lambda;

numScatter  = 6;
roomSize    = [40 20 6];  % m
scatter_g   = 0.15;
Pt          = 1;
noise_var   = 1e-3;

%% RIS element positions (lambda/2 spacing)
ris_center = [roomSize(1)/2, roomSize(2)/2, 3];
d = lambda/2;

x0 = ris_center(1) - (M-1)/2 * d;
y0 = ris_center(2) - (K-1)/2 * d;

elemPos = zeros(N, 3);
idx = 1;
for ii = 1:M
    for jj = 1:K
        elemPos(idx,:) = [x0+(ii-1)*d,  y0+(jj-1)*d,  ris_center(3)];
        idx = idx+1;
    end
end

%% pre-allocate
F      = complex(zeros(numSamples, N));
G      = complex(zeros(numSamples, N));
h_d    = complex(zeros(numSamples, 1));
THETA  = zeros(numSamples, N);
TX_pos = zeros(numSamples, 3);
RX_pos = zeros(numSamples, 3);

%% main loop
fprintf('Generating dataset  N=%d (%dx%d),  samples=%d\n', N, M, K, numSamples);
tic;

for s = 1:numSamples
    tx = [rand*roomSize(1),  rand*roomSize(2),  1+rand];
    rx = [rand*roomSize(1),  rand*roomSize(2),  1+rand];
    TX_pos(s,:) = tx;
    RX_pos(s,:) = rx;

    scat_pos = rand(numScatter, 3) .* roomSize;

    % direct link
    d_dir  = norm(tx - rx);
    h_d(s) = (randn+1j*randn)/sqrt(2) * exp(-1j*kwave*d_dir) / d_dir;

    % per-element channels
    for n = 1:N
        el = elemPos(n,:);

        d_tx = norm(tx - el);
        d_rx = norm(el - rx);

        f0 = (randn+1j*randn)/sqrt(2) * exp(-1j*kwave*d_tx) / d_tx;
        g0 = (randn+1j*randn)/sqrt(2) * exp(-1j*kwave*d_rx) / d_rx;

        fs = 0;  gs = 0;
        for m = 1:numScatter
            da = norm(tx - scat_pos(m,:));  db = norm(scat_pos(m,:) - el);
            dc = norm(el - scat_pos(m,:));  dd = norm(scat_pos(m,:) - rx);

            fs = fs + scatter_g*(randn+1j*randn)/sqrt(2)*exp(-1j*kwave*(da+db))/(da*db);
            gs = gs + scatter_g*(randn+1j*randn)/sqrt(2)*exp(-1j*kwave*(dc+dd))/(dc*dd);
        end

        F(s,n) = f0 + fs;
        G(s,n) = g0 + gs;
    end

    % optimal phase: closed-form solution (Eq. 8)
    THETA(s,:) = wrapToPi(-angle(F(s,:)) - angle(G(s,:)));

    if mod(s, 2000) == 0
        fprintf('  %d / %d  (%.1f s)\n', s, numSamples, toc);
    end
end
toc;

%% quick stats check
fprintf('\nPhase label stats:\n');
fprintf('  mean = %.4f  (expect ~0)\n', mean(THETA(:)));
fprintf('  std  = %.4f  (expect pi/sqrt(3) = %.4f)\n', std(THETA(:)), pi/sqrt(3));

% SNR with and without RIS
h_ris = h_d + sum(G .* exp(1j*THETA) .* F, 2);
snr_no  = 10*log10(Pt * abs(h_d).^2   / noise_var);
snr_ris = 10*log10(Pt * abs(h_ris).^2 / noise_var);
fprintf('\nSNR without RIS:  median = %.1f dB\n', median(snr_no));
fprintf('SNR with RIS:     median = %.1f dB\n', median(snr_ris));
fprintf('RIS gain:         median = %.1f dB\n', median(snr_ris - snr_no));

%% save
save('RIS_dataset_large_2D.mat', ...
     'F', 'G', 'h_d', 'THETA', 'TX_pos', 'RX_pos', 'elemPos', ...
     'N', 'M', 'K', 'numScatter', 'roomSize', 'fc', 'Pt', 'noise_var', '-v7.3');

fprintf('\nSaved to RIS_dataset_large_2D.mat\n');
