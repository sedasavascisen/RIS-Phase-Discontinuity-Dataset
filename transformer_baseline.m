% transformer_baseline.m
% Trains a single-encoder Transformer and a plain MLP for comparison with
% the proposed CNN-LSTM (Table 4 in paper).
%
% Transformer config: 4 attention heads, d_model=64, 2 encoder blocks.
% MLP config: 512-512-256 hidden units, direct phase output (no sin/cos).
%
% Requires: RIS_dataset_large_2D.mat

clear; close all; clc;
rng(42);

datafile = 'RIS_dataset_large_2D.mat';
S = load(datafile);
F = S.F;  G = S.G;  h_d = S.h_d;  THETA = S.THETA;

numSamples = size(F, 1);
timeSteps  = size(F, 2);

% sequence format (Transformer + CNN-LSTM)
Xseq = cell(numSamples, 1);
Yseq = cell(numSamples, 1);
for i = 1:numSamples
    Xseq{i} = [real(F(i,:)); imag(F(i,:)); real(G(i,:)); imag(G(i,:))];
    Yseq{i} = [sin(THETA(i,:)); cos(THETA(i,:))];
end

% flat format (MLP)
X_flat = [real(F), imag(F), real(G), imag(G)];
Y_flat = [sin(THETA), cos(THETA)];

% split (same as main script: 70/15/15)
idx    = randperm(numSamples);
nTrain = round(0.70 * numSamples);
nVal   = round(0.15 * numSamples);
nTest  = numSamples - nTrain - nVal;

trainIdx = idx(1:nTrain);
valIdx   = idx(nTrain+1:nTrain+nVal);
testIdx  = idx(nTrain+nVal+1:end);

Xtrain_seq = Xseq(trainIdx);  Ytrain_seq = Yseq(trainIdx);
Xval_seq   = Xseq(valIdx);    Yval_seq   = Yseq(valIdx);
Xtest_seq  = Xseq(testIdx);

THETA_true = THETA(testIdx, :);

Xtrain_flat = X_flat(trainIdx,:);
Ytrain_d    = THETA(trainIdx,:);   % direct phase labels for MLP
Xtest_flat  = X_flat(testIdx,:);

% z-score normalization for MLP
mu  = mean(Xtrain_flat);
sig = std(Xtrain_flat) + 1e-9;
Xtrain_flat = (Xtrain_flat - mu) ./ sig;
Xtest_flat  = (Xtest_flat  - mu) ./ sig;

fprintf('Split: train=%d, val=%d, test=%d\n\n', nTrain, nVal, nTest);

%% --- Transformer ---
fprintf('Training Transformer...\n');

d_model = 64;  numHeads = 4;  keyDim = d_model / numHeads;

layers_tr = [
    sequenceInputLayer(4, 'Name', 'in')
    fullyConnectedLayer(d_model, 'Name', 'embed')
    reluLayer('Name', 'relu_emb')
    selfAttentionLayer(numHeads, keyDim, 'Name', 'attn1')
    additionLayer(2, 'Name', 'add1')
    layerNormalizationLayer('Name', 'ln1')
    fullyConnectedLayer(d_model, 'Name', 'ff1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.1, 'Name', 'dr1')
    selfAttentionLayer(numHeads, keyDim, 'Name', 'attn2')
    additionLayer(2, 'Name', 'add2')
    layerNormalizationLayer('Name', 'ln2')
    fullyConnectedLayer(d_model, 'Name', 'ff2')
    reluLayer('Name', 'relu2')
    dropoutLayer(0.1, 'Name', 'dr2')
    fullyConnectedLayer(2, 'Name', 'fc_out')
    regressionLayer('Name', 'out')
];

lg_tr = layerGraph(layers_tr);
lg_tr = connectLayers(lg_tr, 'relu_emb', 'add1/in2');
lg_tr = connectLayers(lg_tr, 'ln1',      'add2/in2');

opts_tr = trainingOptions('adam', ...
    'MaxEpochs', 50, 'MiniBatchSize', 256, ...
    'InitialLearnRate', 1e-3, 'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, 'LearnRateDropPeriod', 15, ...
    'ValidationData', {Xval_seq, Yval_seq}, 'ValidationFrequency', 50, ...
    'Shuffle', 'every-epoch', 'ExecutionEnvironment', 'gpu', ...
    'Plots', 'training-progress', 'Verbose', 1);

t0_tr       = tic;
net_tr      = trainNetwork(Xtrain_seq, Ytrain_seq, lg_tr, opts_tr);
train_time_tr = toc(t0_tr);

t0_inf = tic;
Ypred_tr = predict(net_tr, Xtest_seq);
t_inf_tr = toc(t0_inf) / nTest * 1e3;

THETA_pred_tr = zeros(nTest, timeSteps);
for i = 1:nTest
    THETA_pred_tr(i,:) = atan2(Ypred_tr{i}(1,:), Ypred_tr{i}(2,:));
end
err_tr   = angle(exp(1j*(THETA_pred_tr - THETA_true)));
rmse_tr  = sqrt(mean(err_tr(:).^2)) * (180/pi);
fprintf('Transformer  RMSE=%.2f deg,  inf=%.3f ms/sample\n', rmse_tr, t_inf_tr);

%% --- MLP ---
fprintf('\nTraining MLP...\n');

layers_mlp = [
    featureInputLayer(size(Xtrain_flat, 2), 'Name', 'in')
    fullyConnectedLayer(512, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'r1')
    dropoutLayer(0.2, 'Name', 'd1')
    fullyConnectedLayer(512, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'r2')
    dropoutLayer(0.2, 'Name', 'd2')
    fullyConnectedLayer(256, 'Name', 'fc3')
    reluLayer('Name', 'r3')
    fullyConnectedLayer(timeSteps, 'Name', 'fc_out')
    regressionLayer('Name', 'out')
];

opts_mlp = trainingOptions('adam', ...
    'MaxEpochs', 50, 'MiniBatchSize', 256, ...
    'InitialLearnRate', 1e-3, 'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, 'LearnRateDropPeriod', 15, ...
    'Shuffle', 'every-epoch', 'Plots', 'none', 'Verbose', 0);

net_mlp = trainNetwork(Xtrain_flat, Ytrain_d, layers_mlp, opts_mlp);

t0_mlp    = tic;
pred_mlp  = wrapToPi(predict(net_mlp, Xtest_flat));
t_inf_mlp = toc(t0_mlp) / nTest * 1e3;

err_mlp  = angle(exp(1j*(pred_mlp - THETA_true)));
rmse_mlp = sqrt(mean(err_mlp(:).^2)) * (180/pi);
fprintf('MLP (direct) RMSE=%.2f deg,  inf=%.3f ms/sample\n', rmse_mlp, t_inf_mlp);

%% --- Summary table ---
RMSE_cnn = 2.77;  TIME_cnn = 0.08;

fprintf('\n');
fprintf('%-32s  %8s  %10s\n', 'Method', 'RMSE [°]', 'Time [ms]');
fprintf('%s\n', repmat('-', 1, 55));
fprintf('%-32s  %8.2f  %10.3f   <- proposed\n', 'CNN-LSTM (sin/cos)', RMSE_cnn, TIME_cnn);
fprintf('%-32s  %8.2f  %10.3f\n', 'Transformer (4-head, d=64)', rmse_tr, t_inf_tr);
fprintf('%-32s  %8.2f  %10.3f   (phase discontinuity issue)\n', 'MLP (direct phase)', rmse_mlp, t_inf_mlp);
fprintf('%s\n', repmat('-', 1, 55));
fprintf('CNN-LSTM vs Transformer: %.2f° lower RMSE, %.1fx faster\n', ...
        rmse_tr - RMSE_cnn, t_inf_tr / TIME_cnn);

%% save
save('transformer_comparison_results.mat', ...
     'rmse_tr', 't_inf_tr', 'rmse_mlp', 't_inf_mlp', ...
     'RMSE_cnn', 'TIME_cnn', 'THETA_pred_tr', 'pred_mlp', 'THETA_true');
fprintf('\nSaved: transformer_comparison_results.mat\n');
