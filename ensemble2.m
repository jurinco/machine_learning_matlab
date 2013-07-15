%% Intialization
clear; close all; clc

addpath 'io'
addpath 'classifiers/libsvm-3.13'

data_path = 'data/';
model_path = 'classifiers/models/';
predictions_path = 'predictions/';


%% Predict using the normalized poly-9 model
load([model_path 'svm_sphog_n_32k_poly9.mat']);
load([data_path 'tr_labels.mat']);

nTrain = 32000;
nTest = 10000;


% Normalized Data
load([data_path 'feats_sphog_12_n.mat']);

fprintf('Predicting labels for the last %d training instances.\n',nTest);

tic
[pred accuracy dec_vals] = svmpredict(tr_labels(nTrain+1:end,:), tr_feats_sphog_n(nTrain+1:end,:), model, '-b 1');
display_elapsed_time

pred_filename = 'svm_sphog_n_32k_model_n_poly9.mat';
save([predictions_path pred_filename], 'pred','accuracy','dec_vals');
clear pred accuracy dec_vals;


% Deskewed & Normalized Data
te_labels_filename = 'tr_labels';
load([data_path te_labels_filename '.mat']);
te_labels = eval(te_labels_filename);
te_labels = te_labels(nTrain+1:end,:);
clear te_labels_filename;

te_data_filename = 'feats_sphog_12_dn';
load([data_path data_filename '.mat']);

te_feats = eval(te_data_filename);
te_feats = te_feats(nTrain+1:end,:);
clear te_data_filename;


fprintf('Predicting labels for the last %d training instances.\n',nTest);

tic
[pred accuracy dec_vals] = svmpredict(te_labels, te_feats, model, '-b 1');
display_elapsed_time




fprintf('Predicting labels for the last %d training instances.\n',nTest);

tic
[pred accuracy dec_vals] = svmpredict(tr_labels(nTrain+1:end,:), tr_feats_sphog_dn(nTrain+1:end,:), model, '-b 1');
display_elapsed_time

pred_filename = 'svm_sphog_dn_32k_model_n_poly9.mat';
save([predictions_path pred_filename], 'pred','accuracy','dec_vals');
clear pred accuracy dec_vals;
