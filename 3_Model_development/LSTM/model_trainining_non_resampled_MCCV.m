%% Load and prepare data for training
load('X_converted.mat') % Load the converted input data
load('CCN_corrected.mat') % Load the measured CCN data

% time matchup
idx_select = round(24*(time_valid(:,7)-CCN.time(1))+1);
CCN_valid = CCN.data(idx_select);

% remove samples with CCN data missing
idx_tmp = ~isnan(CCN_valid);
time_valid = time_valid(idx_tmp,:);
X = X(idx_tmp); % this is the input dataset: a cell array with each cell containing the multivariate time series input of a sample
CCN_valid = CCN_valid(idx_tmp);
T = log10(CCN_valid); % this is the target variable

%% Data split
index.Test = find(time_valid(:,1)==2022 & ismember(time_valid(:,2),[1 3 5 7 9 11])); % index of the testing set; monthly block

para_split.block_size = 'daily'; % 'hourly','daily','weekly','monthly'; Block size for train-validation data split
para_split.f_validation = 0.15; % Validation fraction
para_split.num_MCCV = 10; % Number of MCCV (training times)
para_split.w_method = 'none'; % distribution for weighted bootstrapping; 'Burr','normal','none'; Choose 'none' if not use weighted resampling strategy

[index.Train,index.Validation] = train_validation_split(time_valid,index.Test,para_split,T);

save('Train_non_resampled.mat','time_valid','T','para_split','index')


%% Define model type and hyperparameters
model_type = 'LSTM'; % Choose the model architecture: 'LSTM'; 'GRU'; 'BiLSTM'

hyperparameters.numHiddenUnits = 40; % Number of hidden units
hyperparameters.dropout = 0.2;     % Dropout ratio
hyperparameters.L2 = 2e-4; % L2 regularization
hyperparameters.maxEpoch = 20;       % Maximum number of epochs

save('Train_non_resampled.mat','hyperparameters','model_type','-append')


%% Training and evaluation
tmp_name = 'Train_non_resampled_tmp';
T_bin = [1 1.4 [1.5:0.1:2.8] 3]';

[netSet, Y, Y_mean, Stat, Stat_Y_mean] = model_training(X, T, index, hyperparameters, model_type, T_bin, tmp_name);

save('Train_non_resampled.mat','netSet', 'Y', 'Y_mean', 'Stat', 'Stat_Y_mean','-append')
