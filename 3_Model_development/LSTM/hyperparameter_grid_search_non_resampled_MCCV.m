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
para_split.num_MCCV = 10; % Number of Monte-Carlo cross validation (training times)
para_split.w_method = 'none'; % distribution for weighted bootstrapping; 'Burr','normal','none'
% para_split.w_adj = 1.3; % the adjusting factor for pdf from w_method to derive the weighting factor
% para_split.num_resampling = 200000; % Number of samples in bootstrapping 

[index.Train,index.Validation] = train_validation_split(time_valid,index.Test,para_split,T);

save('Grid_search_non_resampled.mat','time_valid','T','para_split','index','-v7.3')


%% Define the grid of hyperparameters
grid_search.numHiddenUnits = [40, 60, 80, 100]; % Number of hidden units
grid_search.dropout = [0.2, 0.4, 0.6];     % Dropout ratio
grid_search.L2 = [1e-4, 2e-4, 1e-3, 2e-3]; % L2 regularization
grid_search.maxEpoch = [20, 30, 50];       % Maximum number of epochs

save('Grid_search_non_resampled.mat','grid_search','-append')


%% Prepare for grid search
Y_pred_test = cell(para_split.num_MCCV,1);
results = cell(para_split.num_MCCV,1);
bestModel = cell(para_split.num_MCCV,1);
bestResult = cell(para_split.num_MCCV,1);
bestY_pred = cell(para_split.num_MCCV,1);

XTest = X(index.Test);
YTest = T(index.Test);

%% Train the model
model_type = 'LSTM'; % Choose the model architecture: 'LSTM'; 'GRU'; 'BiLSTM'

tmp_name = 'Grid_search_non_resampled_tmp';
for i = 1:para_split.num_MCCV
    XTrain = X(index.Train{i});
    YTrain = T(index.Train{i});
    XValidation = X(index.Validation{i});
    YValidation = T(index.Validation{i});

    [Y_pred_test{i},results{i},bestModel{i},bestResult{i},bestY_pred{i}] ...
        = hyperparameter_grid_search(XTrain,YTrain,XValidation,YValidation,XTest,YTest,grid_search,model_type,tmp_name);
    
    save('Grid_search_non_resampled.mat','Y_pred_test','results','best*','model_type','-append')
end




