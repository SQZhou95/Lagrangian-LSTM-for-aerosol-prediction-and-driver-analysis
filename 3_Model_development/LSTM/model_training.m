function [netSet, Y, Y_mean, Stat, Stat_Y_mean] = model_training(X, T, index, hyperparameters, model_type, T_bin, tmp_name)
% Y_mean: mean of Y derived from the first N MCCV experiments, without ranking
% Stat_Y_mean: Stat corresponding to Y_mean
%
% Get the number of MCCV experiments
num_MCCV = length(index.Train);

% Get the number of Test samples
num_Test = length(index.Test);

% Initiate the output
netSet = cell(num_MCCV,1);
Y.Train = cell(num_MCCV,1);
Y.Validation = cell(num_MCCV,1);
Y.Test = NaN(num_Test,num_MCCV);
Y_mean.Test = NaN(num_Test,num_MCCV);

Stat.RMSE.Train = NaN(num_MCCV,1);
Stat.RMSE.Validation = NaN(num_MCCV,1);
Stat.RMSE.Test = NaN(num_MCCV,1);
Stat.R2.Train = NaN(num_MCCV,1);
Stat.R2.Validation = NaN(num_MCCV,1);
Stat.R2.Test = NaN(num_MCCV,1);
Stat.Slope.Train = NaN(num_MCCV,1);
Stat.Slope.Validation = NaN(num_MCCV,1);
Stat.Slope.Test = NaN(num_MCCV,1);

bin_length = length(T_bin)-1;
Stat_Y_mean.R2.Test = NaN(num_MCCV,1);
Stat_Y_mean.RMSE.Test = NaN(num_MCCV,1);
Stat_Y_mean.RMSE.Test_bin = NaN(num_MCCV,bin_length);
Stat_Y_mean.Slope.Test = NaN(num_MCCV,1);


% Training
XTest = X(index.Test);
TTest = T(index.Test);
for i = 1:num_MCCV
    numChannels = size(X{1},1);
    numResponses = size(T,2);

    % Training and Validation
    XTrain = X(index.Train{i});
    TTrain = T(index.Train{i});
    XValidation = X(index.Validation{i});
    TValidation = T(index.Validation{i});

    % Define the model layers
    switch model_type
        case 'LSTM'
            layers = [ ...
                sequenceInputLayer(size(XTrain{1}, 1), Normalization="zscore")
                lstmLayer(numHiddenUnits, OutputMode="last")
                dropoutLayer(dropout)
                fullyConnectedLayer(1) % size(YTrain,2)
                regressionLayer];
        case 'GRU'
            layers = [ ...
                sequenceInputLayer(size(XTrain{1}, 1), Normalization="zscore")
                gruLayer(numHiddenUnits, OutputMode="last")
                dropoutLayer(dropout)
                fullyConnectedLayer(1) % size(YTrain,2)
                regressionLayer];
        case 'BiLSTM'
            layers = [ ...
                sequenceInputLayer(size(XTrain{1}, 1), Normalization="zscore")
                bilstmLayer(numHiddenUnits, OutputMode="last")
                dropoutLayer(dropout)
                fullyConnectedLayer(1) % size(YTrain,2)
                regressionLayer];
        otherwise
            error('Please provide a valid architecture type: LSTM, GRU, or BiLSTM')
    end

    options = trainingOptions("adam", ...
        MaxEpochs=hyperparameters.maxEpoch, ...
        ValidationData={XValidation TValidation}, ... 
        MiniBatchSize=512, ... %%% For resampled one; Use default value for non-resampled one
        L2Regularization=hyperparameters.L2, ...
        OutputNetwork="best-validation-loss", ...
        InitialLearnRate=0.005, ... %%% The dafault is 0.001 for adam solver
        SequenceLength="shortest", ...
        Plots="none", ...
        Verbose= 1, ...
        Shuffle = 'every-epoch', ...
        ExecutionEnvironment='gpu');

    net = trainNetwork(XTrain, TTrain, layers, options);
    netSet{i} = net;
    disp(['Training_',num2str(i),' has been completed. ',datestr(now)])


    % Statistics calculation
    Y.Train{i} = predict(net,XTrain, SequenceLength="shortest");
    Y.Validation{i} = predict(net,XValidation, SequenceLength="shortest");
    Y.Test(:,i) = predict(net,XTest, SequenceLength="shortest");

    Stat.RMSE.Train(i) = sqrt(sum((TTrain-Y.Train{i}).^2)/length(TTrain));
    Stat.RMSE.Validation(i) = sqrt(sum((TValidation-Y.Validation{i}).^2)/length(TValidation));
    Stat.RMSE.Test(i) = sqrt(sum((TTest-Y.Test(:,i)).^2)/length(TTest));

    Stat.R2.Train(i) = corr(TTrain,Y.Train{i},'type','pearson')^2;
    Stat.R2.Validation(i) = corr(TValidation,Y.Validation{i},'type','pearson')^2;
    Stat.R2.Test(i) = corr(TTest,Y.Test(:,i),'type','pearson')^2;

    fit_train = polyfit(TTrain,Y.Train{i},1);
    Stat.Slope.Train(i) = fit_train(1);
    fit_validation = polyfit(TValidation,Y.Validation{i},1);
    Stat.Slope.Validation(i) = fit_validation(1);
    fit_test = polyfit(TTest,Y.Test(:,i),1);
    Stat.Slope.Test(i) = fit_test(1);
    

    Y_mean.Test(:,i) = mean(Y.Test(:,1:i),2);
    Stat_Y_mean.RMSE.Test(i) = sqrt(sum((TTest-Y_mean.Test(:,i)).^2)/length(TTest));
    Stat_Y_mean.R2.Test(i) = corr(TTest,Y_mean.Test(:,i),'type','pearson')^2;

    fit_test_mean = polyfit(TTest,Y_mean.Test(:,i),1);
    Stat_Y_mean.Slope.Test(i) = fit_test_mean(1);

    disp(['RMSE_Test = ',num2str(Stat.RMSE.Test(i)),'; RMSE_Test_mean = ',num2str(Stat_Y_mean.RMSE.Test(i))])
    disp(['R2_Test = ',num2str(Stat.R2.Test(i)),'; R2_Test_mean = ',num2str(Stat_Y_mean.R2.Test(i))])
    disp(['Slope_Test = ',num2str(Stat.Slope.Test(i)),'; Slope_Test_mean = ',num2str(Stat_Y_mean.Slope.Test(i))])

    % RMSE at different CCN concentration ranges
    for j = 1:bin_length
        idx_tmp = TTest>=T_bin(j) & TTest<T_bin(j+1);

        Stat_Y_mean.RMSE.Test_bin(i,j) = sqrt(sum((TTest(idx_tmp)-Y_mean.Test(idx_tmp,i)).^2)/sum(idx_tmp));
    end


    % Save out the temporary results after every 10 trainings
    if mod(i,10) == 0
        save(tmp_name,'netSet', 'Y', 'Y_mean', 'Stat', 'Stat_Y_mean')
    end
end
