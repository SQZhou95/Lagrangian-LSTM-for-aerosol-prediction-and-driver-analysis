function [Y_pred_test,results,bestModel,bestResult,bestY_pred] ...
        = hyperparameter_grid_search(XTrain,YTrain,XValidation,YValidation,XTest,YTest,grid_search,model_type,tmp_name)

num_grid.numHiddenUnits = length(grid_search.numHiddenUnits);
num_grid.dropout = length(grid_search.dropout);
num_grid.L2 = length(grid_search.L2);
num_grid.maxEpoch = length(grid_search.maxEpoch);

% % Initialize variables to store all results
% RMSE.Train = NaN(num_grid.numHiddenUnits, num_grid.dropout, num_grid.L2, num_grid.maxEpoch);
% RMSE.Validation = NaN(num_grid.numHiddenUnits, num_grid.dropout, num_grid.L2, num_grid.maxEpoch);
% RMSE.Test = NaN(num_grid.numHiddenUnits, num_grid.dropout, num_grid.L2, num_grid.maxEpoch);
% R2.Train = NaN(num_grid.numHiddenUnits, num_grid.dropout, num_grid.L2, num_grid.maxEpoch);
% R2.Validation = NaN(num_grid.numHiddenUnits, num_grid.dropout, num_grid.L2, num_grid.maxEpoch);
% R2.Test = NaN(num_grid.numHiddenUnits, num_grid.dropout, num_grid.L2, num_grid.maxEpoch);


% Initialize variables to store the best results
num_Test = length(XTest);
Y_pred_test = NaN(num_Test,num_grid.numHiddenUnits*num_grid.dropout*num_grid.L2*num_grid.maxEpoch);
bestModel = [];
bestY_pred = [];
results = [];
bestResult = struct();
bestRMSE_test = Inf;

% Grid search
p = 1;
for i = 1:num_grid.numHiddenUnits
    numHiddenUnits = grid_search.numHiddenUnits(i);

    for j = 1:num_grid.dropout
        dropout = grid_search.dropout(j);

        for k = 1:num_grid.L2
            l2Regularization = grid_search.L2(k);

            for h = 1:num_grid.maxEpoch
                maxEpoch = grid_search.maxEpoch(h);
                
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
                        

                % Set training options
                options = trainingOptions("adam", ...
                    MaxEpochs=maxEpoch, ...  % MiniBatchSize=512 --> for resampled one
                    MiniBatchSize=512, ...
                    ValidationData={XValidation YValidation}, ...
                    L2Regularization=l2Regularization, ...
                    OutputNetwork="best-validation-loss", ...
                    InitialLearnRate=0.005, ... %%% The dafault is 0.001 for adam solver
                    SequenceLength="shortest", ...
                    Plots="none", ...
                    Verbose= 1, ...
                    Shuffle = 'every-epoch');

                % Train the network
                try
                    net = trainNetwork(XTrain, YTrain, layers, options);
                    
                    % Validate the network
                    YPred.Train = predict(net, XTrain); % Predict on the Train set
                    YPred.Validation = predict(net, XValidation); % Predict on the Validation set
                    YPred.Test = predict(net, XTest); % Predict on the Test set
                    RMSE.Train = rmse(YTrain,YPred.Train); % Compute RMSE
                    RMSE.Validation = rmse(YValidation,YPred.Validation);
                    RMSE.Test = rmse(YTest,YPred.Test);
                    R2.Train = corr(YTrain,YPred.Train)^2; % Compute RMSE
                    R2.Validation = corr(YValidation,YPred.Validation)^2;
                    R2.Test = corr(YTest,YPred.Test)^2;
                    fit_Train = polyfit(YTrain,YPred.Train,1);
                    Slope.Train = fit_Train(1);
                    fit_Validation = polyfit(YValidation,YPred.Validation,1);
                    Slope.Validation = fit_Validation(1);
                    fit_Test = polyfit(YTest,YPred.Test,1);
                    Slope.Test = fit_Test(1);
                                       
                    % Save results
                    Y_pred_test(:,p) = YPred.Test;
                    p = p+1;

                    results = [results; struct( ...
                        'NumHiddenUnits', numHiddenUnits, ...
                        'Dropout', dropout, ...
                        'L2Regularization', l2Regularization, ...
                        'MaxEpoch', maxEpoch, ...
                        'RMSE_Train', RMSE.Train,...
                        'RMSE_Validation', RMSE.Validation,...
                        'RMSE_Test', RMSE.Test,...
                        'R2_Train', R2.Train,...
                        'R2_Validation', R2.Validation,...
                        'R2_Test', R2.Test,...
                        'Slope_Train', Slope.Train,...
                        'Slope_Validation', Slope.Validation,...
                        'Slope_Test',Slope.Test)];
                    
                    % Update the best model
                    if RMSE.Test < bestRMSE_test
                        bestRMSE_test = RMSE.Test;
                        bestModel = net;
                        bestY_pred = YPred;
                        bestResult = struct( ...
                            'NumHiddenUnits', numHiddenUnits, ...
                            'Dropout', dropout, ...
                            'L2Regularization', l2Regularization, ...
                            'MaxEpoch', maxEpoch, ...
                            'RMSE_Train', RMSE.Train,...
                            'RMSE_Validation', RMSE.Validation,...
                            'RMSE_Test', RMSE.Test,...
                            'R2_Train', R2.Train,...
                            'R2_Validation', R2.Validation,...
                            'R2_Test', R2.Test,...
                            'Slope_Train', Slope.Train,...
                            'Slope_Validation', Slope.Validation,...
                            'Slope_Test',Slope.Test);
                    end

                    disp(struct2table(results))
                    disp(struct2table(bestResult))
                catch ME
                    fprintf('Error encountered with hyperparameters: %s\n', ME.message);
                end
            end
            save([tmp_name,'.mat'],'Y_pred_test','results','bestModel','bestResult','bestY_pred','i','j','k','h')
        end
    end
end

% Display best results
disp('Best Hyperparameters:');
disp(bestResult);
