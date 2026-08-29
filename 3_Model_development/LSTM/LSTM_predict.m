function y = LSTM_predict(x,netSet)
y_tmp = NaN(length(x),length(netSet));
for i = 1:length(netSet)
    y_tmp(:,i) = predict(netSet{i},x, SequenceLength="shortest", ExecutionEnvironment="gpu");
end
y = mean(y_tmp,2);
end
