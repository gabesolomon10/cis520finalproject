% Random Forest for predicting outcomes
% Randomly divide data into testing and training sets

% Set seed
rng(10);

% Split data into testing and training indices, 85/15
n = size(train_labels, 1);
p = randperm(n);
train_indices = p(1:round(.85 * n));
test_indices = p(round(.85 * n) + 1:n);

% Now split into sets for each dimension reduced figure
% use tweet_reduced_X, X_new, double_deduced_data
train1 = X_new(train_indices, :); % using PCA from both features
test1 = X_new(test_indices,:);

train2 = tweet_reduced_X(train_indices, :);
test2 = tweet_reduced_X(test_indices,:);

train3 = double_reduced_data(train_indices, :);
test3 = double_reduced_data(test_indices,:);

% Read in outcome vector
Y = train_labels;
Y_train = Y(train_indices, :);
Y_test = Y(test_indices, :);

% Predict Random forest for each feature
p = size(Y, 2);
train_sets = {train1, train2, train3};
test_sets = {test1, test2, test3};
RF_models = cell(3, 9);
errs = NaN(3, 9);
for i = 1:3
    for j = 1:p
        % Using 500 trees
        RF_models{i, j} = TreeBagger(500, train_sets{i},...
            Y_train(:, j), 'Method','regression');
        % Make prediction
        y_hat = predict(RF_models{i, j}, test_sets{i});
        % Get test error
        errs(i, j) = error_metric(y_hat, Y_test(:, j));
    end
end
        
disp(errs);

% It appears that the random forest models using the tweet_reduced data
% does by far the best (set 2)