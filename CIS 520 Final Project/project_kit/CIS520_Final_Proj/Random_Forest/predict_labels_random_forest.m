function pred_labels=predict_labels_random_forest(train_inputs,train_labels,test_inputs)

    % for now, we will just use our 2nd PCA on this
    % Project X onto new PC's
    X = train_inputs;
    Y_train = train_labels;
    tweets = X(:,22:2021);
    n = size(X, 1);
    p = size(train_labels, 2);

    % Compute PC's
    % Don't need to normalize b/c all columns in same units
    [coeff_tweets, score_tweets,~,~,~] = pca(tweets);
    tweet_reduced_X = [X(:, 1:21), score_tweets(:, 1:35)];

    % Need to project test set onto PC's
    test_tweets = test_inputs(:,22:2021);
    test_tweets_PC = test_tweets * coeff_tweets(:, 1:35); % PC-reduced scores
    X_test = [test_inputs(:, 1:21), test_tweets_PC];

    % Compute RF model
    RF_models = cell(1, p);
    y_hat = NaN(size(test_inputs, 1), p);
    for j = 1:p
            % Using 500 trees
            RF_models{1, j} = TreeBagger(500, tweet_reduced_X,...
                Y_train(:, j), 'Method','regression');
            % Make prediction
            y_hat(:, j) = predict(RF_models{1, j}, X_test);     
    end
    pred_labels = y_hat;

end