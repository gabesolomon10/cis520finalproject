function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)

%pred_labels = elastic_net(train_inputs,train_labels,test_inputs);

% for now, we will just use our 2nd PCA on this
% Project X onto new PC's
X = train_inputs;
Y_train = train_labels;
tweets = log(X(:,22:2021));
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
% 
% % Now want to take into account former predictions
% RF_models = cell(1, p);
% y_hat = NaN(size(test_inputs, 1), p);
% for j = 1:p
%         % Using 500 trees
%         if j == 1
%             RF_models{1, j} = TreeBagger(500, tweet_reduced_X,...
%             Y_train(:, j), 'Method','regression');
%             % Make prediction
%             y_hat(:, j) = predict(RF_models{1, j}, X_test);    
%         else
%             % For each successive tree, we also now know the previous Y's
%             RF_models{1, j} = TreeBagger(500, [tweet_reduced_X, ...
%                 Y_train(:, 1:(j-1))], Y_train(:, j), 'Method','regression');
% 
%             % Make prediction
%             y_hat(:, j) = predict(RF_models{1, j}, [X_test, y_hat(:, 1:(j-1))]);
%         end
%         
% end
% pred_labels = .5*y_hat + .5*pred_labels1;


% First compute covariance matrix of standardized Y's
% Add variables one at a time that are most correlated



%Now do GMM clustering
% rng(4);
% 
% %socioecon = X(:,1:21);
% %12 is best cluster
% 
% C = 5;
% GMModel = fitgmdist(tweet_reduced_X,C,'Start', 'randSample','RegularizationValue',0.1);
% cluster_assignment_train = cluster(GMModel, tweet_reduced_X);
% cluster_X = [tweet_reduced_X cluster_assignment_train];
% 
% cluster_assignment_test = cluster(GMModel, X_test);
% X_test = [X_test cluster_assignment_test];

% % Compute RF on clusters overall
% RF_models_GMM = cell(1, p);
% y_hat = NaN(size(test_inputs, 1), p);
% for j = 1:p
%         % Using 500 trees
%         RF_models_GMM{1, j} = TreeBagger(500, cluster_X,...
%             Y_train(:, j), 'Method','regression');
%         % Make prediction
%         y_hat(:, j) = predict(RF_models_GMM{1, j}, X_test);     
% end
% pred_labels = y_hat;



% % Compute RF model
% RF_models_GMM = cell(C, p);
% y_hat_GMM = NaN(size(test_inputs, 1), p);
% 
% for c = 1:C
%     disp(strcat("cluster", num2str(c)))
%     for j = 1:p
%         % Using 1000 trees
%         ind = find(cluster_X(:,end) == c);
%         X_clustered = cluster_X(ind,:);
%         
%         RF_models{c, j} = TreeBagger(500, X_clustered(:,1:end-1),...
%             Y_train(ind, j), 'Method','regression');
%         
%         % Make prediction
%         
%         %First add row number to each row
% 
%         %Get indices of cluster
%         orig_indices = find(X_test(:,end) == c);
%         ind = find(X_test(:,end) == c);
%         
%         temp_test = X_test(ind,1:end-1);
%        
%         % temporary on each loop to hold indices of each cluster in orig d
%         results = predict(RF_models{c, j}, temp_test);
%         
%         %Loop through results matrix to find right place
%         for i = 1:size(results,1)
%             y_hat_GMM(orig_indices(i),j) = results(i,:);
%         end
%                 
%     end
% end
% pred_labels = y_hat_GMM;

%pred_labels = (pred_labels_1 + pred_labels_2) / 2;


end