% Wrapper RBF function
% Takes in X_train, Y_train, X_test
% Returns Y_hat

function y_hat = predict_labels_rbf(X_train, Y_train, X_test)

    % Standardize training data
    standardized_X = X_train./(max(X_train) - min(X_train));

    % Generate clusters
    [~, c] = get_K_clusters(standardized_X);
    
    % Compute kernel matrix
    Z = get_kernel_matrices(standardized_X, c);
    
    % Compute RBF
    rbf_model = rbf(Z, Y_train);
    
    % Make predictions
    y_hat = predict_y_rbf(X_test, rbf_model, c);

end