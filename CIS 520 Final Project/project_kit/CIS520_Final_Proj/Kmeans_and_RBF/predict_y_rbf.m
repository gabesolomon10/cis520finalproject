% Predict Y given RBF, kernel

function y_hat = predict_y_rbf(X_test, rbf_model, cluster_centers)

    X = X_test;
    n = size(X_test, 1);
    p = size(rbf_model, 2);
    y_hat = NaN(n,p);
    % Standardize X_test
    standardized_X = X./(max(X) - min(X));
    
    % Find cluster assignments
    kernel_matrix = get_kernel_matrices(standardized_X, cluster_centers);
    X = [ones(n, 1), kernel_matrix];
    % Compute Y_hat
    for i = 1:n
        for j = 1:p
            y_hat(i,j) = dot(rbf_model(:, j)', X(i,:));
        end
    end
    
    