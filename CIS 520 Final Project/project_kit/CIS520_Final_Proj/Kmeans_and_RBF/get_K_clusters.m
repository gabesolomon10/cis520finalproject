% Now that we've done CV, compute K-means on training data

function [idx, c] = get_K_clusters(X)

    % First, standardized each column
    standardized_X = X./(max(X) - min(X));
    
    [idx, c] = kmeans(standardized_X, 80, 'Start', 'sample');
end