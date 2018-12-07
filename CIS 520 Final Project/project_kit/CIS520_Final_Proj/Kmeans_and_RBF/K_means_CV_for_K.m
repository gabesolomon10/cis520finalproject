% Do K means clustering on input data

% First, standardized each column
standardized_X = X./(max(X) - min(X));

% Now compute K clusters, using L2 distance
n = size(standardized_X, 1);
cv_errors = zeros(61, 1);
i = 0;
% CV to find optimal k
for K = [3,5:5:300]
    i = i + 1;
    disp(i);
    p = randperm(n);
    % Select each decile of the permutation to be the part left out
    for j = 1:10
        chunk_size = size(p, 2) / 10;
        test_indices = p(((j-1)*chunk_size + 1):j*chunk_size);
        train_indices = p(~ismember(p, test_indices));
        [kmeans_model, C] = kmeans(standardized_X(train_indices,:), K,...
            'Start', 'sample');
        % Assign points from out of sample
        [validation_labels, err] = get_closest_cluster(...
            standardized_X(test_indices,:), C);
        % Look at sum of out of sample L2 distances for a given K
        cv_errors(i) = cv_errors(i) + err;
    end
    cv_errors(i) = cv_errors(i) / 10; % avg cv errors
end

% Plot to find optimal K
% Roughly 150 clusters looks good
plot([3,5:5:300], cv_errors)
xlabel('K')
ylabel('Avg CV SSE')

% Good after about 80 clusters, we can always reduce after