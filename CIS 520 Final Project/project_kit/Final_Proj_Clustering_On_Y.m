% Final Project Initial Clustering

% (1) Cluster based on outcomes
% First, we need to normalize each y variable to cluster on same scale

colnames = ['health_aamort', 'health_fairpoor', 'health_mentunh', ...
    'health_pcdiab', 'health_pcexcdrin', 'health_pcinact', ...
    'health_pcsmoker', 'health_physunh', 'heath_pcobese'];

% May also want to try standardizing later depending on results
Y = train_labels;
normalized_Y = zscore(Y);

% K-means cluster normalized Y
% First, cross validate K
n = size(Y, 1);
cv_errors = zeros(61, 1);
i = 0;
for K = [3,5:5:300]
    i = i + 1;
    p = randperm(n);
    % Select each decile of the permutation to be the part left out
    for j = 1:10
        chunk_size = size(p, 2) / 10;
        test_indices = p(((j-1)*chunk_size + 1):j*chunk_size);
        train_indices = p(~ismember(p, test_indices));
        [kmeans_model, C] = kmeans(normalized_Y(train_indices,:), K);
        % Assign points from out of sample
        [validation_labels, err] = get_closest_cluster(...
            normalized_Y(test_indices,:), C);
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

% Thus, our final model will be 150 clusters 
K = 150;
label_based_clusters = kmeans(normalized_Y, K);

% We can now map these clusters back to the feature space
% Or more specifically assign similar-valued test points to these clusters
% (Maybe with KNN)
% Then compute average cluster value of each variable & run regression over
% each cluster
