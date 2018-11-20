% Initial PCA & CCA for dimensionality reduction
%	- PCA on all features (Trevor)\
%	- PCA on tweets (Trevor)\
%	- PCA on demographics w/ clustered tweets (Trevor)\
%	- CCA of demographics vs tweets (Trevor)\

% In all cases, we will standardize the data
% X = ZV^T
X = train_inputs;
[coeff_all, score_all,~,~,explained_all]...
 = pca(X, 'VariableWeights','variance');

% plot variances to find reasonable cutoff for number of PCs
cum_explained = cumsum(explained_all);
plot(cumsum(explained_all)) % We get about 90% variance explained with 200 pcs
plot(explained_all)  % Appears to go way down by 50

% let's just select the features that gets 80% of variance explained,
% This happens at 123 variables included
coeff_all = coeff_all(1:1018, 1:123);

% Construct new X in reduced dimension feature space
X_new = score_all*coeff_all;
size(X_new)


% (2) Do PCA on tweets
tweets = X(:,22:2021);

% Don't need to normalize b/c all columns in same units
[coeff_tweets, score_tweets,~,~,explained_tweets] = pca(tweets);
% look at plot of explained
cum_explained = cumsum(explained_tweets);
plot(cum_explained)
% 8 tweet topic PCs explain 80% of variance
% 35 tweet topic PCs explain 90% of the variance
% USE THE 35 TOP TWEET PC'S NOW FOR 90%, THEN MAYBE REDUCE LATER
tweets_new = score_tweets * coeff_tweets(1:1018, 1:35);

% (3) Now use these reduced-dimension PC's in another PCA with
%     demographic features
tweet_reduced_X = [X(:, 1:21), tweets_new];
[coeff_combo, score_combo,~,~,explained_combo] = ...
    pca(tweet_reduced_X, 'VariableWeights','variance');
cum_explained = cumsum(explained_combo);
plot(cum_explained)
for i = 1:size(cum_explained,1)
    disp(i)
    disp(cum_explained(i))
end
% We get 95% reconstruction accuracy using 35 PC's. This is what we will
% use
double_reduced_data = score_combo * coeff_combo(:, 1:35);
