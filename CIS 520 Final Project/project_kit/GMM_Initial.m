%Do GMM on socioeconomic factors
rng(4);

socioecon = train_inputs(:,1:21);
GMModel = fitgmdist(socioecon,100,'RegularizationValue',0.1);

%Initialize with actual values
GMModel2 = fitgmdist(socioecon,100,'Start', 'randSample','RegularizationValue',0.1);


AIC = zeros(1,200);
GMModels = cell(1,200);
for k = 1:200
    GMModels{k} = fitgmdist(socioecon,k,'Start', 'randSample','RegularizationValue',0.1);
    AIC(k)= GMModels{k}.AIC;
end

[minAIC,numComponents] = min(AIC);
numComponents

%Best number of components is 2? Does this make sense
tweets = train_inputs(:,22:2021);

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

%Do GMM on tweets
AIC_tweets = zeros(1,200);
GMModels_tweets = cell(1,200);
for k = 1:200
    GMModels{k} = fitgmdist(tweets_new,k,'Start', 'randSample','RegularizationValue',0.1);
    AIC(k)= GMModels{k}.AIC;
end

[minAIC,numComponents] = min(AIC);
numComponents

%12 components is best (according to AIC)

%120 compone