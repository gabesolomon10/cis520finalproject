%Do GMM on socioeconomic factors
rng(4);

socioecon = train_inputs(:,1:21);
GMModel = fitgmdist(socioecon,100,'RegularizationValue',0.1);

%Initialize with actual values
GMModel2 = fitgmdist(socioecon,100,'Start', 'randSample','RegularizationValue',0.1);


AIC = zeros(1,11);
GMModels = cell(1,11);
for k = 1:20:201
    GMModels{k} = fitgmdist(socioecon,100,'Start', 'randSample','RegularizationValue',0.1);
    AIC(k)= GMModels{k}.AIC;
end

[minAIC,numComponents] = min(AIC);
numComponents

%Best number of components is 2? Does this make sense