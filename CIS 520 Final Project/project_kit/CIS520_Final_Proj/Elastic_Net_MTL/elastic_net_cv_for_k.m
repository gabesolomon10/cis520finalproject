% Cross validate k, dimensionality of latent basis
% In elastic net

function [coeff, score, explained] = elastic_net_cv_for_k(train_inputs,train_labels,test_inputs)
    rng(4);

    X = train_inputs;

    % Standardize Y (maybe after)
    Ymatx = train_labels;

    n = size(X,1);
    p = size(X,2);
    T = size(train_labels, 2);

    %Standardize predictors? Lasso does automatically

    %Take logs of probabilities? Worth a shot
    logs = log10(X(:,22:end));
    X_logs = [X(:,1:21) logs];

    Xtest = test_inputs;
    logs = log10(Xtest(:,22:end));
    X_test_logs = [Xtest(:,1:21) logs];

    %Can only use elastic net for one outcome variable

    %Separate into different outcome variables
    Y = cell(1,T);
    yhat_test = cell(1,T);
    yhat_train = cell(1,T);
    B = zeros(T, p+1);
    for i = 1:T
        Y{i} = Ymatx(:,i);
    end

    mdl_X = X_logs;
    mdl_X_test = X_test_logs;

    %Lower numLambda may be helpful
    %Run initial elastic net

    %Tune hyperparameters CV and numlambda for speed - maybe reltol?

    % Train successive models
    for i = 1:T
        disp(i)

        % Add previous predicted values to model
        %for j = 2:i
        %    mdl_X = [mdl_X, yhat_train{i-1}];
        %    mdl_X_test = [mdl_X_test, yhat_test{i-1}];
        %end 

        [Beta,FitInfo] = lasso(mdl_X,Y{i},'Alpha',0.75,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
        idx1Lambda1SE = FitInfo.Index1SE;
        coef = Beta(:,idx1Lambda1SE);
        coef0 = FitInfo.Intercept(idx1Lambda1SE);
        B(i,:) = [coef0, coef'];

        %yhat_train{i} = mdl_X*coef + coef0;
        %yhat_test{i} = mdl_X_test*coef + coef0;

    end

    % We now have a (p+1)xT matrix, B, of coefficients
    % We will now try to pull out some of the covariance structure to get at a
    % shared information model
    % Maybe we need to standardize first?
    [coeff, score, ~, ~, explained] = pca(B);
end