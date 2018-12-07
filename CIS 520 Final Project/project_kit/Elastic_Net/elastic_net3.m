%Elastic Net
function y_hat = elastic_net3(train_inputs,train_labels,test_inputs)
rng(85);


%Pull in data
X = train_inputs;
Y = train_labels;
n = size(X,1);
p = size(X,2);

%Create an added column based on the sum of column 1 and 3
b = X(:,1)+X(:,3);
c = zeros(size(X,1),1);

%Indicator variable: 1 if column 1 + 3 are above 30
for i = 1:size(b)
    if b(i) > 30
        c(i) = 1;
    end
end
 
%Standardize predictors? Lasso does automatically
 
%Take logs of all probabilities - because probabilities belong in log space
logs1 = log10(X(:,1:6));
logs2 = log10(X(:,12:13));
logs3 = log10(X(:,22:end));
X_logs = [logs1 X(:,7:11) logs2 X(:,14:22) logs3 c];
 
%Can only use elastic net for one outcome variable, so create 9 separate
%elastic nets on the data and combine at the end
 
%Separate into different outcome variables
Y1 = Y(:,1);
Y2 = Y(:,2);
Y3 = Y(:,3);
Y4 = Y(:,4);
Y5 = Y(:,5);
Y6 = Y(:,6);
Y7 = Y(:,7);
Y8 = Y(:,8);
Y9 = Y(:,9);
 
%Lower numLambda may be helpful
%Run initial elastic net
 
%Tune hyperparameters CV and numlambda for speed - maybe reltol?
 
%Elastic net for Y1
[B1,FitInfo1] = lasso(X_logs,Y1,'Alpha',0.75,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx1Lambda1SE = FitInfo1.Index1SE;
coef1 = B1(:,idx1Lambda1SE);
coef0_1 = FitInfo1.Intercept(idx1Lambda1SE);
 
%Elastic net for Y2
[B2,FitInfo2] = lasso(X_logs,Y2,'Alpha',0.75,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx2Lambda1SE = FitInfo2.Index1SE;
coef2 = B2(:,idx2Lambda1SE);
coef0_2 = FitInfo2.Intercept(idx2Lambda1SE);
 
%Elastic net for Y3
[B3,FitInfo3] = lasso(X_logs,Y3,'Alpha',0.9,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx3Lambda1SE = FitInfo3.Index1SE;
coef3 = B3(:,idx3Lambda1SE);
coef0_3 = FitInfo3.Intercept(idx3Lambda1SE);
 
%Elastic net for Y4
[B4,FitInfo4] = lasso(X_logs,Y4,'Alpha',0.75,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx4Lambda1SE = FitInfo4.Index1SE;
coef4 = B4(:,idx4Lambda1SE);
coef0_4 = FitInfo4.Intercept(idx4Lambda1SE);
 
%Elastic net for Y5
[B5,FitInfo5] = lasso(X_logs,Y5,'Alpha',0.75,'CV',3, 'NumLambda', 10, 'RelTol', 1e-4);
idx5Lambda1SE = FitInfo5.Index1SE;
coef5 = B5(:,idx5Lambda1SE);
coef0_5 = FitInfo5.Intercept(idx5Lambda1SE);
 
%Elastic net for Y6
[B6,FitInfo6] = lasso(X_logs,Y6,'Alpha',0.75,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx6Lambda1SE = FitInfo6.Index1SE;
coef6 = B6(:,idx6Lambda1SE);
coef0_6 = FitInfo6.Intercept(idx6Lambda1SE);
 
%Elastic net for Y7
[B7,FitInfo7] = lasso(X_logs,Y7,'Alpha',0.75,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx7Lambda1SE = FitInfo7.Index1SE;
coef7 = B7(:,idx7Lambda1SE);
coef0_7 = FitInfo7.Intercept(idx7Lambda1SE);
 
%Elastic net for Y8
[B8,FitInfo8] = lasso(X_logs,Y8,'Alpha',0.75,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx8Lambda1SE = FitInfo8.Index1SE;
coef8 = B8(:,idx8Lambda1SE);
coef0_8 = FitInfo8.Intercept(idx8Lambda1SE);
 
%Elastic net for Y9
[B9,FitInfo9] = lasso(X_logs,Y9,'Alpha',0.75,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx9Lambda1SE = FitInfo9.Index1SE;
coef9 = B9(:,idx9Lambda1SE);
coef0_9 = FitInfo9.Intercept(idx9Lambda1SE);

%Now adjust the test data to match the adjustments we made to the training
%data
Xtest = test_inputs;

b = Xtest(:,1)+Xtest(:,3);
c = zeros(size(Xtest,1),1);
for i = 1:size(b)
    if b(i) > 30
        c(i) = 1;
    end
end

logs1 = log10(Xtest(:,1:6));
logs2 = log10(Xtest(:,12:13));
logs3 = log10(Xtest(:,22:end));
Xtest_logs = [logs1 Xtest(:,7:11) logs2 Xtest(:,14:22) logs3 c];

%Once we've made all sufficient changes, we can actually make predictions
%using the lasso model

%y_hats are predicted
y_hat1 = Xtest_logs*coef1 + coef0_1;
y_hat2 = Xtest_logs*coef2 + coef0_2;
y_hat3 = Xtest_logs*coef3 + coef0_3;
y_hat4 = Xtest_logs*coef4 + coef0_4;
y_hat5 = Xtest_logs*coef5 + coef0_5;
y_hat6 = Xtest_logs*coef6 + coef0_6;
y_hat7 = Xtest_logs*coef7 + coef0_7;
y_hat8 = Xtest_logs*coef8 + coef0_8;
y_hat9 = Xtest_logs*coef9 + coef0_9;

%Final project is concatenation of y predictions
y_hat = [y_hat1 y_hat2 y_hat3 y_hat4 y_hat5 y_hat6 y_hat7 y_hat8 y_hat9];
 
end