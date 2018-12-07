%Elastic net alpha tester

X = train_inputs;
Y = train_labels;

X_train = X(randpermutation(1:900),:);
Y_train = Y(randpermutation(1:900),:);
X_test = X(randpermutation(901:1019),:);
Y_test = Y(randpermutation(901:1019),:);

logs = log10(X_train(:,22:end));
X_train_logs = [X_train(:,1:21) logs];


logs = log10(X_test(:,22:end));
X_test_logs = [X_test(:,1:21) logs];
%Standardize predictors? Lasso does automatically

%Can only use elastic net for one outcome variable

%Separate into different outcome variables
Y1 = Y_train(:,1);
Y2 = Y_train(:,2);
Y3 = Y_train(:,3);
Y4 = Y_train(:,4);
Y5 = Y_train(:,5);
Y6 = Y_train(:,6);
Y7 = Y_train(:,7);
Y8 = Y_train(:,8);
Y9 = Y_train(:,9);

Y1test = Y_test(:,1);
Y2test = Y_test(:,2);
Y3test = Y_test(:,3);
Y4test = Y_test(:,4);
Y5test = Y_test(:,5);
Y6test = Y_test(:,6);
Y7test = Y_test(:,7);
Y8test = Y_test(:,8);
Y9test = Y_test(:,9);


%Test on Y1

[B1,FitInfo1] = lasso(X_train_logs,Y1,'Alpha',0.5,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx1Lambda1SE = FitInfo1.Index1SE;
coef1 = B1(:,idx1Lambda1SE);
coef0_1 = FitInfo1.Intercept(idx1Lambda1SE);

[B2,FitInfo2] = lasso(X_train_logs,Y1,'Alpha',0.7,'CV',3, 'NumLambda', 20, 'RelTol', 1e-3);
idx2Lambda1SE = FitInfo2.Index1SE;
coef2 = B2(:,idx1Lambda1SE);
coef0_2 = FitInfo2.Intercept(idx1Lambda1SE);

[B3,FitInfo3] = lasso(X_train_logs,Y1,'Alpha',0.9,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx3Lambda1SE = FitInfo3.Index1SE;
coef3 = B3(:,idx3Lambda1SE);
coef0_3 = FitInfo3.Intercept(idx3Lambda1SE);

y_hat1 = X_test_logs*coef1 + coef0_1;
y_hat2 = X_test_logs*coef2 + coef0_2;
y_hat3 = X_test_logs*coef3 + coef0_3;

error1_1 = sqrt((norm(y_hat1 - Y1test,2))^2/120)/(max(Y1) - min(Y1))
error2_1 = sqrt((norm(y_hat2 - Y1test,2))^2/120)/(max(Y1) - min(Y1))
error3_1 = sqrt((norm(y_hat3 - Y1test,2))^2/120)/(max(Y1) - min(Y1))
%alpha of .9 is the best


%Now on Y2
[B1,FitInfo1] = lasso(X_train_logs,Y2,'Alpha',0.5,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx1Lambda1SE = FitInfo1.Index1SE;
coef1 = B1(:,idx1Lambda1SE);
coef0_1 = FitInfo1.Intercept(idx1Lambda1SE);

[B2,FitInfo2] = lasso(X_train_logs,Y2,'Alpha',0.7,'CV',3, 'NumLambda', 20, 'RelTol', 1e-3);
idx2Lambda1SE = FitInfo2.Index1SE;
coef2 = B2(:,idx1Lambda1SE);
coef0_2 = FitInfo2.Intercept(idx1Lambda1SE);

[B3,FitInfo3] = lasso(X_train_logs,Y2,'Alpha',0.9,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx3Lambda1SE = FitInfo3.Index1SE;
coef3 = B3(:,idx3Lambda1SE);
coef0_3 = FitInfo3.Intercept(idx3Lambda1SE);

y_hat1 = X_test_logs*coef1 + coef0_1;
y_hat2 = X_test_logs*coef2 + coef0_2;
y_hat3 = X_test_logs*coef3 + coef0_3;

error1_1 = sqrt((norm(y_hat1 - Y2test,2))^2/120)/(max(Y2) - min(Y2))
error2_1 = sqrt((norm(y_hat2 - Y2test,2))^2/120)/(max(Y2) - min(Y2))
error3_1 = sqrt((norm(y_hat3 - Y2test,2))^2/120)/(max(Y2) - min(Y2))
%Either .5 or .9 are best

%Test for Y3
[B1,FitInfo1] = lasso(X_train_logs,Y3,'Alpha',0.5,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx1Lambda1SE = FitInfo1.Index1SE;
coef1 = B1(:,idx1Lambda1SE);
coef0_1 = FitInfo1.Intercept(idx1Lambda1SE);

[B2,FitInfo2] = lasso(X_train_logs,Y3,'Alpha',0.7,'CV',3, 'NumLambda', 20, 'RelTol', 1e-3);
idx2Lambda1SE = FitInfo2.Index1SE;
coef2 = B2(:,idx1Lambda1SE);
coef0_2 = FitInfo2.Intercept(idx1Lambda1SE);

[B3,FitInfo3] = lasso(X_train_logs,Y3,'Alpha',0.9,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx3Lambda1SE = FitInfo3.Index1SE;
coef3 = B3(:,idx3Lambda1SE);
coef0_3 = FitInfo3.Intercept(idx3Lambda1SE);

y_hat1 = X_test_logs*coef1 + coef0_1;
y_hat2 = X_test_logs*coef2 + coef0_2;
y_hat3 = X_test_logs*coef3 + coef0_3;

error1_1 = sqrt((norm(y_hat1 - Y3test,2))^2/120)/(max(Y3) - min(Y3))
error2_1 = sqrt((norm(y_hat2 - Y3test,2))^2/120)/(max(Y3) - min(Y3))
error3_1 = sqrt((norm(y_hat3 - Y3test,2))^2/120)/(max(Y3) - min(Y3))
%.5 is best

%Now Y4
[B1,FitInfo1] = lasso(X_train_logs,Y4,'Alpha',0.5,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx1Lambda1SE = FitInfo1.Index1SE;
coef1 = B1(:,idx1Lambda1SE);
coef0_1 = FitInfo1.Intercept(idx1Lambda1SE);

[B2,FitInfo2] = lasso(X_train_logs,Y4,'Alpha',0.7,'CV',3, 'NumLambda', 20, 'RelTol', 1e-3);
idx2Lambda1SE = FitInfo2.Index1SE;
coef2 = B2(:,idx1Lambda1SE);
coef0_2 = FitInfo2.Intercept(idx1Lambda1SE);

[B3,FitInfo3] = lasso(X_train_logs,Y4,'Alpha',0.9,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx3Lambda1SE = FitInfo3.Index1SE;
coef3 = B3(:,idx3Lambda1SE);
coef0_3 = FitInfo3.Intercept(idx3Lambda1SE);

y_hat1 = X_test_logs*coef1 + coef0_1;
y_hat2 = X_test_logs*coef2 + coef0_2;
y_hat3 = X_test_logs*coef3 + coef0_3;

error1_1 = sqrt((norm(y_hat1 - Y4test,2))^2/120)/(max(Y4) - min(Y4))
error2_1 = sqrt((norm(y_hat2 - Y4test,2))^2/120)/(max(Y4) - min(Y4))
error3_1 = sqrt((norm(y_hat3 - Y4test,2))^2/120)/(max(Y4) - min(Y4))
%.9 is best

%Y5
[B1,FitInfo1] = lasso(X_train_logs,Y5,'Alpha',0.5,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx1Lambda1SE = FitInfo1.Index1SE;
coef1 = B1(:,idx1Lambda1SE);
coef0_1 = FitInfo1.Intercept(idx1Lambda1SE);

[B2,FitInfo2] = lasso(X_train_logs,Y5,'Alpha',0.7,'CV',3, 'NumLambda', 20, 'RelTol', 1e-3);
idx2Lambda1SE = FitInfo2.Index1SE;
coef2 = B2(:,idx1Lambda1SE);
coef0_2 = FitInfo2.Intercept(idx1Lambda1SE);

[B3,FitInfo3] = lasso(X_train_logs,Y5,'Alpha',0.9,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx3Lambda1SE = FitInfo3.Index1SE;
coef3 = B3(:,idx3Lambda1SE);
coef0_3 = FitInfo3.Intercept(idx3Lambda1SE);

y_hat1 = X_test_logs*coef1 + coef0_1;
y_hat2 = X_test_logs*coef2 + coef0_2;
y_hat3 = X_test_logs*coef3 + coef0_3;

error1_1 = sqrt((norm(y_hat1 - Y5test,2))^2/120)/(max(Y5) - min(Y5))
error2_1 = sqrt((norm(y_hat2 - Y5test,2))^2/120)/(max(Y5) - min(Y5))
error3_1 = sqrt((norm(y_hat3 - Y5test,2))^2/120)/(max(Y5) - min(Y5))
% 5 is particularly hard! .5 and .9 are tossups - .1066 error though

%Y6
[B1,FitInfo1] = lasso(X_train_logs,Y6,'Alpha',0.5,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx1Lambda1SE = FitInfo1.Index1SE;
coef1 = B1(:,idx1Lambda1SE);
coef0_1 = FitInfo1.Intercept(idx1Lambda1SE);

[B2,FitInfo2] = lasso(X_train_logs,Y6,'Alpha',0.7,'CV',3, 'NumLambda', 20, 'RelTol', 1e-3);
idx2Lambda1SE = FitInfo2.Index1SE;
coef2 = B2(:,idx1Lambda1SE);
coef0_2 = FitInfo2.Intercept(idx1Lambda1SE);

[B3,FitInfo3] = lasso(X_train_logs,Y6,'Alpha',0.9,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx3Lambda1SE = FitInfo3.Index1SE;
coef3 = B3(:,idx3Lambda1SE);
coef0_3 = FitInfo3.Intercept(idx3Lambda1SE);

y_hat1 = X_test_logs*coef1 + coef0_1;
y_hat2 = X_test_logs*coef2 + coef0_2;
y_hat3 = X_test_logs*coef3 + coef0_3;

error1_1 = sqrt((norm(y_hat1 - Y6test,2))^2/120)/(max(Y6) - min(Y6))
error2_1 = sqrt((norm(y_hat2 - Y6test,2))^2/120)/(max(Y6) - min(Y6))
error3_1 = sqrt((norm(y_hat3 - Y6test,2))^2/120)/(max(Y6) - min(Y6))
%.5 and .9 also tied - .0815, a bit higher than normal - .6 a bit better (a
%bit)

%Y7
[B1,FitInfo1] = lasso(X_train_logs,Y7,'Alpha',0.5,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx1Lambda1SE = FitInfo1.Index1SE;
coef1 = B1(:,idx1Lambda1SE);
coef0_1 = FitInfo1.Intercept(idx1Lambda1SE);

[B2,FitInfo2] = lasso(X_train_logs,Y7,'Alpha',0.7,'CV',3, 'NumLambda', 20, 'RelTol', 1e-3);
idx2Lambda1SE = FitInfo2.Index1SE;
coef2 = B2(:,idx1Lambda1SE);
coef0_2 = FitInfo2.Intercept(idx1Lambda1SE);

[B3,FitInfo3] = lasso(X_train_logs,Y7,'Alpha',0.9,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx3Lambda1SE = FitInfo3.Index1SE;
coef3 = B3(:,idx3Lambda1SE);
coef0_3 = FitInfo3.Intercept(idx3Lambda1SE);

y_hat1 = X_test_logs*coef1 + coef0_1;
y_hat2 = X_test_logs*coef2 + coef0_2;
y_hat3 = X_test_logs*coef3 + coef0_3;

error1_1 = sqrt((norm(y_hat1 - Y7test,2))^2/120)/(max(Y7) - min(Y7))
error2_1 = sqrt((norm(y_hat2 - Y7test,2))^2/120)/(max(Y7) - min(Y7))
error3_1 = sqrt((norm(y_hat3 - Y7test,2))^2/120)/(max(Y7) - min(Y7))
%.9 is best

%Y8
[B1,FitInfo1] = lasso(X_train_logs,Y8,'Alpha',0.5,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx1Lambda1SE = FitInfo1.Index1SE;
coef1 = B1(:,idx1Lambda1SE);
coef0_1 = FitInfo1.Intercept(idx1Lambda1SE);

[B2,FitInfo2] = lasso(X_train_logs,Y8,'Alpha',0.7,'CV',3, 'NumLambda', 20, 'RelTol', 1e-3);
idx2Lambda1SE = FitInfo2.Index1SE;
coef2 = B2(:,idx1Lambda1SE);
coef0_2 = FitInfo2.Intercept(idx1Lambda1SE);

[B3,FitInfo3] = lasso(X_train_logs,Y8,'Alpha',0.9,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx3Lambda1SE = FitInfo3.Index1SE;
coef3 = B3(:,idx3Lambda1SE);
coef0_3 = FitInfo3.Intercept(idx3Lambda1SE);

y_hat1 = X_test_logs*coef1 + coef0_1;
y_hat2 = X_test_logs*coef2 + coef0_2;
y_hat3 = X_test_logs*coef3 + coef0_3;

error1_1 = sqrt((norm(y_hat1 - Y8test,2))^2/120)/(max(Y8) - min(Y8))
error2_1 = sqrt((norm(y_hat2 - Y8test,2))^2/120)/(max(Y8) - min(Y8))
error3_1 = sqrt((norm(y_hat3 - Y8test,2))^2/120)/(max(Y8) - min(Y8))
%.5 is clearly best!!

%Y9
[B1,FitInfo1] = lasso(X_train_logs,Y9,'Alpha',0.5,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx1Lambda1SE = FitInfo1.Index1SE;
coef1 = B1(:,idx1Lambda1SE);
coef0_1 = FitInfo1.Intercept(idx1Lambda1SE);

[B2,FitInfo2] = lasso(X_train_logs,Y9,'Alpha',0.7,'CV',3, 'NumLambda', 20, 'RelTol', 1e-3);
idx2Lambda1SE = FitInfo2.Index1SE;
coef2 = B2(:,idx1Lambda1SE);
coef0_2 = FitInfo2.Intercept(idx1Lambda1SE);

[B3,FitInfo3] = lasso(X_train_logs,Y9,'Alpha',0.9,'CV',3, 'NumLambda', 10, 'RelTol', 1e-3);
idx3Lambda1SE = FitInfo3.Index1SE;
coef3 = B3(:,idx3Lambda1SE);
coef0_3 = FitInfo3.Intercept(idx3Lambda1SE);

y_hat1 = X_test_logs*coef1 + coef0_1;
y_hat2 = X_test_logs*coef2 + coef0_2;
y_hat3 = X_test_logs*coef3 + coef0_3;

error1_1 = sqrt((norm(y_hat1 - Y9test,2))^2/120)/(max(Y9) - min(Y9))
error2_1 = sqrt((norm(y_hat2 - Y9test,2))^2/120)/(max(Y9) - min(Y9))
error3_1 = sqrt((norm(y_hat3 - Y9test,2))^2/120)/(max(Y9) - min(Y9))
%.5, .8, .9 all the same


