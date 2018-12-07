%Test RBF

X = train_inputs;
Y = train_labels; 

randpermutation = randperm(1019,1019);

X_train = X(randpermutation(1:100),:);
Y_train = Y(randpermutation(1:100),:);
X_test = X(randpermutation(101:1019),:);
Y_test = Y(randpermutation(101:1019),:);

Y_hat = predicct_labels_rbf(X_train,Y_train,X_test);

error = error_metric(Y_hat,Y_test);