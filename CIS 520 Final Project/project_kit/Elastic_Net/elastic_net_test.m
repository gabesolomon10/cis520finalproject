%Test elastic net
rng(85);

%Create testing and training data
X = train_inputs;
Y = train_labels;

randpermutation = randperm(1019,1019);

X_train = X(randpermutation(1:900),:);
Y_train = Y(randpermutation(1:900),:);
X_test = X(randpermutation(901:1019),:);
Y_test = Y(randpermutation(901:1019),:);

%Make prediction
y_hat = predict_labels_elastic_net(X_train,Y_train,X_test);

%Find error
test_error = error_metric(y_hat,Y_test);
test_error
