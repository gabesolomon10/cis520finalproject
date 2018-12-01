%Test elastic net

X = train_inputs;
Y = train_labels;

randpermutation = randperm(1019,1019);

X_train = X(randpermutation(1:900),:);
Y_train = Y(randpermutation(1:900),:);
X_test = X(randpermutation(900:1019),:);
Y_test = Y(randpermutation(900:1019),:);

y_hat = elastic_net(X_train,Y_train,X_test);
