
%Get Psuedo-test set
predictions = predict_labels(X(1:900, :), train_labels(1:900, :), X(901:1019, :));
error_metric(predictions, train_labels(901:1019, :))
