Note: All of the prediction functions for our models are named with the convention “predict_labels_[model_name]()”


————————————————————————————————————
Generative Method: PCA + Random Forest
————————————————————————————————————
Description: Our generative method we used here was PCA. Specifically, we used PCA on the tweet data (columns 22-2021) and selected the 35 largest principal components.  We then used unaltered demographic features plus these 35 principal components from the tweets to build a random forest for each target outcome, Y. Contents of all implementations and training can be found in the in “PCA_and_Random_Forest” folder. Details on each file below:

- predict_labels_random_forest.m
	This file contains the main method that takes in training data and test inputs and generates predictions for the test set using a Random Forest of PCA-reduced data as described above

- PCA_CCA_Initial.m
	This file contains the exploratory work done to determine the best number of principal components to include for each of 3 separate PCA’s: one on all variables, one on tweets only, and one on demographic data as well as the tweet-reduced principal components. This file generates plots

- Random_Forest_CV.m
	This file contains the validation of which combination of variables for PCA yields the best out of sample error results (using the provided error metric)

- error_metric.m
	The provided error metric


Additional Training Details: 
	We selected 35 principal components from the tweets because, after cross-validation, we found that using the 35 largest principal components explained 90% of the variance. Note that other PCA combinations were tried, but examining test error results of different random forest models showed PCA on the tweets alone to be the most effective.
	
————————————————————————————————————
Discriminative Method: Elastic Net
————————————————————————————————————
Folder: Elastic_Net

Description: The discriminative method we used was an elastic net.  We took logs of the tweet data (columns 22-end), because probabilites are better understood in log-space.  We then fit nine separate elastic net regressions, one for each of the outcome variables, each with an alpha value of .75.  After constructing nine separate elastic nets, we combined the predictions together to create a final matrix of nine predictions, one for each outcome variable. 

-predict_labels_elastic_net.m: This file contains the elastic net function used for the discriminiative method, as well as the leaderboard submission.  This function takes in the training data, takes the logs of all the LDA probabilities for each topic, and then fits nine separate elastic net regressions, one for each of the nine outcome variables.  The nine outcome vectors and then concatenated together to create a final prediction matrix for each input.

-elastic_net_hyperparameter_search.m: This file contains a test across different alpha values in the elastic net.  For each of the nine elastic net regressions in predict_labels_elastic_net, the alpha value is tested across the range [,5,.7,.9] and cross-validated to find the best alpha value for each of the nine elastic net regressions.

-elastic_net2.m: This file is very similar to predict_labels_elastic_net, but with the alpha hyperparameter for each elastic net set to .5, not .75.  This was done as a testing procedure.

-elastic_net3.m: This file is very similar to predict_labels_elastic_net, but with a few small transformations on the socioeconomic data. No noticable improvements were recorded, so this file was not really used much.

-elastic_net_test.m: This function divides the data given into pseudo test and training sets in order to test the accuracy of the predict_labels_elastic_net function.



————————————————————————————————————
Instance-Based Method: K-Means + RBF
————————————————————————————————————
Description: We built and ran a Radial Basis Function model that uses K-means clustering as the basis for selecting K cluster centers around which we computed Gaussian Kernel regression. After cross validation, a good choice for K was found to be 80 clusters. The kernels from which the regression was run were Gaussian kernels between each observation and the 80 cluster centers. Kernel widths were set to be the average distance between nearest neighbors in the training set. Contents of all implementations and training can be found in the in “Kmeans_and_RBF” folder. Details on each file below:


- predict_labels_rbf.m
	This file contains the main method (“predict_labels_rbf()”) that takes in training data and test inputs and generates predictions for the test set using an RBF model as described above. It uses the other methods in the folder to generate these predictions

- K_means_CV_for_K.m
	This file contains the cross validation for selecting K in our K-means clustering used to select the kernel centers

- get_K_clusters.m
	Computes K means clustering for 80 clusters using the training data. First standardizes the data, then returns the indices of the points in each cluster, as well as the coordinates of the cluster centers.

- get_kernel_matrices.m
	Given a dataset, X, and a set of cluster centers, returns the kernel matrix between X and the cluster centers. Specifically, computes Gaussian kernels with a width of the average nearest neighbor distance between points of X

- nearest_neighbor_distance.m
	Computes the average nearest neighbor distance between points in a dataset X

- rbf.m
	Returns the slope of the radial basis function regression given a kernel matrix Z and a target matrix Y

- predict_y_rbf.m
	Given test data, an RBF model (i.e. slope), and cluster centers, uses the RBF model to predict test labels. Specifically, computes the kernel matrix between the test data and the cluster centers, then runs the RBF on that kernel matrix.

- rbf_test.m
	Provides test code for randomly dividing the provided data into testing and training sets, then making predictions on the generated test set using RBFs


————————————————————————————————————
Something novel: A pseudo multitask learning elastic net
————————————————————————————————————
Description: 
