K-Nearest Neighbors (k-NN) Classifier

Overview:

This repository contains an implementation of the k-Nearest Neighbors (k-NN) algorithm for classification using the MNIST dataset. The k-NN algorithm is a simple and effective method for classifying data points based on the majority class among their k-nearest neighbors.

Handwritten Digit Recognition:
The primary goal of this project is to demonstrate the effectiveness of the k-NN algorithm in recognizing handwritten digits. The MNIST dataset, consisting of 28x28 pixel grayscale images of handwritten digits, is used for training and testing the classifier.

Dataset:

The MNIST dataset is used for training and testing the k-NN classifier. The dataset consists of handwritten digits, and each data point is a 28x28 pixel grayscale image.

MNIST Training Dataset

MNIST Test Dataset

Dependencies:

Python 3.x

NumPy

wget (for downloading datasets)

Usage:

Download the MNIST datasets using the provided links.

Run the Jupyter notebook or Python script to train and test the k-NN classifier.

Adjust hyperparameters such as k (number of neighbors) and n (norm for distance calculation) for experimentation.

Code Structure:

knn_classifier.ipynb: Google colab notebook containing the k-NN classifier implementation.

mnist_train.csv: Training dataset file.

mnist_test.csv: Test dataset file.

Functions:

shuffle(X, Y): Shuffles input features (X) and labels (Y) randomly.

Ln_norm_distances(train_X, test_x, n): Calculates the L-norm distances between training and test data points.

majority_based_knn(distances_matrix, train_Y, k): Implements the k-NN algorithm for classification.

calculate_accuracy(predicted_labels, actual_labels): Calculates the accuracy of the model.

Results:

The accuracy of the k-NN classifier is evaluated on both the validation and test datasets for different values of k and n.
