# iris-classification-knn
A supervised learning project that classifies Iris flower species (Setosa, Versicolor, Virginica) using K-Nearest Neighbors. Covers data exploration, feature scaling, an 80/20 train-test split, model training, and evaluation via confusion matrix and F1 score.


# Iris Species Classification using KNN

A beginner-friendly supervised machine learning project that builds a classification model
to predict Iris flower species based on sepal and petal measurements.

## Overview
This project walks through the full ML pipeline: loading and exploring the classic Iris dataset,
scaling features with `StandardScaler`, splitting data into training and test sets, training a
K-Nearest Neighbors classifier, and evaluating performance using a classification report,
confusion matrix, and F1 score.

## Tech Stack
- Python
- scikit-learn
- pandas
- matplotlib / seaborn

## Dataset
The Iris dataset (150 samples, 4 features, 3 balanced classes) via `sklearn.datasets.load_iris`.

## Results
Achieves ~97% accuracy and a macro F1 score of ~0.97 on the held-out test set.
