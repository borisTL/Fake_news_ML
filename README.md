# Fake News Detection Project

## Overview

This project aims to detect fake news using machine learning techniques. We utilize a dataset containing news articles labeled as either "FAKE" or "REAL" and build a logistic regression model to classify them.

## Steps Applied

### 1. Importing Libraries:

We import the necessary libraries for data manipulation, model building, and evaluation:
- `pandas` - for working with data in tabular format
- `train_test_split` from `sklearn.model_selection` - for splitting data into training and testing sets
- `TfidfVectorizer` from `sklearn.feature_extraction.text` - for converting text data into numerical features using TF-IDF
- `LogisticRegression` from `sklearn.linear_model` - for building logistic regression model
- `accuracy_score` from `sklearn.metrics` - for calculating model accuracy

### 2. Loading Data:

We load the dataset from a CSV file using `pd.read_csv()` and drop rows with missing values using the `dropna()` method.

### 3. Data Preparation:

We extract the text data column and the label column from the DataFrame. The labels are transformed such that the value 'FAKE' becomes 1, and all other labels become 0.

### 4. Splitting into Training and Testing Data:

We split the data into training and testing sets in a 80/20 ratio using the `train_test_split()` function.

### 5. Feature Extraction:

We create a `TfidfVectorizer` object to convert text data into numerical features using TF-IDF. Then we fit the vectorizer to the training data and transform both training and testing data.

### 6. Model Training:

We create a logistic regression model using `LogisticRegression()` and train it on the training data.

### 7. Prediction and Evaluation:

We predict labels for the test data using the trained model and calculate the accuracy of predictions using `accuracy_score()`. The model achieved an accuracy of approximately 91.24%.

## Next Steps

- Experiment with different machine learning algorithms to improve model performance.
- Explore additional text preprocessing techniques to enhance feature extraction.
- Conduct further analysis to understand misclassifications and improve model interpretability.

## Dependencies

- pandas
- scikit-learn

## Dataset

The dataset used in this project contains news articles labeled as either "FAKE" or "REAL". It is available in CSV format. From Kaggle.

