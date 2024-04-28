
## Wine Quality

The objective is to predict wine quality using physicochemical properties like acidity, pH, alcohol content, and wine type (red or white)
## About Project

This project essentially showcases a complete machine learning pipeline for classification using Logistic Regression and evaluates a regression model using mean squared error. Here is a breakdown of the main steps:

Loads a dataset from a CSV file containing wine quality data.

Splits the dataset into features (X) and the target variable (y), excluding the "alcohol" column from the features.

Splits the data into training and testing sets using a test size of 20% and a random state of 42.

Encodes categorical variables in the features using one-hot encoding.

Handles missing values in both the training and testing sets by imputing the mean value of each column.

Scales the features using StandardScaler to standardize the dataset.

Initializes and trains a Logistic Regression model on the training data.

Makes predictions on the test set using the trained model.

Calculates the accuracy of the model by comparing the predicted values to the actual values in the test set.

Additionally, the code:

Imports NumPy and the mean_squared_error function from scikit-learn.

Generates example true values and predicted values arrays.

Computes the mean squared error between the true values and predicted values.


## Installation

To run this project successfully, you would need to have some packages installed in your Python environment. Here's a list of the main packages used in this code:
1. pandas
2. scikit-learn


You can install them using pip:

```bash
pip install pandas scikit-learn

```
## Results

This project give output like this:


1. Mean Squared Error and Accuracy

```bash
Accuracy: 0.9976923076923077
Mean Squared Error: 0.04399999999999999

```