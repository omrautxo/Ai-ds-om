
## Company Bankruptcy Prediction dataset

The project aims to construct a Random Forest classifier to predict company bankruptcy using the provided dataset. It addresses class imbalance through oversampling and assesses the model's performance using MSE and accuracy metrics
## About Project

This project perform a machine learning tasks. Here is a breakdown of the main steps:

1. Reads a CSV file named 'data.csv' containing a dataset related to company bankruptcy prediction.

2. Checks for missing values in the dataset and prints out the count of missing values for each column.

3. Splits the dataset into features (X) and the target variable (y).

4. Performs oversampling using RandomOverSampler to handle class imbalance in the target variable.

5. Prints out the original class distribution and the resampled class distribution.

6. Splits the resampled dataset into training and testing sets.

7. Initializes and trains a Random Forest classifier using the training data.

8. Makes predictions on the test set using the trained classifier.

9. Calculates the Mean Squared Error (MSE) between the actual and predicted values.

10. Calculates the accuracy of the model by comparing the predicted values to the actual values in the test set.
##  Installation
To run this project successfully, you would need to have some packages installed in your Python environment. Here's a list of the main packages used in this code:
1. pandas
2. scikit-learn
3. imbalanced-learn (imblearn)


You can install pandas using pip:

```bash
pip install pandas

```

You can install scikit-learn using pip:

```bash
pip install scikit-learn

```

You can install imbalanced-learn (imblearn) using pip:

```bash
pip install imbalanced-learn

```
## Results

This project give output like this:

1. Missing Values Summary:

```bash
Bankrupt?                                                   0
 ROA(C) before interest and depreciation before interest    0
 ROA(A) before interest and % after tax                     0
 ROA(B) before interest and depreciation after tax          0
 Operating Gross Margin                                     0
                                                           ..
 Liability to Equity                                        0
 Degree of Financial Leverage (DFL)                         0
 Interest Coverage Ratio (Interest expense to EBIT)         0
 Net Income Flag                                            0
 Equity to Liability                                        0
Length: 96, dtype: int64

```
2. Original class distribution: 

```bash
Original class distribution: Counter({0: 6599, 1: 220})

```
2. Resampled class distribution: 

```bash
Resampled class distribution: Counter({1: 6599, 0: 6599})

```


4. Mean Squared Error And Accuracy

```bash
Mean Square Error: 0.003409090909090909
Accuracy: 0.9965909090909091

```