
## Walmart Recruiting Store sales

The goal is to forecast weekly sales for different departments in Walmart stores using historical sales data and additional features like dates and holiday indicators to enhance inventory management and revenue projections
## About Project

This Project performs the following steps:

1. Imports necessary libraries: pandas, seaborn, numpy, matplotlib, LinearRegression, train_test_split, and mean_squared_error from sklearn.

2. Defines two arrays: true_values and predicted_values.

3. Computes the mean squared error between the true_values and predicted_values arrays.

4. Reads a CSV file named "train.csv".

5. Converts the "Date" column of df to datetime format.

6. One-hot encodes the "Date" column of df.

7. Prepares the data for linear regression by dropping the "Weekly_Sales" column from encoded_df and assigning the remaining columns to X and the "Weekly_Sales" column to y.

8. Splits the data into training and testing sets using train_test_split.

9. Trains a linear regression model using the training data.

10. Makes predictions using the trained model on the testing data.

11. Calculates the mean squared error between the actual testing values (y_test) and the predicted values (y_pred).

12. Prints the mean squared error.
## Installation


To run this project successfully, you would need to have several packages installed in your Python environment. Here's a list of the main packages used in this code: 
1. pandas 
2. seaborn
3. numpy
4. matplotlib.pyplot
5. scikit-learn (sklearn)

You can install them using pip:

```bash
pip install pandas seaborn numpy matplotlib scikit-learn

```
## Results

This project give output like this:


1. Mean Squared Error:

```bash
Mean Squared Error: 0.04399999999999999

```