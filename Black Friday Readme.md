
 ## Black Friday sales

The objective of this project is to develop a robust predictive model using a Black Friday sales dataset. This entails implementing comprehensive preprocessing techniques, including addressing missing values, employing label and one-hot encoding methods, transforming categorical variables into numerical representations, and ensuring appropriate feature scaling. The ultimate goal is to create a high-performing model capable of accurately forecasting Black Friday sales trends
##  About Project

This project perform a machine learning tasks using Linear Regression and evaluates a regression model using mean squared error. Here is a breakdown of the main steps:

1. Reads a CSV file named "train.csv" into a Pandas DataFrame.

2. Handles missing values by filling NaN values with 0 for specific columns.

3. Encodes categorical columns ('Product_ID', 'Gender', 'Age', 'Stay_In_Current_City_Years') into numerical values using LabelEncoder and custom mappings.

4. Applies one-hot encoding for 'City_Category' column.

5. Scales numerical columns using StandardScaler.

6. Splits the data into features (X) and the target variable (y).

7. Initializes a Linear Regression model.

8. Splits the data into training and testing sets (80% training, 20% testing).

9. Trains the Linear Regression model on the training data.

10. Makes predictions on the test data.

11. Evaluates the model using Mean Squared Error (MSE) and prints the result.

##  Installation

To run this project successfully, you have to install some packages in python environment.

1. pandas
2. scikit-learn

You can install them using pip:

```bash
pip install pandas
pip install scikit-learn

```
## Results

This project give output like this:

1. Missing Values Summary

```bash
User_ID                            0
Product_ID                         0
Gender                             0
Age                                0
Occupation                         0
City_Category                      0
Stay_In_Current_City_Years         0
Marital_Status                     0
Product_Category_1                 0
Product_Category_2            173638
Product_Category_3            383247
Purchase                           0
dtype: int64

```
2. Mean Squared Error 

```bash
Mean Squared Error: 0.8455185912138407

```
