import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("C:/Users/OM NILESH RAUT/.ipython/DataSet/bank_transactions.csv")

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in the dataset:")
print(missing_values)

# Impute missing values for numerical columns with the mean
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

# Impute missing values for categorical columns with the mode
categorical_columns = df.select_dtypes(include=['object']).columns
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Verify if there are any missing values left
missing_values_after_imputation = df.isnull().sum()
print("Missing values in the dataset after imputation:")
print(missing_values_after_imputation)

# Filter rows where bank balance is more than 40000 and transaction amount is more than 2000
filtered_df = df[(df['CustomerDOB'] > '1980-01-01') & (df['CustomerDOB'] < '2005-01-01')  & (df['CustAccountBalance'] > 30000) & (df['TransactionAmount (INR)']>5000)]

# Calculate the total number of customers who meet the criteria
total_customers = len(filtered_df)

# Print the filtered data and total number of customers
print("People with bank balance more than 40000 and transaction amount more than 20000:")
print(filtered_df)
print("\nTotal number of customers: ", total_customers)


