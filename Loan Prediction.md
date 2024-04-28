
## Loan Prediction

The goal is to predict loan approval based on applicant features like demographics, income, and credit history. Features include gender, marital status, education, income, loan details, credit history, and loan approval status
## About Project

This code is a simple example of preprocessing data, handling missing values, encoding categorical variables, dealing with class imbalance through oversampling, and training a Random Forest classifier for a binary classification problem. Here's a breakdown of the key steps:

1. Loads a dataset from a CSV file.

2. Identifies and imputes missing values in the dataset:
i) Imputes missing values for numerical columns with the mean.

ii) Imputes missing values for categorical columns with the mode.

3. Drops the 'Loan_ID' column as it is not useful for prediction.

4. Converts categorical variables to numerical using LabelEncoder.

5. Performs oversampling using RandomOverSampler to handle class imbalance.

6. Splits the dataset into training and testing sets.

7. nitializes and trains a Random Forest classifier.

8. Predicts on the test set and calculates the accuracy of the model.
## Installation

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

Results

This project give output like this:


1. Missing values in the dataset:

```bash
Missing values in the dataset:
Loan_ID               0
Gender               13
Married               3
Dependents           15
Education             0
Self_Employed        32
ApplicantIncome       0
CoapplicantIncome     0
LoanAmount           22
Loan_Amount_Term     14
Credit_History       50
Property_Area         0
Loan_Status           0
dtype: int64

```

2. Missing values in the dataset after imputation:

```bash
Missing values in the dataset after imputation:
Loan_ID              0
Gender               0
Married              0
Dependents           0
Education            0
Self_Employed        0
ApplicantIncome      0
CoapplicantIncome    0
LoanAmount           0
Loan_Amount_Term     0
Credit_History       0
Property_Area        0
Loan_Status          0
dtype: int64

```

3. Original class distribution:

```bash
Original class distribution: Counter({1: 422, 0: 192})

```

4. Resampled class distribution:

```bash
Resampled class distribution: Counter({1: 422, 0: 422})

```

5. Accuracy:

```bash
Accuracy: 0.9112426035502958

```