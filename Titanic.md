
## Titanic Dataset
The aim is to predict passenger survival on the Titanic using machine learning algorithms, considering attributes like age, sex, and ticket class
## About Project

This project perform a machine learning tasks that uses a RandomForestClassifier to predict the 'Embarked' column. Here is a breakdown of the main steps:

1. Data Loading and Preprocessing:

i) The Titanic dataset is loaded from a CSV file.

ii) Label encoding is applied to the 'Sex' and 'Ticket' columns.

iii) Missing values in the 'Age', 'Embarked', and 'Cabin' columns are handled by filling them with the mean or 'nun' (a placeholder).

2. Outlier Handling:

i) There is a function handle_outliers defined to handle outliers in the 'Age', 'SibSp', 'Parch', and 'Fare' columns using the IQR method.

3. Feature Selection:

i) Feature selection is mentioned as a step that is not done in the code. This is a crucial step in machine learning where you choose the most relevant features for your model.

4. Model Training and Evaluation:

i) The data is split into training and testing sets.

ii) A RandomForestClassifier is trained on the training data.

iii) The model makes predictions on the test set.

iv) Mean Squared Error (MSE) and Accuracy score are calculated to evaluate the model's performance.
## Installation

To run this project successfully, you would need to have several packages installed in your Python environment. Here's a list of the main packages used in this code:
1. pandas
2. matplotlib
3. seaborn
4. scikit-learn


You can install pandas using pip:

```bash
pip install pandas
```

You can install matplotlib using pip: 
```bash
pip install matplotlib
```

You can install seaborn using pip: 

```bash
pip install seaborn
```

You can install scikit-learn using pip:

```bash
pip install scikit-learn
```
## Results

This project give output like this:

1. Missing Values Summary:

```bash
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64

``` 
2. Handle Missing Values:

```bash
PassengerId    0
Survived       0
Pclass         0
Name           0
Sex            0
Age            0
SibSp          0
Parch          0
Ticket         0
Fare           0
Cabin          0
Embarked       0
dtype: int64

```

3. Mean Squared Error and Accuracy of Random Tree

```bash
Mean Squared Error: 0.3240223463687151
Accuracy of Random Tree: 0.8938547486033519

```