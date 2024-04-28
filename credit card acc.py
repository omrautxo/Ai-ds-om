import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv("C:/Users/OM NILESH RAUT/.ipython/DataSet/creditcard.csv")

# Assuming the last column is the label and rest are features
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize a logistic regression model
model = LogisticRegression(max_iter=1000)  # Increase max_iter to ensure convergence

# Train the model on the scaled training data
model.fit(X_train_scaled, y_train)

# Predict labels for the scaled test set
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
