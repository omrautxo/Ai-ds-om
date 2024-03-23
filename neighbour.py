#Logistic Regression , Navie Bayes

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#creating Instance variable
logr=LogisticRegression()
nb=GaussianNB()
kn=KNeighborsClassifier()
dt=tree.DecisionTreeClassifier()
rf=RandomForestClassifier()
gbm=GradientBoostingClassifier(n_estimators=10)

#get data
df=pd.read_csv("C:/Users/OM NILESH RAUT/.ipython/DataSet/Iris (1).csv")
# print(df)
X=df.drop('Id', axis=1)
X=X.drop('Species', axis=1)
y=df['Species']

#trainig
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#prediction
logr.fit(X_train,y_train)
y_pred=logr.predict(X_test)

nb.fit(X_train,y_train)
y_pred1=nb.predict(X_test)

kn.fit(X_train,y_train)
y_pred2=kn.predict(X_test)

dt.fit(X_train,y_train)
Y_pred3=dt.predict(X_test)

rf.fit(X_train,y_train)
Y_pred4=dt.predict(X_test)

gbm.fit(X_train,y_train)
Y_pred5=dt.predict(X_test)



#Accuracy Score
print("Logistic Regression",accuracy_score(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print(confusion_matrix(y_test,y_pred))
print("Naive Bayes:",accuracy_score(y_test,y_pred1))
print("K Neighbor:",accuracy_score(y_test,y_pred2))
print("Decision Tree:",accuracy_score(y_test,Y_pred3))
print("Random Forest :",accuracy_score(y_test,Y_pred4))
print("Gradient Classifier :",accuracy_score(y_test,Y_pred5))