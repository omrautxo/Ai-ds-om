import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

reg=LinearRegression()

df=pd.read_csv("C:/Users/OM NILESH RAUT/.ipython/DataSet/Boston.csv")

print(df.isnull().sum())





x=df.drop('MEDV', axis=1)
y=df['MEDV']

X_train,X_test,Y_train,Y_test=train_test_split(x,y,random_state=1,test_size=0.2)

train=reg.fit(X_train,Y_train)

y_pred=reg.predict(X_test)

score=mean_squared_error(Y_test,y_pred)

print(score)