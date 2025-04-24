# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SABARINATH R
RegisterNumber:  212223100048
*/
import pandas as pd
from sklearn.metrics import r2_score
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
r2= r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
## Dataset
![image](https://github.com/user-attachments/assets/abef65d4-b219-46a9-b935-4f67bf7ed341)


## Info
![image](https://github.com/user-attachments/assets/cff8eb11-6e81-4ba7-a3ed-124f1a2a5f11)


## NULL value
![image](https://github.com/user-attachments/assets/6ee43067-c547-4b83-b5ad-c561cb99c825)


## Encoded
![image](https://github.com/user-attachments/assets/282dce0d-ffba-4744-ba89-14e28a76e438)


## x and y value
![image](https://github.com/user-attachments/assets/cc20c1ca-9cf6-4cf4-9dc4-c061694ecb8b)

![image](https://github.com/user-attachments/assets/f27fe43d-9b06-43c9-ba54-04d0b106d66f)


## Algorithm
![image](https://github.com/user-attachments/assets/e9497877-3758-43e3-b387-a0b5ebd0bab9)

## R2 Score
![image](https://github.com/user-attachments/assets/63f58e6e-bd00-46b3-9fc0-c20b898ee3c3)

## Predicted
![image](https://github.com/user-attachments/assets/a9774313-d3f9-4734-bf2b-fe88f1ec528d)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
