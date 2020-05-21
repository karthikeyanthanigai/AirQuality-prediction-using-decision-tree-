
# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("AirQuality.csv")
#this function will provide the descriptive statistics of the dataset.(only int value)
dataset.describe()


#determine X and y variables(this values are taken as independent variables and dependent variable)
X = dataset.iloc[:9356,[2,3,4,5,6,7,8,9,10,11,12]].values
y = dataset.iloc[:9356,-3].values

#split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X, y , test_size=0.4, random_state = 0, shuffle = True )

#decision tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_features='sqrt')
regressor.fit(X_train,y_train)

#predict the test data
y_pred = regressor.predict(X_test)


#r2score
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, y_pred)



