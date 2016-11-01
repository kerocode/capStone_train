
# imports
from utilities import Utilities
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression

wine_white_results = open('wine_white_results.txt','w')
wine_red_results = open('wine_red_results.txt','w')
# read data into a DataFrame
data = pd.read_csv("winequality-white.csv",delimiter=';')
data2 = pd.read_csv("winequality-red.csv",delimiter=';')


X = data[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
y=data[["quality"]]
_indent = Utilities.draw_whatever("-",6)

#calculate correlation matrix
corMat = DataFrame(data.iloc[:,:11].corr())
wine_white_results.writelines( _indent +'correlation matrix'+ _indent +'\n')
wine_white_results.writelines(str(corMat) +'\n')
# instantiate a logistic regression model, and fit with X and y
model = LinearRegression()
model = model.fit(X, y.values.ravel())

# check the accuracy on the training set
score = model.score(X, y)
wine_white_results.write("accuracy :"+ str(score) +'\n')
# print intercept and coefficients
wine_white_results.write("intercept_ :"+ str(model.intercept_) +'\n')

wine_white_results.write(Utilities.draw_whatever("-",100)+'\n')
# examine the coefficients
# pair the feature names with the coefficients
wine_white_results.write('pair the feature names with the coefficients'+'\n')
_coef= pd.DataFrame(zip(X, model.coef_))
wine_white_results.writelines(str(_coef)+'\n')
wine_white_results.write(Utilities.draw_whatever("+-",100)+'\n')


#red

X2 = data2[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]]
y2=data2[["quality"]]
#calculate correlation matrix
corMat2 = DataFrame(data2.iloc[:,:11].corr())
wine_red_results.writelines( _indent +'correlation matrix'+ _indent +'\n')
wine_red_results.writelines(str(corMat2) +'\n')
# instantiate a logistic regression model, and fit with X and y
model2 = LinearRegression()
model2 = model2.fit(X2, y2.values.ravel())

# check the accuracy on the training set
score2 = model2.score(X2, y2)
wine_red_results.write("accuracy :"+ str(score2) +'\n')
# print intercept and coefficients
wine_red_results.write("intercept_ :"+ str(model2.intercept_) +'\n')

wine_red_results.write(Utilities.draw_whatever("-",100)+'\n')
# examine the coefficients
# pair the feature names with the coefficients
wine_red_results.write('pair the feature names with the coefficients'+'\n')
_coef2= pd.DataFrame(zip(X2, model2.coef_))
wine_red_results.writelines(str(_coef2)+'\n')
wine_red_results.write(Utilities.draw_whatever("+-",100)+'\n')



