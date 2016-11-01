
# imports
from utilities import Utilities
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression


parkinson_disease_results = open('parkinson_disease_results.txt','w')
target_url= ("http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data")

# # read data into a DataFrame
data = pd.read_csv(target_url)
motor_UPDRS = data[['motor_UPDRS']]
total_UPDRS = data [['total_UPDRS']]
X = data[['age','sex','test_time','Jitter(%)','Jitter(Abs)','Jitter(Abs)','Jitter:RAP','Jitter:PPQ5','Jitter:DDP','Shimmer',
     'Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','Shimmer:APQ11','Shimmer:DDA','NHR','HNR','RPDE','DFA','PPE']]
_indent = Utilities.draw_whatever("-",6)
#calculate correlation matrix
corMat = DataFrame(data.iloc[:,2:].corr())
parkinson_disease_results.writelines( _indent +'correlation matrix'+ _indent +'\n')
parkinson_disease_results.writelines(str(corMat) +'\n')
# instantiate a logistic regression model, and fit with X and y
model_1 = LinearRegression()
model_1 = model_1.fit(X, motor_UPDRS.values.ravel())

parkinson_disease_results.writelines(_indent + "motor_UPDRS \n" + _indent)
# check the accuracy on the training set
score_1 = model_1.score(X, motor_UPDRS.values.ravel())
parkinson_disease_results.write("accuracy :"+ str(score_1) +'\n')

# print intercept and coefficients
parkinson_disease_results.write("intercept_ :"+ str(model_1.intercept_) +'\n')

parkinson_disease_results.write(Utilities.draw_whatever("-",100)+'\n')
# examine the coefficients
# pair the feature names with the coefficients
parkinson_disease_results.write('pair the feature names with the coefficients'+'\n')
_coef= pd.DataFrame(zip(X, model_1.coef_))
parkinson_disease_results.writelines(str(_coef)+'\n')
parkinson_disease_results.write(Utilities.draw_whatever("+-*",200)+'\n')


# instantiate a logistic regression model, and fit with X and y
model_2 = LinearRegression()
model_2 = model_2.fit(X, total_UPDRS.values.ravel())

parkinson_disease_results.writelines(_indent + "total_UPDRS \n" + _indent)
# check the accuracy on the training set
score_2 = model_2.score(X, total_UPDRS.values.ravel())
parkinson_disease_results.write("accuracy :"+ str(score_2) +'\n')

# print intercept and coefficients
parkinson_disease_results.write("intercept_ :"+ str(model_2.intercept_) +'\n')

parkinson_disease_results.write(Utilities.draw_whatever("-",100)+'\n')
# examine the coefficients
# pair the feature names with the coefficients
parkinson_disease_results.write('pair the feature names with the coefficients'+'\n')
_coef= pd.DataFrame(zip(X, model_2.coef_))
parkinson_disease_results.writelines(str(_coef)+'\n')
parkinson_disease_results.write(Utilities.draw_whatever("+-*",200)+'\n')
