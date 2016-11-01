
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


# breast_cancer_results = open('breast_cancer_results.txt','w')
target_url= ("http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data")
col=['subject#','age','sex ','test_time','motor_UPDRS','total_UPDRS','Jitter(%)','Jitter(Abs)','Normal Nucleoli','Mitoses','Class']

# # read data into a DataFrame
data = pd.read_csv(target_url)
print data.head()

# d['Type'] = 'benign'
# # map Type to 0 if class is 2 and 1 if class is 4
# d['Type'] = d.Class.map({2:0, 4:1})
# X = d[['Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion',
#                             'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses']]
# y=d[['Type']]
# _indent = Utilities.draw_whatever("-",6)
# #calculate correlation matrix
# corMat = DataFrame(d.iloc[:,2:10].corr())
# breast_cancer_results.writelines( _indent +'correlation matrix'+ _indent +'\n')
# breast_cancer_results.writelines(str(corMat) +'\n')
# # instantiate a logistic regression model, and fit with X and y
# model = LogisticRegression()
# model = model.fit(X, y.values.ravel())
#
# # check the accuracy on the training set
# score = model.score(X, y)
# breast_cancer_results.write("accuracy :"+ str(score) +'\n')
# breast_cancer_results.writelines(" what percentage had malignant?\n")
# breast_cancer_results.write(str(y.mean())+'\n')
# breast_cancer_results.write(Utilities.draw_whatever("-",100)+'\n')
# # examine the coefficients
# _coef= pd.DataFrame(zip(X, np.transpose(model.coef_)))
# breast_cancer_results.writelines(str(_coef)+'\n')
# breast_cancer_results.write(Utilities.draw_whatever("+-",100)+'\n')
#
# # evaluate the model using 10-fold cross-validation
# score_cv = cross_val_score(LogisticRegression(), X, y.values.ravel(), scoring='accuracy', cv=10)
# breast_cancer_results.write("accuracy :"+ str(score_cv) +'\n')
# breast_cancer_results.writelines("average accuracy : " + str(score_cv.mean())+"\n")
#
# breast_cancer_results.write(Utilities.draw_whatever("+-",100)+'\n')
#
# # evaluate the model by splitting into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# model2 = LogisticRegression()
# model2.fit(X_train, y_train)
# # predict class labels for the test set
# predicted = model2.predict(X_test)
# #confusion  matrix
# confusion_matrix = metrics.confusion_matrix(y_test, predicted)
# report = metrics.classification_report(y_test, predicted)
# breast_cancer_results.writelines(str(confusion_matrix)+"\n")
# breast_cancer_results.writelines('TP: '+ str(confusion_matrix[0][0])+"\n")
# breast_cancer_results.writelines('TN: '+ str(confusion_matrix[1][1])+"\n")
# breast_cancer_results.writelines('FN: '+ str(confusion_matrix[1][0])+"\n")
# breast_cancer_results.writelines('FP: '+ str(confusion_matrix[0][1])+"\n")
#
# breast_cancer_results.write(Utilities.draw_whatever("+-",100)+'\n')
#
# breast_cancer_results.writelines(str(report)+'\n')
#
