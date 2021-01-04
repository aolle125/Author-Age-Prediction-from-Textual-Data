# Importing the libraries

import pandas as pd
import numpy as np
import time
import string
import re
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
import utils
import ridge_closed_form,ridge_regression,lasso_regression,linear_regression

# Expermenting with various regression techniques


# 1) Trying Bayesian Linear Regression with L2 regularization

#Finding the best Hyperparameters for BayesianRidge using GridSearch and 5-fold cross validation

#parameters = {'alpha_1':[0.01,0.0001,0.00001,0.0000001], 'alpha_2':[0.01,0.0001,0.00001,0.0000001], 'lambda_1':[0.01,0.0001,0.00001,0.0000001], 'lambda_2':[0.01,0.0001,0.00001,0.0000001]}
#model = BayesianRidge(n_iter=100000)
#clf = GridSearchCV(model, parameters, cv=5)
#clf.fit(X_train, y_train)
#clf.best_params_

#The best parameters came out to be the default parameters

def BayesianRegression(X_train, y_train, X_test, y_test):
    model = BayesianRidge(n_iter=100000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


#2) Trying Ensemble Regression methods provided by the sklearn library
# For a regression problem, the outputs of individual models are averaged to obtain the output of the ensemble model


#Finding the best Hyperparameters for Ensemble Regression methods using GridSearch and 5-fold cross validation
#parameters = {'n_estimators':[50,100,200,300]}
#models = [GradientBoostingRegressor(n_estimators = 100), RandomForestRegressor(n_estimators = 100), ExtraTreesRegressor(n_estimators = 200), AdaBoostRegressor(n_estimators = 50)]
#for model in models:
    #clf = GridSearchCV(model, parameters, cv=5)
    #clf.fit(X_train, y_train)
    #clf.best_params_


def Ensemble(X_train, y_train, X_test, y_test):
    models = [GradientBoostingRegressor(n_estimators = 100), RandomForestRegressor(n_estimators = 100), ExtraTreesRegressor(n_estimators = 100), AdaBoostRegressor(n_estimators = 100)]
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print("The MAE for",model.__class__.__name__, "Regression is:",utils.calc_mae(y_pred,y_test))
        print("The Pearson's Correlation Coefficient (r) for",model.__class__.__name__, "Regression is: ", utils.r(y_pred,y_test))
        print()


# 3) Trying the Support Vector Regressor with the 'RBF' kernel and C value as 10, found using Grid Search and 5-fold cross validation

#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100]}
#svc = svm.SVR()
#clf = GridSearchCV(svc, parameters, cv=5)
#clf.fit(X_train, y_train)


def SupportVectorRegression(X_train, y_train, X_test, y_test):
    model = SVR(kernel = 'rbf', C=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred









