""" 
Applies feature selection to data using mutual information and
runs different model on the new preprocessed data with the 
best features according to mutual_information 
"""

import sys
sys.path.append('../../src/')
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import operator

from commons.utils import *

def feature_selection_with_mutual_info(estimator, parameters):
    """ 
    Builds a pipeline using SelectKBest and a regression
    estimator and returns the test MAE for the model 
    """
    
    print("Peforming feature selection using mutual information...")

    mutual_info_fs = SelectKBest(mutual_info_regression)

    print("Building the Pipeline...")
    pipeline = Pipeline([('feature_selection', mutual_info_fs),
            ('estimator', estimator)])


    print("Performing GridSearchCV on {0}...".format(estimator))
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=mae_scorer_gs, cv=3, n_jobs=-1)
    cv.fit(X_train, y_train)

    print("Best Model: ", cv.best_estimator_)
    print("Best Params: ", cv.best_params_)
    best_estimator = cv.best_estimator_
    predictions = best_estimator.predict(X_test)
    
    return absolute_error(y_test, predictions)

# print "LR: ", feature_selection_with_mutual_info(LinearRegression(), parameters = { "feature_selection__k": range(1, 49) })
# print "Kernel Ridge: ", feature_selection_with_mutual_info(KernelRidge(), parameters={"feature_selection__k": range(1, 49)})

# print "Ridge: ", feature_selection_with_mutual_info(Ridge(), parameters = {
#         "feature_selection__k": range(1, 49)
#         # "estimator__alpha": [0, .01, .1, 1, 10]
# })