""" Class that optimizes the model using feature selection and hyperparameter optimization """


import sys
sys.path.append('../../src/')
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from feature_selection.mutual_info_feature_selection import *
import utils


class OptimizeModel():
    """ 
    Builds a pipeline by using feature selection and hyperparameter optimization 
    of an estimator

    Feature Selection: SelectKBest using Mutual Information
    Hyperparameter Optimization: Cross Validation GridSearchCV
    GridSearchCV find the best features using SelectKBest while also 
    finding the best hyperparameters
    """

    def __init__(self, model, param_grid):
        self.model = model
        self.param_grid = param_grid

    def feature_selection_and_hyperparameter_optimization(self, X_train, y_train, X_test, y_test):

        mutual_info_fs = SelectKBest(mutual_info_regression)

        print("Building the Pipeline...")
        pipeline = Pipeline([('feature_selection', mutual_info_fs),
                ('estimator', self.model)])


        print("Performing GridSearchCV on {0}...".format(self.model))
        self.cv = GridSearchCV(pipeline, param_grid=self.param_grid, scoring=mae_scorer_gs, cv=5, verbose=10)
        self.cv.fit(X_train, y_train)

        self.best_estimator = self.cv.best_estimator_ # obtain the best estimator after grid search
        prediction = self.best_estimator.predict(X_test)        
        self.mae_score = utils.absolute_error(y_test, prediction)
        return self.best_estimator

