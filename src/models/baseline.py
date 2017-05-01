""" 
Implementation of a baseline model for prediction of player ratings 
The baseline model just predicts the mean of the player ratings. A good baseline 
gives us a sense of what we are trying to acheive
"""

import sys
sys.path.append('../../src/')
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from commons.utils import *


class MeanBaselineModel(BaseEstimator, ClassifierMixin):  
    """ Baseline Model - Predicts the mean of the ratings """

    def __init__(self):
        self.mean_ratings = None

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

        self.mean_ratings = np.mean(y)

        return self

    def predict(self, X, y=None):
        """ This should be called after fitting the model """
        return np.asarray([self.mean_ratings for x in X])
    
    def score(self, X, y=None):
        """ 
        Scoring Method for MeanBaseline Model 
        Uses MAE (mean absolute error) to evaluate predictions
        """

        predictions = self.predict(X)
        return absolute_error(y, predictions)

# Loading in training data
train = np.load('../../data/numpy_data/train.npy') 
X = train[0:,0:-1]
y = train[:, -1]

  
print "Average MAE (MeanBaselineModel): ", model_cross_validation(estimator=MeanBaselineModel(), X=X, y=y, scoring_func=mae_scorer_cv, cv=10)
# Average MAE => 6.28727070701