""" Experimenting with Linear Regression on 2k dataset """


import sys
sys.path.append('../../src/')
import numpy as np
from sklearn.linear_model import LinearRegression
from commons.utils import *


def run_linear_offensive_defensive_features():
    print "Average MAE with all features (Linear Regression): ", model_cross_validation(LinearRegression(), X=X_all, y=y_all, scoring_func=mae_scorer_cv, cv=10)
    # Average MAE 3.618
    print "Average MAE with only offensive features (Linear Regression): ", model_cross_validation(LinearRegression(), X=train_offensive, y=y_all, scoring_func=mae_scorer_cv, cv=10)
    # Average MAE 3.63939107251
    print "Average MAE with only defensive features (Linear Regression): ", model_cross_validation(LinearRegression(), X=train_defensive, y=y_all, scoring_func=mae_scorer_cv, cv=10)
    # Average MAE 3.68146163841
