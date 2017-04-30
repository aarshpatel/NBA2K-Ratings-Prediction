""" Experimenting with Linear Regression on 2k dataset """


import numpy as np
from sklearn.linear_model import LinearRegression
from utils import *

baseline_linreg = LinearRegression()

print "Average MAE (Linear Regression): ", model_cross_validation(LinearRegression(), X=X, y=y, scoring_func=mae_scorer_cv, cv=10)
# Average MAE 3.618