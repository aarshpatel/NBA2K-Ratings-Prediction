""" Experimenting with Linear Regression on 2k dataset """


import numpy as np
from sklearn.linear_model import LinearRegression
from utils import *

baseline_linreg = LinearRegression()

print "Average MAE with all features (Linear Regression): ", model_cross_validation(LinearRegression(), X=X_all, y=y_all, scoring_func=mae_scorer_cv, cv=10)
# Average MAE 3.618
print "Average MAE with only offensive features (Linear Regression): ", 