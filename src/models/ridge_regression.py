""" Experimenting with Ridge Regression on 2k dataset """

import numpy as np
from sklearn.linear_model import Ridge
from utils import *


ridge_parameters = {
    'alpha': [0.01, 0.1, 1.0, 10, 100],
    'normalize': [True, False],
    'fit_intercept': [True, False]
}


ridge = Ridge()
best_ridge = get_best_estimator(ridge, 1, 10, mae_scorer_gs, ridge_parameters, X_train, y_train)

print "Average MAE (Ridge Regression): ", model_cross_validation(best_ridge, X, y, mae_scorer_cv, 10)


