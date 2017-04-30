""" Experimenting with Ridge Regression on 2k dataset """

import numpy as np
from sklearn.linear_model import Ridge
from utils import *


def get_ridge_mae(X, y, X_data, y_data, param_grid):
    ridge = Ridge()
    optimal_ridge = get_best_estimator(ridge, 1, 10, mae_scorer_gs, param_grid, X_data, y_data)
    return model_cross_validation(optimal_ridge, X, y, mae_scorer_cv, 10)


ridge_parameters = {
    'alpha': [0.01, 0.1, 1.0, 10, 100],
    'normalize': [True, False],
    'fit_intercept': [True, False]
}


print "Average MAE with all Features (Ridge Regression): ", get_ridge_mae(X, y, X_train, y_train, ridge_parameters)

print "Average MAE with only offensive features (Ridge Regression): ", get_ridge_mae(train_offensive, y, X_train_offensive, y_train, ridge_parameters)

print "Average MAE with only defensive features (Ridge Regression): ", get_ridge_mae(train_defensive, y, X_train_defensive, y_train, ridge_parameters)





