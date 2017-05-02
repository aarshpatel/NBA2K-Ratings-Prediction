""" Experimenting with Ridge Regression on 2k dataset """

import sys
sys.path.append('../../src/')
import numpy as np
from sklearn.linear_model import Ridge
from commons.utils import *

ridge_parameters = {
    'alpha': [0.01, 0.1, 1.0, 10, 100]
}

def run_ridge_regression_all_features():
    # 3.6038503309
    return get_model_mae(Ridge(), X_all, y_all, X_train, y_train, ridge_parameters)

def run_ridge_regression_offensive_features():
    # 3.63253090173
    return get_model_mae(Ridge(), train_offensive, y_all, X_train_offensive, y_train, ridge_parameters)

def run_ridge_regression_defensive_features():
    # 3.67993687709
    return get_model_mae(Ridge(), train_defensive, y_all, X_train_defensive, y_train, ridge_parameters)





