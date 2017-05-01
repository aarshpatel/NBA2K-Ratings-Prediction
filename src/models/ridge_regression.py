""" Experimenting with Ridge Regression on 2k dataset """

import numpy as np
from sklearn.linear_model import Ridge
from utils import *

ridge_parameters = {
    'alpha': [0.01, 0.1, 1.0, 10, 100],
    'normalize': [True, False],
    'fit_intercept': [True, False]
}


all_features_mae = get_model_mae(Ridge(), X_all, y_all, X_train, y_train, ridge_parameters)
offensive_features_mae = get_model_mae(Ridge(), train_offensive, y_all, X_train_offensive, y_train, ridge_parameters)
defensive_features_mae = get_model_mae(Ridge(), train_defensive, y_all, X_train_defensive, y_train, ridge_parameters)


print "Average MAE with all Features (Ridge Regression): ", all_features_mae
# 3.6038503309
print "Average MAE with only offensive features (Ridge Regression): ", offensive_features_mae
# 3.63253090173
print "Average MAE with only defensive features (Ridge Regression): ", defensive_features_mae
# 3.67993687709





