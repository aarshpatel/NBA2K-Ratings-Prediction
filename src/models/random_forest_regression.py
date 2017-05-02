""" Experimenting with Ridge Regression on 2k dataset """

import sys
sys.path.append('../../src/')
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from commons.utils import *

random_forest_params = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_depth': [5, 8, 15, 25, 30, None],
    'min_samples_split': [2, 5, 10, 15, 30],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['log2', 'sqrt', None],
    'n_jobs': [-1]
}

#all_features_mae = get_model_mae(RandomForestRegressor(), X_all, y_all, X_train, y_train, random_forest_params)
#offensive_features_mae = get_model_mae(RandomForestRegressor(), train_offensive, y_all, X_train_offensive, y_train, random_forest_params)
#defensive_features_mae = get_model_mae(RandomForestRegressor(), train_defensive, y_all, X_train_defensive, y_train, random_forest_params)

