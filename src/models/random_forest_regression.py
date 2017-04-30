""" Experimenting with Ridge Regression on 2k dataset """

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from utils import *

param_random_forest = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [5, 8, 15, 25, 30, None],
    'min_samples_split': [2, 5, 10, 15, 50],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['log2', 'sqrt', None]
}

random_forest = RandomForestRegressor()
best_forest = get_best_estimator(random_forest, 1, 10, mae_scorer_gs, param_random_forest, X_train, y_train)

print "Average MAE (Random Forest Regression): ", model_cross_validation(best_forest, X, y, mae_scorer_cv, 10)



