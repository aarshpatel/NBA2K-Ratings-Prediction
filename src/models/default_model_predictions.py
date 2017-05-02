""" 
Experimenting with different regression models (with default parameters) and obtaining MAE CV estimates 
"""


import sys
sys.path.append('../../src/')
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from baseline import MeanBaselineModel
from sklearn.kernel_ridge import KernelRidge


from commons.utils import *
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler("../experiment_logs/default_model_predictions.log")
handler.setLevel(logging.INFO)

logger.addHandler(handler)

mean_baseline = MeanBaselineModel()
lin_reg = LinearRegression()
ridge = Ridge()
rf = RandomForestRegressor()

mean_baseline_score = model_cross_validation(mean_baseline, X_all, y_all, mae_scorer_cv, 10)
lin_reg_score = model_cross_validation(lin_reg, X_all, y_all, mae_scorer_cv, 10)
ridge_score = model_cross_validation(ridge, X_all, y_all, mae_scorer_cv, 10)
rf_score = model_cross_validation(rf, X_all, y_all, mae_scorer_cv, 10)

logger.info("Mean Baseline Model: Average MAE with all features: {0}".format(mean_baseline_score))
logger.info("Linear Regression Model: Average MAE with all features: {0}".format(lin_reg_score))
logger.info("Ridge Regression Model: Average MAE with all features: {0}".format(ridge_score))
logger.info("Random Forest Regressor Model: Average MAE with all features: {0}".format(rf_score))