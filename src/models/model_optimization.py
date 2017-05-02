import sys
sys.path.append('../../src/')

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from commons.utils import *
from commons.OptimizeModel import OptimizeModel
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler("../experiment_logs/optimized_models.log")
handler.setLevel(logging.INFO)

logger.addHandler(handler)

print("Optimizing Linear Regression Model")
lin_reg_model = OptimizeModel(LinearRegression(), param_grid={
    "feature_selection__k": range(45, 49)
})

lin_reg_optimized = lin_reg_model.feature_selection_and_hyperparameter_optimization(X_train, y_train, X_test, y_test)

lin_reg_test_mae = absolute_error(test_all_y, lin_reg_optimized.predict(test_all_X))
print("TEST MAE Linear Regression: {0}".format(lin_reg_test_mae))
# logger.info("MAE Lin Reg Optimized: {0}".format(lin_reg_model.mae_score))

print("Optimizing Ridge Regression")
ridge_model = OptimizeModel(Ridge(), param_grid={
    "feature_selection__k": range(45, 49),
    # "estimator__alpha": np.arange(0, 1, .1)
    "estimator__alpha": [0, .01, .001, .1, 1, 10]
})

ridge_model_optimzed = ridge_model.feature_selection_and_hyperparameter_optimization(X_train, y_train, X_test, y_test)
ridge_test_mae = absolute_error(test_all_y, ridge_model_optimzed.predict(test_all_X))
print("TEST MAE Ridge: {0}".format(ridge_test_mae))

# logger.info("MAE Ridge Optimized: {0}".format(ridge_model.mae_score))

print("Optimizing Random Forest Regressor")
rf_model = OptimizeModel(RandomForestRegressor(), param_grid={
    "feature_selection__k": range(45, 49),
    "estimator__n_estimators": [10, 50, 100, 500]
})

rf_model_optimized = rf_model.feature_selection_and_hyperparameter_optimization(X_train, y_train, X_test, y_test)

rf_test_mae = absolute_error(test_all_y, rf_model_optimized.predict(test_all_X))
print("TEST MAE RF: {0}".format(rf_test_mae))

# logger.info("RF Optimized: {0}".format(rf_model.mae_score))