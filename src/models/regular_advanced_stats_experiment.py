import logging
import sys
sys.path.append('../../src/')
from commons.utils import *
from baseline import MeanBaselineModel
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


def run_baseline_all_features():
    # Average MAE => 6.28727070701
    return model_cross_validation(estimator=MeanBaselineModel(), X=X_all, y=y_all, scoring_func=mae_scorer_cv, cv=10)

def run_baseline_regular_features():
    # Average MAE => 6.28727070701
    return model_cross_validation(estimator=MeanBaselineModel(), X=train_regular, y=y_all, scoring_func=mae_scorer_cv, cv=10)

def run_baseline_advanced_features():
    # Average MAE => 6.28727070701
    return model_cross_validation(estimator=MeanBaselineModel(), X=train_advanced, y=y_all, scoring_func=mae_scorer_cv, cv=10)

def run_linear_regression_all_features():
    # Average MAE 3.618
    return model_cross_validation(LinearRegression(), X=X_all, y=y_all, scoring_func=mae_scorer_cv, cv=10)

def run_linear_regression_regular_features():
    # Average MAE 3.63939107251
    return model_cross_validation(LinearRegression(), X=train_regular, y=y_all, scoring_func=mae_scorer_cv, cv=10)

def run_linear_regression_advanced_features():
    # Average MAE 3.68146163841
    return model_cross_validation(LinearRegression(), X=train_advanced, y=y_all, scoring_func=mae_scorer_cv, cv=10)

ridge_parameters = {
    'alpha': [0.01, 0.1, 1.0, 10, 100]
}

def run_ridge_regression_all_features():
    # 3.6038503309
    return get_model_mae(Ridge(), X_all, y_all, X_train, y_train, ridge_parameters)

def run_ridge_regression_regular_features():
    # 3.63253090173
    return get_model_mae(Ridge(), train_regular, y_all, X_train_regular, y_train, ridge_parameters)

def run_ridge_regression_defensive_features():
    # 3.67993687709
    return get_model_mae(Ridge(), train_advanced, y_all, X_train_advanced, y_train, ridge_parameters)

random_forest_params = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_depth': [5, 8, 15, 25, 30, None],
    'min_samples_split': [2, 5, 10, 15, 30],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['log2', 'sqrt', None],
    'n_jobs': [-1]
}

def run_forest_regression_all_features():
    # 3.61590050102
    return get_model_mae(RandomForestRegressor(), X_all, y_all, X_train, y_train, random_forest_params)

def run_forest_regression_regular_features():
    # 3.66233785138
    return get_model_mae(RandomForestRegressor(), train_regular, y_all, X_train_regular, y_train, random_forest_params)

def run_forest_regression_advanced_features():
    # 3.64111572297
    return get_model_mae(RandomForestRegressor(), train_advanced, y_all, X_train_advanced, y_train, random_forest_params)



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler("../experiment_logs/regular_advanced_all_features_experiment.log")
handler.setLevel(logging.INFO)

logger.addHandler(handler)


logger.info("Running All Features vs. Regular Features vs. Advanced Features Experiment:")

logger.info("    ==> Baseline Model")
logger.info("        Baseline Model: Average MAE with all features: {0}".format(run_baseline_all_features()))
logger.info("        Baseline Model: Average MAE with regular features: {0}".format(run_baseline_regular_features()))
logger.info("        Baseline Model: Average MAE with advanced features: {0}".format(run_baseline_advanced_features()))

logger.info("    ==> Linear Regression")
logger.info("        Linear Regression: Average MAE with all features: {0}".format(run_linear_regression_all_features()))
logger.info("        Linear Regression: Average MAE with regular features: {0}".format(run_linear_regression_regular_features()))
logger.info("        Linear Regression: Average MAE with advanced features: {0}".format(run_linear_regression_advanced_features()))

logger.info("    ==> Ridge Regression")
logger.info("        Ridge Regression: Average MAE with all features: {0}".format(run_ridge_regression_all_features()))
logger.info("        Ridge Regression: Average MAE with regular features: {0}".format(run_ridge_regression_regular_features()))
logger.info("        Ridge Regression: Average MAE with advanced features: {0}".format(run_ridge_regression_defensive_features()))

# logger.info("    ==> Random Forest Regression")
# logger.info("        Random Forest Regression: Average MAE with all features: {0}".format(run_forest_regression_all_features()))
# logger.info("        Random Forest Regression: Average MAE with regular features: {0}".format(run_forest_regression_regular_features()))
# logger.info("        Random Forest Regression: Average MAE with advanced features: {0}".format(run_forest_regression_advanced_features()))


