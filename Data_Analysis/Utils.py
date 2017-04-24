import numpy as np

from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score


def root_mean_squared(act_y, pred_y):
    """ Root Mean Squared Error """
    rmse = np.sqrt(mean_squared_error(act_y, pred_y))
    return rmse

def absolute_error(act_y, pred_y):
    """ Mean Absolute Error"""
    mae = mean_absolute_error(act_y, pred_y)
    return mae

# Scoring functions for GridSearch/RFE
rmse_scorer_gs = make_scorer(root_mean_squared, greater_is_better=False)
mae_scorer_gs = make_scorer(absolute_error, greater_is_better=False)

# Scoring for regular cross validation (cross_val_score)
rmse_scorer_cv = make_scorer(root_mean_squared)
mae_scorer_cv = make_scorer(absolute_error)

def model_cross_validation(estimator, X, y, scoring_func, cv):
	""" Returns the mean of all cross validation scores """
	return np.mean(cross_val_score(estimator=estimator, X=X, y=y, scoring=scoring_func, cv=cv))
	