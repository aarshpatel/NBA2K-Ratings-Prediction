import numpy as np

from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error


def root_mean_squared(act_y, pred_y):
    """ Root Mean Squared Error """
    rmse = np.sqrt(mean_squared_error(act_y, pred_y))
    return rmse


def absolute_error(act_y, pred_y):
    """ Mean Absolute Error"""
    mae = mean_absolute_error(act_y, pred_y)
    return mae


def make_rmse_scorer():
    return make_scorer(root_mean_squared, greater_is_better=False)


def make_mae_scorer():
    return make_scorer(absolute_error, greater_is_better=False)