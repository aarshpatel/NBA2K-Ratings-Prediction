""" Experimenting with Linear Regression on 2k dataset """


import sys
sys.path.append('../../src/')
import numpy as np
from sklearn.linear_model import LinearRegression
from commons.utils import *
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.FileHandler("../experiment_logs/lin_reg_experiments.log")
handler.setLevel(logging.INFO)

logger.addHandler(handler)

logger.info("Linear Regression: Average MAE with all features: {0}".format(model_cross_validation(LinearRegression(), X=X_all, y=y_all, scoring_func=mae_scorer_cv, cv=10)))