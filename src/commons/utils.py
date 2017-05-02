""" 
Implementation of the Utils file. This script contains all of the code that will be
used in multiple scripts
"""
import numpy as np

from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from scipy import stats

from OptimizeModel import OptimizeModel


# Loads all of the training data
train = np.load('../../data/numpy_data/train.npy')
# Split the training data into just offensive and defensive data
train_offensive = np.load('../../data/numpy_data/train_offensive.npy')
train_defensive = np.load('../../data/numpy_data/train_defensive.npy')
# Split the training data into just regular and advanced data
train_regular = np.load('../../data/numpy_data/train_regular.npy')
train_advanced = np.load('../../data/numpy_data/train_advanced.npy')

# All features of the training data
X_all = train[0:,0:-1]
y_all = train[:, -1]

# Load the X training data with all features
X_train = np.load('../../data/numpy_data/X_train.npy')
# Load the X training data with offensive and defensive data splits
X_train_offensive = np.load('../../data/numpy_data/X_train_offensive.npy')
X_train_defensive = np.load('../../data/numpy_data/X_train_defensive.npy')
# Load the X training data with regular and advanced data splits
X_train_regular = np.load('../../data/numpy_data/X_train_regular.npy')
X_train_advanced = np.load('../../data/numpy_data/X_train_advanced.npy')
# Loads the y training data
y_train = np.load('../../data/numpy_data/y_train.npy')

# Load the X testing data with all features
X_test = np.load('../../data/numpy_data/X_test.npy')
# Load the X testing data with offensive and defensive data splits
X_test_offensive = np.load('../../data/numpy_data/X_test_offensive.npy')
X_test_defensive = np.load('../../data/numpy_data/X_test_defensive.npy')
# Load the X testing data with regular and advanced data splits
X_test_regular = np.load('../../data/numpy_data/X_test_regular.npy')
X_test_advanced = np.load('../../data/numpy_data/X_test_advanced.npy')
# Load the y testing data
y_test = np.load('../../data/numpy_data/y_test.npy')


# Load in the heldout test data
test = np.load('../../data/numpy_data/test.npy')


# All features of the training data
test_all_X = test[0:,0:-1]
test_all_y = test[:, -1]

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


def get_model_mae(estimator, X, y, X_train, y_train, X_test, y_test, param_grid):
    """ 
    Performs hyperparmeter optimization and then evaluates the best model 
    using K-Fold Cross Validation to get a MAE estimate.
    """
    optimal = OptimizeModel(model=estimator, param_grid=param_grid)

    optimal.feature_selection_and_hyperparameter_optimization(X_train, y_train, X_test, y_test)
    optimal_model = optimal.best_estimator
    return model_cross_validation(optimal_model, X, y, mae_scorer_cv, 10)
