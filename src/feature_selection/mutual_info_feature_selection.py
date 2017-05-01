""" 
Applies feature selection to data using mutual information and
runs different model on the new preprocessed data with the 
best features according to mutual_information 
"""


from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import operator

train = np.load('../../data/numpy_data/train.npy') 
X = train[0:,0:-1]
y = train[:, -1]


X_train = np.load('../../data/numpy_data/X_train.npy')
y_train = np.load('../../data/numpy_data/y_train.npy')
X_test = np.load('../../data/numpy_data/X_test.npy')
y_test = np.load('../../data/numpy_data/y_test.npy')

def absolute_error(act_y, pred_y):
    """ Mean Absolute Error"""
    mae = mean_absolute_error(act_y, pred_y)
    return mae

mae_scorer_gs = make_scorer(absolute_error, greater_is_better=False)


def feature_selection_with_mutual_info(estimator, parameters):
    """ 
    Builds a pipeline using SelectKBest and a regression
    estimator and returns the test MAE for the model 
    """

    mutual_info_fs = SelectKBest(mutual_info_regression)

    print("Building the Pipeline...")
    pipeline = Pipeline([('feature_selection', mutual_info_fs),
            ('estimator', estimator)])


    print("Performing GridSearchCV...")
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=mae_scorer_gs, cv=3, n_jobs=-1)
    cv.fit(X_train, y_train)

    print("Best Model: ", cv.best_estimator_)
    print("Best Params: ", cv.best_params_)
    best_estimator = cv.best_estimator_
    predictions = best_estimator.predict(X_test)
    
    return absolute_error(y_test, predictions)

print "LR: ", feature_selection_with_mutual_info(LinearRegression(), parameters = { "feature_selection__k": range(1, 49) })
# print("\n")
print "Ridge: ", feature_selection_with_mutual_info(Ridge(), parameters = {
        "feature_selection__k": range(1, 49)
        # "estimator__alpha": [0, .01, .1, 1, 10]
})