import sys
sys.path.append('../../src/')
from commons.utils import *
from baseline import run_baseline_offensive_defensive_features
from linear_regression import run_linear_offensive_defensive_features
from ridge_regression import run_ridge_offensive_defensive_features
from random_forest_regression import run_forest_offensive_defensive_features


print "Running All Features vs. Offensive Features vs. Defensive Features Experiment:\n\n"

print "    ==> Baseline Model"
run_baseline_offensive_defensive_features()
print "    ==> Linear Regression Model"
run_linear_offensive_defensive_features()
run_ridge_offensive_defensive_features()
run_forest_offensive_defensive_features()


