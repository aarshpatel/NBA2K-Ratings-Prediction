""" 
Feature Selection using Mutual Information 
We want to know which features correlate well with the
ratings and get a better intuition on the relationship between 
nba statistics and 2k player ratings
"""

import sys
sys.path.append('../../src')

from sklearn.feature_selection import mutual_info_regression, SelectKBest
import pandas as pd
import numpy as np
import operator
from commons.utils import *

df = pd.read_csv("../../data/scrape_data/feature_engineered_dataset.csv")
feature_names = list(df.columns)[:-1]

mutual_info = mutual_info_regression(X_all, y_all)

print("Writing mutual info scores for features into text file... ")
with open("mutual_info_features.txt", "w") as f:
    for feature, score in sorted(dict(zip(feature_names, mutual_info)).items(), key=operator.itemgetter(1), reverse=True):
        f.write("{0} {1}\n".format(feature, score))