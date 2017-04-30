""" 
Feature Selection using Mutual Information 
We want to know which features correlate well with the
ratings and get a better intuition on the relationship between 
nba statistics and 2k player ratings
"""

from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import numpy as np

train = np.load('../../data/numpy_data/train.npy') 
X = train[0:,0:-1]
y = train[:, -1]

df = pd.read_csv("../../data/scrape_data/feature_engineered_dataset.csv")
feature_names = list(df.columns)[:-1]

mutual_info = mutual_info_regression(X, y)

print(mutual_info)

for feature, score in dict(zip(feature_names, mutual_info)).iteritems():
    print(feature, score)
