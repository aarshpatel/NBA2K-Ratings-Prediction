# Cluster Based Regression Model
# We fit a kmeans model to figure out clusters in the dataset
# For each cluster, we can use a regression model to fit on the data points inside that cluster
# For prediction, we find which point that cluster is in, and then use the regression model trained 
# within that cluster and get the prediction using that regression model

import sys
sys.path.append('../../src/')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")
import seaborn
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.base import BaseEstimator, ClassifierMixin


from commons.utils import *

# Fit a KMeans Model

def graph_elbow(X_all, k, output):
    K = range(1,k)
    KM = [KMeans(n_clusters=k).fit(X_all) for k in K]
    centroids = [k.cluster_centers_ for k in KM]

    D_k = [cdist(X_all, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D,axis=1) for D in D_k]
    dist = [np.min(D,axis=1) for D in D_k]
    avgWithinSS = [sum(d)/X_all.shape[0] for d in dist]

    # Total with-in sum of square
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(X_all)**2)/X_all.shape[0]
    bss = tss-wcss

    kIdx = 2

    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, avgWithinSS, 'b*-')
    ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')
    plt.savefig(output)

# graph_elbow(X_all, 10, "../../graphs/kmeans_elbow_method.png")

#  from the graph we figure out 3 clusters was the best in terms of average within cluster sum of squares

# kmeans = KMeans(n_clusters=3)
# cluster_indicator = kmeans.fit_predict(X_all)
# print(cluster_indicator)

class ClusterRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, K):
        self.K = K
        self.kmeans = KMeans(n_clusters=self.K, random_state=1)
    def fit(self, X, Y):
        cluster_to_data = {} # holds a mapping from cluster indicator to the data points in that cluster
        self.cluster_to_trained_classifiers = {}
        self.fitted_kmeans = self.kmeans.fit(X)
        for idx, cluster_indicator in enumerate(self.fitted_kmeans.labels_):
            x_data = X[idx, :][np.newaxis] 
            y_data = Y[idx]
            if cluster_indicator in cluster_to_data:
                cluster_to_data[cluster_indicator][0].append(x_data)
                cluster_to_data[cluster_indicator][1].append(y_data)

            else:
                cluster_to_data[cluster_indicator] = [[x_data], [y_data]]

        cluster_to_data = {cluster_indicator: (np.vstack(data[0]), np.array(data[1])) for cluster_indicator, data in cluster_to_data.iteritems()}        

        
        # Train each cluster's data points on a Ridge Regression model
        for cluster, data in cluster_to_data.iteritems():
            # print("Training Ridge Regression on Cluster {0}".format(cluster))
            ridge = Ridge(alpha=10)
            ridge.fit(data[0], data[1])
            self.cluster_to_trained_classifiers[cluster] = ridge
        
    
    def predict(self, X):
        predictions = self.fitted_kmeans.predict(X)
        new_predictions = [] 
        for idx, cluster_prediction in enumerate(predictions):
            trained_classifier = self.cluster_to_trained_classifiers[cluster_prediction]
            class_prediction = trained_classifier.predict(X[idx, :][np.newaxis])
            new_predictions.append(class_prediction[0])
        return np.asarray(new_predictions)

    def score(self,X,Y):
        """ Returns the Mean Absolute Error between the actual y values and the predictions """
        Yhat = self.predict(X)
        return absolute_error(Y,Yhat)

cluster_to_score = {}
for k in range(1, 20):
    cv_mae_score = model_cross_validation(ClusterRegression(K=k), X_all, y_all, mae_scorer_cv, 10)
    # cluster_regression = ClusterRegression(K=k)
    # cluster_regression.fit(X_train, y_train)
    # mae_score = cluster_regression.score(X_test, y_test)
    cluster_to_score[k] = cv_mae_score 
    print("MAE Score with {0} clusters: {1}".format(k, cv_mae_score))


plt.plot(cluster_to_score.keys(), cluster_to_score.values(), '-o') 
plt.title("Prediction Error vs K")
plt.ylabel("MAE")
plt.xlabel("Number of Clusters (K)")
plt.show()
plt.savefig("../../graphs/prediction_error_ridge.png", dpi=200)
 

test_cluster = ClusterRegression(K=3)
test_cluster.fit(X_all, y_all)
predictions = test_cluster.predict(X_test)
print "MAE Test: ", absolute_error(y_test, predictions)