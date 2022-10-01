import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
from matplotlib import pyplot as plt

def calc_minkowski_distance(data, centroids, p):
    all_dist = []
    for val in data:
        dist = []
        for centroid in centroids:
            dist.append(sum(abs(m1-m2)**p for m1, m2 in zip(val, centroid))**(1/p))
        all_dist.append(dist)
        
    return np.array(all_dist)


def kmeans(data, k=2, max_iterations=100, p=2):
    
    if isinstance(data, pd.DataFrame):data = data.values
    
    index = np.random.choice(len(data), k, replace=False)
    centroids = data[index, :]
    cluster = np.argmin(calc_minkowski_distance(data, centroids, p), axis=1)
    
    for iteration in max_iterations:
        centroids = np.vstack([data[cluster==i, :].mean(axis=0) for i in range(k)])
        tmp_cluster = np.argmin(calc_minkowski_distance(data, centroids, p), axis=1)
        
        if np.array_equal(cluster, tmp_cluster):
            break
        cluster = tmp_cluster
    
    return cluster, centroids