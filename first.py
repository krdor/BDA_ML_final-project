import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn import metrics

# Function that takes dataset, does the clustering with chosen algorithms and shows the plots separately
#   for each dataset
def clust_n_plot(data, k):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].scatter(data[:, 0], data[:, 1])
    validation = {'method': ['kmeans', 'meanshift', 'dbscan'], 'Silhouette Coefficient': [],
                  'Calinski-Harabasz Index': [], 'Davies-Bouldin Index': []}

    kmeans = KMeans(n_clusters=k, init='random', n_init='auto')
    kmeans.fit_predict(data)
    axs[0, 1].scatter(data[:, 0], data[:, 1], c=kmeans.labels_)
    axs[0, 1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='red', alpha=.5)
    validation['Silhouette Coefficient'].append(metrics.silhouette_score(data, kmeans.labels_, metric='euclidean'))
    validation['Calinski-Harabasz Index'].append(metrics.calinski_harabasz_score(data, kmeans.labels_))
    validation['Davies-Bouldin Index'].append(metrics.davies_bouldin_score(data, kmeans.labels_))

    bandwidth = estimate_bandwidth(data, quantile=.2)
    meanshift = MeanShift(bandwidth=bandwidth)
    meanshift.fit_predict(data)
    axs[1, 0].scatter(data[:, 0], data[:, 1], c=meanshift.labels_)
    validation['Silhouette Coefficient'].append(metrics.silhouette_score(data, meanshift.labels_, metric='euclidean'))
    validation['Calinski-Harabasz Index'].append(metrics.calinski_harabasz_score(data, meanshift.labels_))
    validation['Davies-Bouldin Index'].append(metrics.davies_bouldin_score(data, meanshift.labels_))

    dbscan = DBSCAN(eps=.2)
    dbscan.fit_predict(data)
    axs[1, 1].scatter(data[:, 0], data[:, 1], c=dbscan.labels_)
    validation['Silhouette Coefficient'].append(metrics.silhouette_score(data, dbscan.labels_, metric='euclidean'))
    validation['Calinski-Harabasz Index'].append(metrics.calinski_harabasz_score(data, dbscan.labels_))
    validation['Davies-Bouldin Index'].append(metrics.davies_bouldin_score(data, dbscan.labels_))

    print(pd.DataFrame(data=validation).to_string())
    plt.show()


# Initializing parameters
clust_amt = 3
n_samples = 500

# Generating artificial datasets and arranging them into a list to iterate over
blobs, _ = make_blobs(n_samples=500, centers=clust_amt, cluster_std=1, random_state=1407)
noisy_circles, _ = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons, _ = make_moons(n_samples=n_samples, noise=0.05)

# preparing a real dataset
df = pd.read_csv('wine-clustering.csv')
wines = df[['Alcohol', 'Color_Intensity']].to_numpy()

datasets = [blobs, noisy_circles, noisy_moons, wines]

# Calling the clustering and plotting function for each dataset
for dataset in datasets:
    clust_n_plot(dataset, clust_amt)
