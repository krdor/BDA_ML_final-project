import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, MeanShift, DBSCAN, estimate_bandwidth
from sklearn import metrics

default_settings = {
    "quantile": 0.3,
    "eps": 0.3,
    "dataset_name": "",
    "x_axis": "x coordinates",
    "y_axis": "y coordinates"
}

n_clusters = 3
n_samples = 500

# Generating artificial datasets and arranging them into a list to iterate over
blobs, _ = make_blobs(n_samples=500, centers=n_clusters, cluster_std=1, random_state=1407)
noisy_circles, _ = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons, _ = make_moons(n_samples=n_samples, noise=0.05)

# preparing a real dataset
df = pd.read_csv('wine-clustering.csv')
wines = df[['Alcohol', 'Color_Intensity']].to_numpy()

datasets = [
    (blobs,
     {
         "quantile": 0.2,
         "eps": 0.5,
         "dataset_name": "artificially constructed clusters"
     }),
    (noisy_circles,
     {
         "quantile": 0.2,
         "eps": 0.2,
         "dataset_name": "circles with noise"
     }),
    (noisy_moons,
     {
         "quantile": 0.2,
         "eps": 0.2,
         "dataset_name": "moons with noise"
     }),
    (wines,
     {
         "quantile": 0.2,
         "eps": 0.4,
         "dataset_name": "wines dataset",
         "x_axis": "Alcohol",
         "y_axis": "Color_Intensity"
     })
]

for (dataset, algo_params) in datasets:
    params = default_settings.copy()
    params.update(algo_params)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"Results for {params['dataset_name']}")
    axs[0, 0].scatter(dataset[:, 0], dataset[:, 1])
    axs[0, 0].set(title='Raw data', xlabel=params['x_axis'], ylabel=params['y_axis'])
    validation = {'method': ['kmeans', 'meanshift', 'dbscan'], 'Silhouette Coefficient': [],
                  'Calinski-Harabasz Index': [], 'Davies-Bouldin Index': []}

    kmeans = KMeans(n_clusters=n_clusters, init='random', n_init='auto')
    kmeans.fit_predict(dataset)
    axs[0, 1].scatter(dataset[:, 0], dataset[:, 1], c=kmeans.labels_)
    axs[0, 1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=50, c='red', alpha=.5)
    axs[0, 1].set(title='Data processed with KMeans algorithm', xlabel=params['x_axis'], ylabel=params['y_axis'])
    validation['Silhouette Coefficient'].append(metrics.silhouette_score(dataset, kmeans.labels_, metric='euclidean'))
    validation['Calinski-Harabasz Index'].append(metrics.calinski_harabasz_score(dataset, kmeans.labels_))
    validation['Davies-Bouldin Index'].append(metrics.davies_bouldin_score(dataset, kmeans.labels_))

    bandwidth = estimate_bandwidth(dataset, quantile=params['quantile'])
    meanshift = MeanShift(bandwidth=bandwidth)
    meanshift.fit_predict(dataset)
    axs[1, 0].scatter(dataset[:, 0], dataset[:, 1], c=meanshift.labels_)
    axs[1, 0].set(title='Data processed with MeanShift algorithm', xlabel=params['x_axis'], ylabel=params['y_axis'])
    validation['Silhouette Coefficient'].append(
        metrics.silhouette_score(dataset, meanshift.labels_, metric='euclidean'))
    validation['Calinski-Harabasz Index'].append(metrics.calinski_harabasz_score(dataset, meanshift.labels_))
    validation['Davies-Bouldin Index'].append(metrics.davies_bouldin_score(dataset, meanshift.labels_))

    dbscan = DBSCAN(eps=params['eps'])
    dbscan.fit_predict(dataset)
    axs[1, 1].scatter(dataset[:, 0], dataset[:, 1], c=dbscan.labels_)
    axs[1, 1].set(title='Data processed with DBSCAN algorithm', xlabel=params['x_axis'], ylabel=params['y_axis'])
    validation['Silhouette Coefficient'].append(metrics.silhouette_score(dataset, dbscan.labels_, metric='euclidean'))
    validation['Calinski-Harabasz Index'].append(metrics.calinski_harabasz_score(dataset, dbscan.labels_))
    validation['Davies-Bouldin Index'].append(metrics.davies_bouldin_score(dataset, dbscan.labels_))

    print("\n", params['dataset_name'])
    print(pd.DataFrame(data=validation).to_string())
    plt.show()
