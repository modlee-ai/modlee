import sys, io

# import keras_tuner
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow import keras
from keras import layers, Input, activations
import numpy as np
import ast
from datetime import datetime    

from ast import literal_eval

from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from time import time


def bench_k_means(kmeans, name, data, labels):

    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    results_labels = ['inertia','homo','compl','v','ars','ami','sil']
    
    results_dict = {results_labels[i]:results[i+2] for i in range(len(results_labels))}
    
    # Show the results
    formatter_result = ("{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}")    
    
    return results_dict

class Data:
    
    def __init__(self,x_train,y_train,x_val,y_val):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        
        self.dataset_dims = self.x_train.shape
        self.output_dim = max([int(np.max(y_train)),y_train.shape[-1]])
        # self.data_var = np.var(x_train)

        self.difficulty_metrics = self.get_difficulty_metrics()
        
    def get_difficulty_metrics(self):
        
        data, labels = self.x_train,self.y_train

        inds = [i for i in range(data.shape[0])]
        np.random.shuffle(inds)

        data = data[inds[:10000]]
        labels = labels[inds[:10000]]
#         print(data.shape)
#         print(labels.shape)
        
        data = np.reshape(data,(data.shape[0],np.product(data.shape[1:])))

        if len(labels.shape)>1:
            if labels.shape[-1]==1:
                labels = np.squeeze(labels,axis=-1)
            else:
                labels = np.array([np.argmax(l) for l in labels])                
        # print(labels)
            
        (n_samples, n_features), n_digits = data.shape, np.unique(labels).size

        pca = PCA(n_components=n_digits).fit(data)
        kmeans = KMeans(init=pca.components_, n_clusters=n_digits, n_init=1)
        results_dict = bench_k_means(kmeans=kmeans, name="Kmeans-PCA", data=data, labels=labels)
        # print(82 * "_")
        
        return results_dict
     


