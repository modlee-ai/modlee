import sys
import io
from datasets.download.streaming_download_manager import xgetsize

import numpy as np
import ast
from datetime import datetime

from ast import literal_eval

from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import numpy as np
from scipy.cluster.vq import kmeans, vq


from time import time

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, Subset, TensorDataset


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

    # homogeneity: each cluster contains only members of a single class.
    # completeness: all members of a given class are assigned to the same cluster.
    # The V-measure is the harmonic mean between homogeneity and completeness:
    # The Rand Index computes a similarity measure between two clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.
    # Adjusted Mutual Information (AMI) is an adjustment of the Mutual Information (MI) score to account for chance. It accounts for the fact that the MI is generally higher for two clusterings with a larger number of clusters, regardless of whether there is actually more information shared.

    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    # The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    results_labels = ['inertia', 'homo', 'compl', 'v', 'ars', 'ami', 'sil']

    results_dict = {results_labels[i]: results[i+2]
                    for i in range(len(results_labels))}

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}")

    return results_dict


def bench_kmeans_unsupervised(x):

    x = x.numpy(force=True).astype(float)
    
    # breakpoint()
    print(x.shape)
    print(type(x[0,0]))
    
    k_values = [1, 2, 3, 5, 10, 15]  # Range of cluster numbers to evaluate

    inertias = []
    for k in k_values:
        centroids, inertia = kmeans(x, k)
        inertias.append(inertia)

    return {k_values[i]: inertias[i] for i in range(len(k_values))}


def stats_text(input_ids, labels, tokenizer):
    # adding in metrics about text and tokenizer
    import torch
    import numpy as np

    # Get the number of unique tokens
    input_ids = torch.from_numpy(input_ids)
    unique_tokens = torch.unique(input_ids)

    # Calculate the count of unique tokens
    num_unique_tokens = unique_tokens.size(0)
    number_of_elements = input_ids.size(0)
    max_tokens_per_element = input_ids.size(1)

    # Get the vocabulary size
    vocab = tokenizer.get_vocab()
    vocab_counts = [count for word, count in vocab.items()]

    v_size = len(vocab)
    v_stat_1 = np.max(vocab_counts)/np.mean(vocab_counts)
    v_stat_2 = np.std(vocab_counts)/np.max(vocab_counts)

    return {'num_unique_tokens': num_unique_tokens,
            'max_tokens_per_element': max_tokens_per_element,
            'v_size': v_size,
            'v_stat_1': v_stat_1,
            'v_stat_2': v_stat_2
            }


def extract_features(model, x):

    try:
        features = []
        with torch.no_grad():
            input_tensor = torch.from_numpy(x)
            outputs = model(input_tensor)
            features.append(outputs)
        features = torch.cat(features, dim=0)
    except:
        features = []

    return features


def get_pretrained_features(x):

    from torchvision.models import resnet18, vgg16, densenet121

    # Load the pre-trained models
    model_resnet = resnet18(pretrained=True)
    model_resnet.fc = torch.nn.Identity()

    model_vgg = vgg16(pretrained=True)
    model_vgg.classifier = torch.nn.Sequential(
        *list(model_vgg.classifier.children())[:-1])

    model_densenet = densenet121(pretrained=True)
    model_densenet.classifier = torch.nn.Identity()

    # Set the models to evaluation mode
    model_resnet.eval()
    model_vgg.eval()
    model_densenet.eval()

    return {'resnet18': extract_features(model_resnet, x),
            'vgg16': extract_features(model_vgg, x),
            'densenet121': extract_features(model_densenet, x)}


class DataStats:

    def __init__(self, x, y=[], tokenizer=None, num_sample=1000):

        # we assume that x will always be in either multi-dim or vector numerical form
        self.x = x
        self.y = y

        # subsample x and y
        inds = [i for i in range(x.shape[0])]
        np.random.shuffle(inds)
        num_sample = min([num_sample, len(inds)])
        self.x = x[inds[:num_sample]]
        if len(y) != 0:
            self.y = y[inds[:num_sample]]

        self.tokenizer = tokenizer

        # set input and output dims of dataset
        self.dataset_dims = self.x.shape
        if len(y) == 0:
            self.output_dim = self.dataset_dims[1:]
        else:
            self.output_dim = max([int(np.max(y)), y.shape[-1]])

        # if x is multi-dim then try to gather a set of pretrained reps
        if len(self.dataset_dims) > 2:
            x_pre = get_pretrained_features(x)
            x_pre['orig'] = self.x
            self.x = x_pre
        else:
            self.x = {'orig': self.x}

        self.data_stats = self.get_difficulty_metrics()

    def get_difficulty_metrics(self, num_sample=1000):

        results_dict = {}

        for key, x in self.x.items():

            data, labels = x, self.y

            if len(labels) != 0 and len(x) != 0:

                data = np.reshape(
                    data, (data.shape[0], np.product(data.shape[1:])))

                if len(labels.shape) > 1:
                    if labels.shape[-1] == 1:
                        labels = np.squeeze(labels, axis=-1)
                    else:
                        labels = np.array([np.argmax(l) for l in labels])
                # print(labels)

                (n_samples, n_features), n_digits = data.shape, np.unique(labels).size

                # TODO - check that this is correct
                pca = PCA(n_components=n_digits).fit(data)
                kmeans = KMeans(init=pca.components_,
                                n_clusters=n_digits, n_init=1)
                results_x_dict = bench_k_means(
                    kmeans=kmeans, name="Kmeans-PCA", data=data, labels=labels)
                # print(82 * "_")
            elif len(x) != 0:

                data = np.reshape(
                    data, (data.shape[0], np.product(data.shape[1:])))

                results_x_dict = bench_kmeans_unsupervised(data)
            else:
                results_x_dict = {}

            results_dict[key] = results_x_dict

            results_dict['dataset_dims'] = self.dataset_dims
            results_dict['output_dim'] = self.output_dim

            if self.tokenizer != None:
                for key, value in stats_text(x, self.y, self.tokenizer).items():
                    results_dict[key] = value

        return results_dict
