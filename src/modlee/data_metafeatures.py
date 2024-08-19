#%%
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import random
import math
import time

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from torch.utils.data import DataLoader
import spacy
from pymfe.mfe import MFE

import torch
import torchvision
from torchvision.models import (
    resnet18,
    vgg16,
    densenet121,
    alexnet,
    mobilenet_v3_small,
)  # , vit_l_32
import torch.nn.functional as F

from modlee.utils import closest_power_of_2, _make_serializable

fixed_resize = 32
import logging, warnings

for _logger in ["sklearn", "pymfe", "numpy"]:
    pl_logger = logging.getLogger(_logger)
    pl_logger.propagate = False
    pl_logger.setLevel(logging.ERROR)
warnings_to_ignore = [
    "Number of distinct clusters",
    "Will set it as 'np.nan'",
    "Input data for shapiro has range zero",
    "Can't extract feature",
    "invalid value encountered in divide",
    "Mean of empty slice",
    "It is not possible make equal discretization",
    "invalid value encountered in scalar divide",
    "invalid value encountered in double_scalars",
    "invalid value encountered in double scalars",
]
for warning_to_ignore in warnings_to_ignore:
    warnings.filterwarnings("ignore", f".*{warning_to_ignore}.*")

module_available = True


def bench_kmeans_unsupervised(batch, n_clusters=[2, 4, 8, 16, 32], testing=False):
    """
    Calculate k-means clusters for a batch of data.

    :param batch: The batch of data.
    :param n_clusters: Number of clusters to calculate, defaults to [2, 4, 8, 16, 32],
    :param testing: Flag for testing and calculating with a smaller batch, defaults to False,
    :return: A dictionary of {'kmeans':calculated_kmeans_clusters}
    """
    if testing == True:
        n_clusters = [2, 4, 8]

    # flatten all but first dimension
    batch = batch.reshape(batch.shape[0], -1)
    kmeans_results = {}
    batch_size = batch.shape[0]
    for nc in n_clusters:
        if nc > batch_size:
            continue
        start_time = time.time()

        kmeans = KMeans(n_clusters=nc, init="random", n_init="auto")  # KMeans
        labels = kmeans.fit_predict(batch)

        inertia = kmeans.inertia_
        silhouette_avg = silhouette_score(
            batch, labels
        )  # x is your data, labels are cluster labels
        ch_score = calinski_harabasz_score(batch, labels)
        db_score = davies_bouldin_score(batch, labels)

        end_time = time.time()

        kmeans_results[nc] = {
            "inertia": inertia,
            "silhouette_score": silhouette_avg,
            "calinski_harabasz_score": ch_score,
            "davies_bouldin_score": db_score,
            "time_taken": end_time - start_time,
        }

    # print(kmeans_results)
    # STOP

    return {"kmeans": kmeans_results}


def extract_features_from_model(model, batch):
    """
    Extract features for a data batch using a neural network model.

    :param model: The model to use for feature extraction.
    :param batch: The data batch on which to calculate features.
    :return: The calculated features.
    """

    features = []
    with torch.no_grad:
        input_tensor = batch  # torch.from_numpy(x)
        outputs = model(input_tensor)
        features.append(outputs)
    features = torch.cat(features, dim=0)

    return features


def pad_image_channels(x, desired_channels=3):
    """
    Pad an image with extra channels.
    Uses dimeension order [batch, channel, width, height].

    :param x: The image tensor to pad.
    :param desired_channels: Desired number of channels, defaults to 3.
    :return: The padded tensor.
    """

    # Calculate the number of channels to pad
    channels_to_pad = desired_channels - x.shape[1]
    # Create a tensor with zeros for padding
    padding_tensor = torch.zeros((x.shape[0], channels_to_pad, x.shape[2], x.shape[3]))
    # Concatenate the original tensor and the padding tensor along the channel dimension
    padded_tensor = torch.cat((x, padding_tensor), dim=1)

    return padded_tensor


def sample_image_channels(x, num_sample=3):
    """
    Sample random channels from an image [batch_size, channel, width, height].

    :param x: The image tensor to sample from.
    :param num_sample: Number of channels to sample, defaults to 3.
    :return: A tensor of sampled channels.
    """

    # Generate random indices for channel selection
    random_indices = torch.randperm(x.shape[1])[:num_sample]
    # Select the channels using the random indices
    selected_channels = x[:, random_indices]

    return selected_channels


def sample_image_from_video(x, num_channels=1):
    """
    Sample 3-channel images from a video tensor [batch_size, frames, channels, width, height].

    :param x: The video tensor.
    :param num_channels: The number of channels to sample.
    :return: A tensor of images.
    """

    # Generate random channel indices for each batch element
    random_channel_indices = torch.randint(0, x.shape[1], (x.shape[0], num_channels))

    # Use the channel indices to select the channels
    selected_channels = x[
        torch.arange(x.shape[0]),  # Batch indices
        random_channel_indices[:, 0],  # Random channel indices
        :,
        :,
        :,  # All spatial dimensions
    ]

    # Add a new dimension to the selected_channels tensor
    selected_channels = selected_channels.squeeze(1)

    return selected_channels


def manipulate_x_5(x):
    """
    Process a 5-dimensional tensor, assumed to be video-like [batch_size, frames, channels, width, height], into image-like [batch_size, channels, width, height].

    :param x: The tensor.
    :return: A subsample of the tesnor
    """
    x = sample_image_from_video(x)
    return x


def manipulate_x_4(x):
    """
    Process a 4-dimensional tensor, assumed to be image-like [batch_size, channelw, width, height], into subchannels

    :param x: The image to process.
    :return: Sampled channels from the image.
    """
    if x.shape[1] < 3:
        x = pad_image_channels(x)
    elif x.shape[1] > 3:
        x = sample_image_channels(x)

    global fixed_resize

    resized_tensor = F.interpolate(
        x, size=fixed_resize, mode="bilinear", align_corners=False
    )

    return resized_tensor


def manipulate_x_3(x):
    """
    Process a 3-dimensional tensor [batch_size, width, height] by resizing to a fixed size.

    :param x: The tensor.
    :return: The tensor, resized.
    """

    global fixed_resize

    try:
        resized_tensor = F.interpolate(
            x.unsqueeze(1), size=fixed_resize, mode="bilinear", align_corners=False
        )
        resized_tensor = resized_tensor.squeeze(1)
    except:
        return x

    return resized_tensor


def manipulate_x_2(x):
    """
    Subsample a 2D tensor to the first 10000 values.

    :param x: The tensor to subsample.
    :return: A subsample of the tensor.
    """
    # if data is very large sample first few elements
    if x.shape[-1] > 10000:
        x = x[:, :10000]
    return x


def manipulate_x_1(x):
    """
    Unsqueeze a 1D tensor.

    :param x: The tensor.
    :return: The tensor with an extra beginning dimension.
    """
    return x.unsqueeze(1)


def get_image_features(x, testing=False):
    """
    Get features for a batch of image data.

    :param x: The batch of image data.
    :param testing: Flag to calculate on a smaller test subsample of the data, defaults to False.
    :return: A dictionary of the features.
    """

    # assumptions: x has the following structure (num,ch,h,w), or (num,?,ch,h,w)

    # cases
    #   - x shape : (num,3,h,w): all below should work if h&w are compatible
    #   - x shape : (num,h,w): output only raw
    #   - x shape : (num,w): output only raw
    #   - x shape : (num): output only raw

    # --- manipulate x ---
    #   - x shape : (num,<3,h,w): duplicate image chanels? zero image channels?
    #   - x shape : (num,>3,h,w): take first 3 ch? randomly sample 3 channels?
    #   - x shape : (num,?,ch,h,w): randomly take a slice of "video", then treat as above case ...

    # print('x shape before manipulation: ',x.shape)
    x_raw = x

    if len(x.size()) == 5:
        if x.size()[2] != min(list(x.size())):
            print(
                "WARNING: We require datasets to be formatted (num_dataset_examples,num_images,num_ch,h,w). Encountered tensor from dataset of size {}".format(
                    x.size()
                )
            )
        x = manipulate_x_5(x)

    if len(x.size()) == 4:
        if x.size()[1] != min(list(x.size())):
            print(
                "WARNING: We require datasets to be formatted (num_dataset_examples,num_ch,h,w). Encountered tensor from dataset of size {}".format(
                    x.size()
                )
            )
        x = manipulate_x_4(x)

    if len(x.size()) == 3:
        x = manipulate_x_3(x)

    if len(x.size()) == 2:
        x = manipulate_x_2(x)

    if len(x.size()) == 1:
        x = manipulate_x_1(x)

    assert (
        len(x.size()) <= 5
    ), "datastats: We can only accommodate datasets of up up to 5 dimensions, Encountered tensor from dataset of size {}".format(
        x.size()
    )

    assert (
        len(x.size()) > 1
    ), "datastats: We can only accommodate datasets of between 2 and  5 dimensions, Encountered tensor from dataset of size {}".format(
        x.size()
    )

    # ------------------------------------------------

    # print('x shape after manipulation: ',x.shape)
    # sleep(5)

    if testing == True:
        # Load the pre-trained models
        model_resnet = resnet18(pretrained=True)
        model_resnet.fc = torch.nn.Identity()

        # Set the models to evaluation mode
        model_resnet.eval()

        name_model_pairs = [["resnet18", model_resnet]]

    else:
        # TODO - Deprectaed, whole function should be refactored or removed
        # Load the pre-trained models
        model_resnet = model_vgg = None
        # model_resnet = resnet18(pretrained=True)
        # model_resnet.fc = torch.nn.Identity()

        # model_vgg = vgg16(pretrained=True)
        # model_vgg.classifier = torch.nn.Sequential(
        #     *list(model_vgg.classifier.children())[:-1]
        # )
        # # Set the models to evaluation mode
        # model_resnet.eval()
        # model_vgg.eval()
        name_model_pairs = [
            ["resnet18", model_resnet],
            ["vgg16", model_vgg],
            # ['densenet121',model_densenet],
            # ['alexnet',model_alexnet],
            # ['mobilenet_v3_small',model_mobile_small],
            # ['vit_l_32',model_vit_l_32],
        ]

    feature_dict = {}

    for pair in name_model_pairs:
        # feature_dict[pair[0]] = extract_features(pair[1], x)
        try:
            feature_dict[pair[0]] = torch.zeros(1,1)
            # feature_dict[pair[0]] = extract_features_from_model(pair[1], x)
        except:
            # if model is not compatible with data, just skip for now
            pass

    feature_dict["raw"] = x_raw

    return feature_dict


# NEED TO UPDATE THIS ON OTHER SIDE
def sample_dataloader(train_dataloader, num_sample):
    """
    Sample batches from a dataloader.

    :param train_dataloader: The dataloader to sample from.
    :param num_sample: The number of samples.
    :return: A tuple of dataset_size, batch_elements, and the original size of the batch.
    """

    # goal: take dataloader, sample batches, seperate elements into own arrays for indpendent analysis

    # assumptions:
    #   - batches may be shuffled or not
    #   - user may have memory constraints so loading all batches into memory is not viable
    #   - prior to calling this function we know how many samples we want to take
    #   - we don't know the full size of the dataset and we need to return this

    # 1: Loop through all batches counting number of batches and getting batch_size
    num_batches = 0
    try:
        for i, batch in enumerate(train_dataloader):
            if i == 0:
                # if type(batch)==list or type(batch)==tuple:
                if type(batch) in [list, tuple]:
                    _subbatch = batch[0]
                    batch_size = _subbatch.size()[0]
                    num_batch_elements = len(batch)
                else:
                    # assume train_dataloader returns a tensor
                    batch_size = batch.size()[0]
                    num_batch_elements = 1
                # print(type(batch))
                # print(batch_size)
            num_batches += 1
    except:
        batch_size = train_dataloader.batch_size
        num_batches = len(train_dataloader.dataset) // batch_size
        num_batch_elements = len(next(iter(train_dataloader)))

        pass

    assert num_batches != 0, "num_batches={}".format(num_batches)

    # 2: randomly sample batches at specific inds that total sampling size ~ num_sample: easier on memory
    num_batches_to_sample = num_sample // batch_size
    num_batches_to_sample = min(
        [num_batches_to_sample, num_batches - 1]
    )  # handles case where num_sample>size of dataset

    inds_to_sample = set(
        random.sample(list(np.arange(num_batches - 1)), num_batches_to_sample)
    )  # -1 avoids incommensurate batches

    assert (
        len(inds_to_sample) == num_batches_to_sample
    ), "len(inds_to_sample)={},num_batches_to_sample={}".format(
        len(inds_to_sample), num_batches_to_sample
    )

    sampled_batches = [
        batch for i, batch in enumerate(train_dataloader) if i in inds_to_sample
    ]

    assert num_batches_to_sample == len(
        sampled_batches
    ), "num_batches_to_sample={} != len(sampled_batches)={}".format(
        num_batches_to_sample, len(sampled_batches)
    )

    # 3: organize batch_elements into their own arrays

    dataset_size = num_batches * batch_size

    batch_elements = []
    try:
        for i in range(num_batch_elements):
            batch_elements.append(
                torch.concat([torch.Tensor(b[i]).cpu() for b in sampled_batches])
            )
    except:
        pass

    batch_elements_orig_shapes = [b.shape for b in batch_elements]

    return dataset_size, batch_elements, batch_elements_orig_shapes

def get_n_samples(dataloader, n_samples=100):
    """
    Get a number of samples from a dataloader

    :param dataloader: The dataloader.
    :param n_samples: The number of samples, defaults to 100.
    :return: An iterable of batch elements, each of length n_samples.
    """
    batch = next(iter(dataloader))
    while len(batch[0]) < n_samples:
        _batch = next(iter(dataloader))
        # for b,_b in zip(batch, _batch):
        for i,b in enumerate(_batch):
            if isinstance(b, torch.Tensor):
                batch[i] = torch.cat((batch[i],b), dim=0)
            else:
                batch[i] = list(batch[i]) + list(b)
                # batch[i].append(b)
            batch[i] = batch[i][:n_samples]
            
    return batch
            

class DataMetafeatures(object):
    """
    An object to hold metafeatures for a dataset, as loaded from a dataloader.
    """

    def __init__(self, dataloader, num_sample=1000, testing=False):
        """
        Constructor for the data metafeature object.

        :param dataloader: The dataloader.
        :param num_sample: The number of samples in the subset to calculate metafeatures, defaults to 1000.
        :param testing: Flag for testing and using a smaller subset for calculation (100), defaults to False.
        """

        if testing == True:
            num_sample = 100

        self.testing = testing

        self.dataloader = dataloader

        # -----------------------------

        self.num_sample = num_sample

        # general and independent of any data type or ml task
        self.dataset_size, self.batch_elements, self.batch_elements_orig_shapes = sample_dataloader(
            dataloader, num_sample
        )

        # needs to be defined in child classes based on data type, still ml task independent
        self.batch_features = self.get_raw_batch_elements()

        # general and independent of any data type or ml task
        start_time = time.time()
        self.batch_stats = self.get_stats()
        batch_stats_time = time.time() - start_time
        # self.batch_stats = []

        # Features from PyMFE, a meta-feature extraction library
        start_time = time.time()
        self.mfe_features = self.get_mfe_features()
        self.mfe = self.get_mfe()
        mfe_time = time.time() - start_time
        # print(f"Batch stats: {batch_stats_time}; MFE time: {mfe_time}")
        # for batch_idx,batch_mfe_features in enumerate(self.mfe_features):
        #     print(batch_mfe_features)
        #     self.batch_stats[batch_idx].update({
        #         'mfe_features':batch_mfe_features
        #     })
        # for mfe_key,mfe_value in self.mfe_features.items():
        #     self.batch_stats[mfe_key].update(mfe_value)
        
        self.properties = self.get_properties()

        # general and independent of any data type or ml task
        self.stats_rep = self.get_features()
        # TODO - deprecated "stats_rep" for "features"
        self.features = self.stats_rep
        # breakpoint()
        self.features = {
            # **self.embedding,
            **self.mfe,
            **self.properties
        }
        # self.features = self._make_serializable(self.features)

       # self.stats_rep.update(self.mfe_features)
        self._serializable_stats_rep = self._make_serializable(self.stats_rep)

    def get_raw_batch_elements(self):
        """
        Convert features to a list of dictionaries.

        :return: A list of {'raw': feature} 
        """
        return [{"raw": element} for element in self.batch_elements]

    def get_features(self):
        """
        Get features for batch elements.

        :return: A dictionary of {'batch_element' : features}
        """

        stats_rep = {"dataset_size": self.dataset_size, "num_sample": self.num_sample}

        for i in range(len(self.batch_elements)):

            batch_stat = self.batch_stats[i]
            batch_stat["orig_shape"] = self.batch_elements_orig_shapes[i]
            batch_stat["mfe_features"] = self.mfe_features[i]

            stats_rep["batch_element_{}".format(i)] = batch_stat

        # stats_rep = self._f32_to_f16(stats_rep)
        return stats_rep

    def get_properties(self):
        """
        Get properties — features that are not calculated, e.g. shapes
        """
        ret = {
            'dataset_size':self.dataset_size,
        }
        samples = get_n_samples(self.dataloader)
        for i,sample in enumerate(samples):
            # Get shape
            if not isinstance(sample, torch.Tensor):
                sample = np.array(sample)
            sample_shape = list(sample.shape)
            sample_dim = len(sample_shape)
            ret.update({
                f'elem_{i}_shape':sample_shape,
                f'elem_{i}_dims':sample_dim,
                })
        return ret


    get_stats_rep = get_features

    def get_stats(self):
        """
        Get statistical features for batch elements.
        Includes feature shape, k-means clustering, and time taken to calculate features.

        :return: A list of statistical features.
        """

        batch_stats = []

        for batch in self.batch_features:
            feature_stats = {}
            for feature_name, feature in batch.items():
                feature_shape = feature.shape
                start_time = time.time()
                stats = bench_kmeans_unsupervised(feature, testing=self.testing)
                end_time = time.time()
                feature_stats[feature_name] = {
                    "feature_shape": feature_shape,
                    "stats": stats,
                    "time_taken": end_time - start_time,
                }

            batch_stats.append(feature_stats)

        return batch_stats

    # def _f32_to_f16(self,base_dict):
    def _make_serializable(self, base_dict):
        """
        Make a dictionary serializable (e.g. by pickle or json) by converting floats to strings.

        :param base_dict: The dictionary to convert.
        :return: The serializable dict.
        """
        return _make_serializable(base_dict)

    def get_mfe_features(self):
        """
        Get features for all batch elements with PyMFE.

        :return: A list of metafeatures.
        """
        # mfe_features = {}
        mfe_features = []
        for batch_idx, batch_element in enumerate(self.batch_elements):
            feature_dict = self.get_mfe_on_batch(batch_element)
            mfe_features.append(feature_dict)
        return mfe_features

    def get_mfe_on_batch(self, batch_element):
        """
        Get features for a batch element with PyMFE.

        :param batch_element: The batch element to calculate.
        :return: A dictionary of features for the batch element.
        """
        
        # If the batch element is a tensor, convert to numpy array
        if isinstance(batch_element, torch.Tensor):

            if len(batch_element.shape) > 2:
                if len(batch_element.shape) >= 3:
                    batch_element = torchvision.transforms.functional.resize(
                        batch_element, size=(30, 30)
                    )
                # print(batch_element.shape)
                batch_element = batch_element.flatten(start_dim=1)
            batch_element = batch_element.numpy()
        mfe = MFE(
            # groups="all",
            groups="default"
        )
        mfe.fit(
            batch_element,
            # verbose=2,
        )
        features = mfe.extract()
        feature_dict = {k: v for k, v in zip(*features)}
        return feature_dict
    get_mfe_on_element = get_mfe_on_batch
    
    def get_mfe(self):
        """
        Get PyMFE features for every element in the dataloader

        :return: A dictionary of {mfe_feature : mfe_value}
        """
        samples = get_n_samples(self.dataloader)
        sample_mfes = [self.get_mfe_on_element(sample) for sample in samples]
        ret = {}
        for i,sample_mfe in enumerate(sample_mfes):
            ret.update({f'{k}_{i}':v for k,v in sample_mfe.items()})
        return ret



class ImageDataMetafeatures(DataMetafeatures):
    """
    Image-based DataMetafeatures.
    """

    def __init__(self, dataloader, embd_model=None, *args, **kwargs):
        super().__init__(dataloader, *args, **kwargs)
        if not embd_model:
            self.embd_model = torchvision.models.resnet18(
                weights='IMAGENET1K_V1'
            )
            self.embd_model.eval()
        self.embedding = self.get_embedding()
        self.features.update(self.embedding)
        self.features = self._make_serializable(self.features)
        pass

    def get_raw_batch_elements(self):
        """
        Get the raw batch elements for an image-based dataset.

        :return: A list of image-based features.
        """
        return [
            get_image_features(element, testing=self.testing)
            for element in self.batch_elements
        ]
    
    def get_embedding(self, index=0, max_len=100):
        samples = get_n_samples(self.dataloader)
        # Assume inputs are the first batch element
        x = samples[index]
        with torch.no_grad():
            embds = self.embd_model(x)
            embds = embds[:,:max_len]
       
        # Return distributions of each embedding axis
        ret = {f'embd_{index}_mean_{i}':float(v.numpy()) for i,v in enumerate(embds.mean(axis=0))}
        ret.update({f'embd_{index}_std_{i}':float(v.numpy()) for i,v in enumerate(embds.std(axis=0))}) 
        return ret

class TextDataMetafeatures(DataMetafeatures):
    def __init__(self, dataloader, nlp_model=None, *args, **kwargs):
        super().__init__(dataloader, *args, **kwargs)
        # self.dataloader = dataloader
        if not nlp_model:
            # TODO - consider using a larger model embedding e.g. en_code_web_sm -> 300D
            # and truncate
            self.nlp_model = spacy.load('en_core_web_sm')
        self.embedding = self.get_embedding()
        pass

    def get_embedding(self, index=None, max_len=100, *args, **kwargs):
        """
        Get embeddings from the dataloader.

        :param index: The index in a batch of the string elements to embed, defaults to 1
        :return: A dictionary of {embd_i : embd_value}
        """
        samples = get_n_samples(self.dataloader)
        # Find the index of the first batch of strings
        if not index:
            index = 0
            while not isinstance(samples[index][0], str):
                index += 1
                if index == len(samples):
                    raise IndexError(f"No string elements in {self}, cannot calculate embedding with spaCy")
        embds = list(map(lambda x: self.nlp_model(x).vector, samples[index]))
        embds = torch.Tensor(embds)
        embds = embds[:,:max_len]
        # Return distributions of each embedding axis
        # TODO - consider how batch elements are sorted, should they be indexed by the index that the 
        # string elements appear? at "embd_{INDEX}..."
        # ret = {f'embd_{index}_mean_{i}':float(v.numpy()) for i,v in enumerate(embds.mean(axis=0))}
        # ret.update({f'embd_{index}_std_{i}':float(v.numpy()) for i,v in enumerate(embds.std(axis=0))})
        # TODO - consider this assumption, not indexing the embeddings at all
        ret = {f'embd_mean_{i}':float(v.numpy()) for i,v in enumerate(embds.mean(axis=0))}
        ret.update({f'embd_std_{i}':float(v.numpy()) for i,v in enumerate(embds.std(axis=0))})
        return ret
        breakpoint()

class TabularDataMetafeatures(DataMetafeatures):
    def __init__(self, dataloader, *args, **kwargs):
        super().__init__(dataloader, *args, **kwargs)
        self.stats_rep = self.get_features()
        # Ensure that self.features is flat
        self.features.update(self.stats_rep)
        pass

    def get_features(self):
        stats_rep = {}
        for idx, element in enumerate(self.batch_elements):
            if isinstance(element, torch.Tensor):
                np_element = element.numpy()
                stats = self.calculate_statistical_summary(np_element)
                # Add a prefix to each stat key to ensure uniqueness and flat structure
                for key, value in stats.items():
                    stats_rep[f'batch_element_{idx}_{key}'] = value
        return stats_rep

    def calculate_statistical_summary(self, data):
        df = pd.DataFrame(data)
        summary = {
            'mean': df.mean().tolist(),
            'median': df.median().tolist(),
            'variance': df.var().tolist(),
            'std_dev': df.std().tolist(),
            'min': df.min().tolist(),
            'max': df.max().tolist(),
            'range': (df.max() - df.min()).tolist(),  # Add range calculation
            'quantiles_25': df.quantile(0.25).tolist(),
            'quantiles_50': df.quantile(0.50).tolist(),
            'quantiles_75': df.quantile(0.75).tolist()
        }
        return summary

class TimeSeriesDataMetafeatures(DataMetafeatures):
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def get_single_batch(self):
        for batch in self.dataloader:
            if isinstance(batch, dict) and 'encoder_cont' in batch:
                return batch['encoder_cont']
            elif isinstance(batch, tuple) and len(batch) == 2:
                data_dict = batch[0]
                if 'encoder_cont' in data_dict:
                    return data_dict['encoder_cont']
                else:
                    raise ValueError("Batch is not in expected dictionary format or missing 'encoder_cont' key.")
        raise ValueError("No valid batch found in dataloader.")
    

    def calculate_metafeatures(self):
        data = self.get_single_batch()
        print(f"Data shape: {data.shape}")
        
        if data.size == 0:
            raise ValueError("No data available in the batch.")
        
        # Flatten the 3D data to 2D for metafeature calculation
        num_samples, seq_len, num_features = data.shape
        data_2d = data.reshape(-1, num_features).numpy()  # Convert tensor to NumPy array
        print(f"Flattened data shape: {data_2d.shape}")
        print(f"Flattened data: {data_2d}")
        
        # Extract metafeatures using pymfe
        mfe = MFE(groups=["statistical", "model-based", "info-theory"])
        mfe.fit(data_2d)  # Pass the flattened NumPy array
        ft = mfe.extract()
        pymfe_features = dict(zip(ft[0], ft[1]))
        print(f"Extracted pymfe features: {pymfe_features}")
        
        # Calculate additional features
        additional_features = {}
        
        # Combine all features into a single time series
        combined_series = data_2d.flatten()
        print(f"Combined series: {combined_series}")
        
        # Calculate quantiles
        quantiles = np.percentile(combined_series, [25, 50, 75])
        print(f"Quantiles: {quantiles}")
        
        # Calculate autocorrelation and partial autocorrelation
        autocorr = acf(combined_series, nlags=1)
        print(f"Autocorrelation: {autocorr}")
        if len(combined_series) > 2:
            partial_autocorr = pacf(combined_series, nlags=min(1, len(combined_series) // 2 - 1))
        else:
            partial_autocorr = [np.nan]  # or some default value
        print(f"Partial Autocorrelation: {partial_autocorr}")
        
        if len(autocorr) > 1:
            autocorr_lag1 = autocorr[1]
        else:
            autocorr_lag1 = np.nan  # or some default value
        
        if len(partial_autocorr) > 1:
            partial_autocorr_lag1 = partial_autocorr[1]
        else:
            partial_autocorr_lag1 = np.nan  # or some default value
        
        # Decompose the combined time series to extract trend and seasonality
        if len(combined_series) >= 24:  # Check if we have enough data for a period of 12
            decomposition = seasonal_decompose(combined_series, period=12, model='additive', extrapolate_trend='freq')
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            trend_strength = np.nanmean(trend)
            seasonal_strength = np.nanmean(seasonal)
            print(f"Trend: {trend}")
            print(f"Seasonal: {seasonal}")
        else:
            trend_strength = np.nan
            seasonal_strength = np.nan
        
        additional_features.update({
            "combined_quantile_25": quantiles[0],
            "combined_quantile_50": quantiles[1],
            "combined_quantile_75": quantiles[2],
            "combined_autocorr_lag1": autocorr_lag1,
            "combined_partial_autocorr_lag1": partial_autocorr_lag1,
            "combined_trend_strength": trend_strength,
            "combined_seasonal_strength": seasonal_strength,
        })
        
        # Combine pymfe features with additional features
        combined_features = {**pymfe_features, **additional_features}
        print(f"Combined features: {combined_features}")
        return combined_features
    
    def print_meta(self, features):
        # Calculate the maximum length of the keys
        max_key_length = max(len(key) for key in features.keys())
        
        # Print each key-value pair with the keys aligned
        for key, value in features.items():
            print(f"{key:<{max_key_length}} : {value}")
        return None
