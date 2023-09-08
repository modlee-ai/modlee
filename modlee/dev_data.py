import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision import datasets as tv_datasets
from torch.utils.data import random_split, Subset, TensorDataset, DataLoader

from sklearn.datasets import load_iris

import numpy as np

try:
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer
except:
    pass

from modlee import modlee_model

# ---------------------------------------
#
# Classes that produce pytorch datasets of type torch.utils.data.dataset
#
# ---------------------------------------


class Image:

    def __init__(self, batch_size=32, num_samples=1000):
        self.batch_size = batch_size
        self.num_samples = num_samples

    def mnist(self, include_y=True):

        # Define the data transformations to apply to the images
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to tensors
            # Normalize the image tensors
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load the MNIST test dataset
        dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            transform=transform,
            download=True
        )

        # Set the random seed for reproducibility
        torch.manual_seed(42)
        # Get the total number of samples in the training dataset
        total_samples = len(dataset)
        # Generate a list of indices for random selection
        indices = torch.randperm(total_samples)[
            :min([self.num_samples, total_samples])]
        # Create a subset of the training dataset with the randomly selected indices
        random_subset = Subset(dataset, indices)

        return random_subset

    def cifar10(self, include_y=True):

        # Define the data transformations to apply to the images
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to tensors
            # Normalize the image tensors
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load the MNIST test dataset
        dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            transform=transform,
            download=True
        )

        # Set the random seed for reproducibility
        torch.manual_seed(42)
        # Get the total number of samples in the training dataset
        total_samples = len(dataset)
        # Generate a list of indices for random selection
        indices = torch.randperm(total_samples)[
            :min([self.num_samples, total_samples])]
        # Create a subset of the training dataset with the randomly selected indices
        random_subset = Subset(dataset, indices)

        return random_subset

    def convert_dataset_to_numpy(self, dataset):
        x_array = np.array([np.array(item[0]) for item in dataset])
        y_array = np.array([np.array(item[1]) for item in dataset])
        # # Print the shape of the subset array
        # print("x_array shape:", x_array.shape)
        # print("y_array shape:", y_array.shape)
        return x_array, y_array


class Tabular:

    def __init__(self, batch_size=32, num_samples=1000):
        self.batch_size = batch_size
        self.num_samples = num_samples

    def iris(self, include_y=True):

        # Load the Iris dataset from scikit-learn
        iris = load_iris()

        # Extract the input features and target labels
        X = iris.data
        X_tensor = torch.tensor(X, dtype=torch.float32)

        if include_y == False:
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            y = iris.target
            y_tensor = torch.tensor(y, dtype=torch.long)
            dataset = TensorDataset(X_tensor, y_tensor)

        # Set the random seed for reproducibility
        torch.manual_seed(42)
        # Get the total number of samples in the training dataset
        total_samples = len(dataset)
        # Generate a list of indices for random selection
        indices = torch.randperm(total_samples)[
            :min([self.num_samples, total_samples])]
        # Create a subset of the training dataset with the randomly selected indices
        random_subset = Subset(dataset, indices)

        return random_subset

    def convert_dataset_to_numpy(self, dataset):
        x_array = np.array([np.array(item[0]) for item in dataset])
        y_array = np.array([np.array(item[1]) for item in dataset])
        # # Print the shape of the subset array
        # print("x_array shape:", x_array.shape)
        # print("y_array shape:", y_array.shape)
        return x_array, y_array


class Text:

    def __init__(self, batch_size=32, num_samples=1000):
        self.batch_size = batch_size
        self.num_samples = num_samples

    def imbd(self):

        # Load the IMDB movie review dataset
        dataset = load_dataset("imdb")

        # Print information about the dataset
        # print(dataset)

        # Access the train split
        train_dataset = dataset["train"]

        # Initialize the tokenizer: Note for stats, insist that people import
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Tokenize the input text
        tokenized_inputs = tokenizer(
            train_dataset["text"],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        dataset = Dataset.from_dict(tokenized_inputs)
        dataset = dataset.add_column("labels", train_dataset["label"])

        return dataset, tokenizer

    def convert_dataset_to_torch(self, dataset):

        # Access the tokenized input and target labels
        input_ids = np.asarray(dataset["input_ids"])
        labels = np.asarray(dataset["labels"])

        return input_ids, labels


def get_fashion_mnist(batch_size=64):
    training_loader = DataLoader(
        tv_datasets.FashionMNIST(
            root='data',
            train=True,
            download=True,
            transform=ToTensor(),
        ), batch_size=batch_size, shuffle=True,
    )
    test_loader = DataLoader(
        tv_datasets.FashionMNIST(
            root='data',
            train=False,
            download=True,
            transform=ToTensor(),
        ), batch_size=batch_size, shuffle=True,
    )
    return training_loader, test_loader

# class MNISTDataModule(pl.LightningDataModule):
#     def __init__(self,*args,**kwargs):
#         pl.LightningDataModule.__init__(self,*args,**kwargs)
        
#         self.__dict__.update(kwargs)
        
#     def setup(self,stage:str="train"):
#         pass
    
    