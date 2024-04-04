Dataset guidelines
==================

Here we show pseudo code to illustrate building a pytorch data loader
from a list of data elements in a format that is compatible with
**Modlee Auto Experiment Documentation**

TLDR
----

-  Define your dataset in an unnested format: [[x1, x2, x3, …, y], …]
-  Create a dataloader which is used to train a ModleeModel with a
   Modlee Trainer

Define example custom dataset objects
-------------------------------------

.. code:: ipython3

    import torch
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    
    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            feature1 = torch.tensor(self.data[idx][0], dtype=torch.float32)
            feature2 = torch.tensor(self.data[idx][1], dtype=torch.float32)
            feature3 = torch.tensor(self.data[idx][2], dtype=torch.float32)
            
            features = [feature1,feature2,feature3]  # This is a simplification
            
            target = torch.tensor(self.data[idx][-1], dtype=torch.float32).squeeze()  # Ensure target is a scalar or 1D
            
            return features, target
    
    def example_text():
        return np.random.rand(10)  # 1D array of 10 random numbers
    def example_image():
        return np.random.rand(5, 3)  # 2D array of shape (5, 3) with random numbers
    def example_video():
        return np.random.rand(5, 3, 2)  # 3D array of shape (5, 3, 2) with random numbers
    def example_target():
        return np.random.rand(1)  # scalar value


Create dataset and dataloader
-----------------------------

MODLEE_GUIDELINE
~~~~~~~~~~~~~~~~

Define your raw data so that each element is a list of data objects (any
combination of images,audio,text,video,etc …) with the final element of
the list being your target which must match the output shape of your
neural network - ex: [[x1, x2, x3, …, y], …]

Avoid nested data structures like the following - [[[x1, x2], x3, …, y],
…]

Why?
~~~~

Modlee extracts key meta features from your dataset so your experiment
can be used in aggregate analysis alongside your collaborators data, to
improve Modlee’s model recommendation technology for your connected
environment. The above stated list data structure allows us to easily
extract the information we need. Check out exactly how we do this on our
public `Github Repo <https://github.com/modlee-ai/modlee>`__.

.. code:: ipython3

    data = [[example_text(),example_image(),example_video(),example_target()] for _ in range(4)]
    
    dataset = CustomDataset(data)

Define a PyTorch DataLoader
---------------------------

MODLEE_GUIDELINE
~~~~~~~~~~~~~~~~

Pass your dataset to a PyTorch DataLoader, so that Modlee can
automatically parse it for meta features, allowing you to share it in a
meaningful way with your colleagues.

.. code:: ipython3

    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Iterate through dataloader
    for i,batch in enumerate(dataloader):
        print(f"- batch_{i}")
        features, target = batch
        for j,feature in enumerate(features):
            print(f"feature_{j}.shape = ", feature.shape)
        print("target.shape = ", target.shape)



.. parsed-literal::

    - batch_0
    feature_0.shape =  torch.Size([2, 10])
    feature_1.shape =  torch.Size([2, 5, 3])
    feature_2.shape =  torch.Size([2, 5, 3, 2])
    target.shape =  torch.Size([2])
    - batch_1
    feature_0.shape =  torch.Size([2, 10])
    feature_1.shape =  torch.Size([2, 5, 3])
    feature_2.shape =  torch.Size([2, 5, 3, 2])
    target.shape =  torch.Size([2])


Modality & task compatibility
-----------------------------

We’re working on making modlee compatible with any data modality and
machine learning task which drove us to create the above stated
MODLEE_GUIDELINES. Check out our `Github
Repo <https://github.com/modlee-ai/modlee>`__ to see which have been
tested for auto documentation to date, and if you don’t see one you
need, test it out yourself and contribute!


