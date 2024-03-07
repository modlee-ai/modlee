Recommendation
==============

This example notebook uses the ``modlee`` package to train a recommended
model. We will perform image classification on CIFAR10 from
``torchvision``.

First, import ``torch``- and ``modlee``-related packages.

.. code:: ipython3

    import torch, torchvision
    import torchvision.transforms as transforms
    
    import modlee
    modlee.init(api_key="my-api-key")

Next, we create a dataloader from CIFAR10.

.. code:: ipython3

    transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transforms)
    val_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transforms)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
       )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16
    )


.. parsed-literal::

    /home/ubuntu/projects/.venv/lib/python3.10/site-packages/torchvision/transforms/v2/_deprecated.py:43: UserWarning: The transform `ToTensor()` is deprecated and will be removed in a future release. Instead, please use `v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])`.
      warnings.warn(


.. parsed-literal::

    Files already downloaded and verified
    Using downloaded and verified file: ./data/VOCtest_06-Nov-2007.tar
    Extracting ./data/VOCtest_06-Nov-2007.tar to ./data


Create a ``modlee`` recommender object and fit to the dataset. The
recommended ``ModleeModel`` object will be assigned to
``recommender.model``.

.. code:: ipython3

    recommender = modlee.recommender.from_modality_task(
        modality='image',
        task='classification',
        )
    recommender.fit(train_dataloader)
    modlee_model = recommender.model 



.. parsed-literal::

    [Modlee] -> Just a moment, analyzing your dataset ...
    


We can train the model as we would a basic ``ModleeModel``, with
documentation.

.. code:: ipython3

    with modlee.start_run() as run:
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(
            model=modlee_model,
            train_dataloaders=train_dataloader
        )

We can view the saved assets from training.

.. code:: ipython3

    last_run_path = modlee.last_run_path()
    print(last_run_path)
    artifacts = os.listdir(os.path.join(last_run_path, 'artifacts'))
    print(artifacts)
