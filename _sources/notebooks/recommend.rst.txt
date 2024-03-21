Recommendation
==============

This example notebook uses the ``modlee`` package to train a recommended
model. We will perform image classification on CIFAR10 from
``torchvision``.

First, import ``torch``- and ``modlee``-related packages.

.. code:: ipython3

    import os
    os.environ['MODLEE_API_KEY'] = "replace-with-your-api-key"
    import torch, torchvision
    import torchvision.transforms as transforms
    
    import modlee
    modlee.init(api_key=os.environ.get('MODLEE_API_KEY'))

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


We can train the model as we would a basic ``ModleeModel``, with
documentation.

.. code:: ipython3

    with modlee.start_run() as run:
        trainer = modlee.Trainer(max_epochs=1)
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
