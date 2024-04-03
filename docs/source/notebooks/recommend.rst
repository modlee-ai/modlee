Benchmark with model recommendations
====================================

This example notebook uses the ``modlee`` package to train a recommended
model. We will perform image classification on CIFAR10 from
``torchvision``.

First, import ``torch``- and ``modlee``-related packages.

.. code:: ipython3

    import os 
    # os.environ['MODLEE_API_KEY'] = "replace-with-your-api-key"
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

Create a ``modlee`` recommender object and fit to the dataset. This
process will calculate the dataset metafeatures to send to the server.
The server will return a recommended model for the dataset assigned to
``recommender.model``.

.. code:: ipython3

    recommender = modlee.recommender.from_modality_task(
        modality='image',
        task='classification',
        )
    recommender.fit(train_dataloader)
    modlee_model = recommender.model 
    print(f"\nRecommended model: \n{modlee_model}")

::

   INFO:Analyzing dataset based on data metafeatures...
   INFO:Finished analyzing dataset.
   INFO:The model is available at the recommender object's `model` attribute.

   Recommended model: 
   RecommendedModel(
     (model): GraphModule(
       (Conv): Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1))
       (Conv_1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
       (Relu): ReLU()
       (MaxPool): MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], ceil_mode=False)
       (Conv_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       (Relu_1): ReLU()
       (Conv_3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       (Add): OnnxBinaryMathOperation()
       (Relu_2): ReLU()
       (Conv_4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       (Relu_3): ReLU()
       (Conv_5): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
       ...

We can train the model as we would a basic ``ModleeModel``, with
automatic documentation of metafeatures.

.. code:: ipython3

    with modlee.start_run() as run:
        trainer = modlee.Trainer(max_epochs=1)
        trainer.fit(
            model=modlee_model,
            train_dataloaders=train_dataloader
        )

::

     | Name  | Type        | Params
   --------------------------------------
   0 | model | GraphModule | 11.7 M
   --------------------------------------
   11.7 M    Trainable params
   0         Non-trainable params
   11.7 M    Total params
   46.779    Total estimated model params size (MB)
   Epoch 0: 100%|██████████| 3125/3125 [01:14<00:00, 41.86it/s, v_num=0]

Finally, we can view the saved assets from training.

.. code:: ipython3

    last_run_path = modlee.last_run_path()
    print(f"Run path: {last_run_path}")
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    artifacts = os.listdir(artifacts_path)
    print(f"Saved artifacts: {artifacts}")

::

   Run path: /home/ubuntu/projects/modlee_pypi/examples/mlruns/0/7a47086681324d0e924f9076a1262de9/artifacts/model_graph.py
   Saved artifacts: ['transforms.txt', 'model_graph.py', 'model_graph.txt', 'model_size', 'model', 'cached_vars', 'stats_rep', 'snapshot_1.npy', 'lightning_logs', 'snapshot_0.npy', 'model.py', 'loss_calls.txt', 'model_summary.txt']
