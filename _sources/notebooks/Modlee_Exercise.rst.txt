**Modlee Exercise**
===================

In this exercise, we will be assesing your ability to define a custom
neural network for optimizing performance on a dataset.

You will use the ``modlee`` package to: - Obtain your interview dataset
and solution requirements - Define custom neural networks (~10) - Use
Modlee to train neural networks and preserve your experimentation -
Submit trained neural networks for evaluation

Expectations
------------

-  Candidates typically experiment with at least **10** different model
   architectures during this exercise: defining the model, training, and
   submitting for evaluation

Tips
----

For best performance, ensure that the runtime is set to use a GPU
(``Runtime > Change runtime type > T4 GPU``).

Help & Questions
----------------

If you have any questions about the interview, please reachout on our
`Discord <https://discord.gg/dncQwFdN9m>`__ #help-exercise channel.

You can also use our
`documenation <https://docs.modlee.ai/README.html>`__ as a reference for
using our package.

.. code:: ipython3

    %%capture
    !pip install modlee torch torchvision pytorch-lightning

**Environment setup**
=====================

Step 1
------

We need to install ``modlee`` and its related packages. Make sure that
you have a Modlee account and API key `from the
dashboard <https://www.dashboard.modlee.ai/>`__.

**NOTE: if you are completing a Modlee Screening Exercise for a job
interview, make sure that you sign up for a Modlee Account with the same
email that your invite was sent to.**

Replace ``"replace-with-your-api-key"`` with your API key. Run the
following two cells; they will execute successively. This process may
take a few minutes, so you can `review the
examples <https://docs.modlee.ai/notebooks/document.html>`__ while
waiting.

.. code:: ipython3

    %load_ext autoreload
    %autoreload 2
    
    # Set your API key
    import os
    #simulate setting environment variable
    os.environ['MODLEE_API_KEY'] = "GZ4a6OoXmCXUHDJGnnGWNofsPrK0YF0i"
    api_key = os.environ['MODLEE_API_KEY']
    assert api_key != "replace-with-your-api-key", "Please update the placeholder for your Modlee API key. See above Installation instructions."


.. parsed-literal::

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


Step 2
------

Time to import our packages for this exercise.



.. code:: ipython3

    import os,zipfile,shutil,requests
    
    import torch
    from torch.utils.data import DataLoader
    import torch.nn as nn
    import torch.optim as optim
    import torch.jit as jit
    import torch.nn.functional as F
    import numpy as np
    
    import lightning.pytorch as pl
    import modlee
    from modlee.recommender import from_modality_task as trainer_for_modality_task


.. parsed-literal::

    /Users/tarushsingh/Desktop/work/Python/Modlee/modlee_pypi/venv/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


**Exercise setup**
==================

Please update ``exercise_id``, ``exercise_modality``, and
``exercise_task`` below.

If you are completing a Screening Exercise for an organization please
copy the ``exercise_id``, ``exercise_modality``, and ``exercise_task``
given to you in your invite email and paste below.

.. code:: ipython3

    exercise_id = 'TS-M4_GM_A_E_H__323565277395'
    exercise_modality = 'time_series'
    exercise_task = 'prediction'
    model_size_restriction_MB = '10'
    
    assert exercise_id != "replace-with-your-exercise-id", "Please update the placeholder for your Modlee Exercise ID. See above Installation instructions."
    assert exercise_modality != "replace-with-your-modality", "Please update the placeholder for your Modlee Exercise ID. See above Installation instructions."
    assert exercise_task != "replace-with-your-exercise-task", "Please update the placeholder for your Modlee Exercise Task. See above Installation instructions."
    assert model_size_restriction_MB != "replace-with-your-model_size_restriction_MB", "Please update the placeholder for your Modlee Exercise model_size_restriction_MB. See above Installation instructions."


**Dataset setup**
=================

**Please do not make changes to the following cell**

For the machine learning interview exercise, the dataset has already
been prepared for you. Please utilize the provided code snippet below to
configure your environment appropriately. This dataset is a carefully
curated blend of publicly available datasets, designed to assess
specific competencies in model development and performance evaluation.
As the focus of this exercise is not on data manipulation, you are
requested not to make any modifications to the dataloader settings. This
ensures that you can direct your efforts towards strategic model
building and analysis tasks.

.. code:: ipython3

    root_url = 'https://evalserver.modlee.ai:6060'
    url = f"{root_url}/get-interview-utils"  # Change the port if your Flask app is running on a different one
    response = requests.get(url, params={'api_key': api_key,'exercise_id':exercise_id})
    
    # Check if the request was successful
    if response.status_code == 200:
        with open('interview_utils.py', 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("File downloaded and saved as interview_utils.py")
    else:
        print("Failed to download file:", response.status_code)
    
    from interview_utils import *
    from interview_utils import setup,submit
    
    train_dataloader, val_dataloader, train_dataloader_shape, val_dataloader_shape = setup(api_key,exercise_id)



.. parsed-literal::

    File downloaded and saved as interview_utils.py
    File downloaded successfully: ./modlee_interview_data/modlee_interview_data.zip


.. code:: ipython3

    # DO NOT MODIFY
    input_size = train_dataloader_shape[2]
    output_size = train_dataloader_shape[2]
    output_seq_len = val_dataloader_shape[1]  # 7 time steps for prediction

**Define your models**
======================

**Critical segment of your machine learning interview**

Please define and experiment with various deep neural network
architectures. Your task is to identify and implement a model structure
that will perform optimally on the provided interview dataset. It is
essential that your model design adheres to the specified solution
requirements previously stated. This exercise aims to showcase your
ability to innovate and apply your ML knowledge effectively in
developing custom neural network based solutions.

Please follow our `Custom Model Definition
Guidelines <https://docs.modlee.ai/notebooks/model_definition_guidelines.html>`__
throughout your experimentation to ensure your submitions are evaluated
properly.

.. code:: ipython3

    '''
    ------------------------------------------------
    TODO: Make Changes Here
    ------------------------------------------------
       ExampleLSTM is defined just to get you started.
       Please experiment with many models searching for the best.
       We recommend experimenting with at least 10 different models in this exercise.
    '''
    
    class ExampleLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size, output_seq_len):
            super(ExampleLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            self.output_seq_len = output_seq_len
    
        def forward(self, x):
            # x: [batch_size, seq_len, input_size] where seq_len=10
            lstm_out, _ = self.lstm(x)
            # Initialize an empty list to store the outputs for each of the 5 future time steps
            out = []
            # Use the hidden state from the last time step to predict each future time step
            for t in range(self.output_seq_len):
                out_step = self.fc(lstm_out[:, -1, :])  # Use the last hidden state to predict each step
                out.append(out_step.unsqueeze(1))
    
            out = torch.cat(out, dim=1)  # Concatenate all time steps to form the output sequence
            return out
    
    
    '''
    -------------------------------------------------
    '''




.. parsed-literal::

    '\n-------------------------------------------------\n'



**Train your models**
=====================

**Please do not make changes to the following cell**

In this phase of your machine learning interview exercise, you are
tasked with training your previously defined model using Modlee’s
advanced training infrastructure. Our trainers, specifically the
trainer_for_modality_task instances, are equipped with robust
out-of-the-box training features such as learning rate decay, early
stopping, and more. Please utilize these settings to train your model.
This approach is not only designed to streamline the training process
but also allows us to assess your proficiency in optimizing and refining
deep learning models through architectural adjustments. This step is
crucial in demonstrating your ability to enhance model performance
within given constraints.

.. code:: ipython3

    '''
      ------------------------------------------------
      TODO: Make Changes Here
      ------------------------------------------------
      Define parameters, training loop validation step
    
    '''
    
    # JUST FOR CONTEXT - Since this is not supported in the package just yet,
    # we need to train manually, when it is supported, we will train like this
    
    """
          modlee.mlflow.end_run()
    
          trainer = trainer_for_modality_task(
              modality=exercise_modality,
              task=exercise_task,
              )
    
          trainer.dataloader = unzip_train_dataloader
          trainer.model = modlee_model
    
          trainer.train(max_epochs=10, val_dataloaders=unzip_val_dataloader)
    """
    
    # Define parameters
    hidden_size = 64
    num_layers = 2
    num_epochs = 100
    learning_rate = 0.001
    
    # Create an instance of the model
    model = ExampleLSTM(input_size, hidden_size, num_layers, output_size, output_seq_len)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(train_dataloader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
    
        # Validation
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss.item():.4f}')
        model.train()


.. parsed-literal::

    Epoch [1/100], Step [100/105], Loss: 0.5550
    Epoch [1/100], Val Loss: 2.9454
    Epoch [2/100], Step [100/105], Loss: 0.4752
    Epoch [2/100], Val Loss: 2.6373
    Epoch [3/100], Step [100/105], Loss: 0.3340
    Epoch [3/100], Val Loss: 2.5580
    Epoch [4/100], Step [100/105], Loss: 0.2816
    Epoch [4/100], Val Loss: 2.3125
    Epoch [5/100], Step [100/105], Loss: 0.2822
    Epoch [5/100], Val Loss: 2.2786
    Epoch [6/100], Step [100/105], Loss: 0.3097
    Epoch [6/100], Val Loss: 2.3103
    Epoch [7/100], Step [100/105], Loss: 0.3074
    Epoch [7/100], Val Loss: 2.5317
    Epoch [8/100], Step [100/105], Loss: 0.2729
    Epoch [8/100], Val Loss: 2.6105
    Epoch [9/100], Step [100/105], Loss: 0.2711
    Epoch [9/100], Val Loss: 2.6285
    Epoch [10/100], Step [100/105], Loss: 0.2772
    Epoch [10/100], Val Loss: 2.6405
    Epoch [11/100], Step [100/105], Loss: 0.2452
    Epoch [11/100], Val Loss: 2.7388
    Epoch [12/100], Step [100/105], Loss: 0.2896
    Epoch [12/100], Val Loss: 2.7312
    Epoch [13/100], Step [100/105], Loss: 0.2271
    Epoch [13/100], Val Loss: 2.7014
    Epoch [14/100], Step [100/105], Loss: 0.2480
    Epoch [14/100], Val Loss: 2.6967
    Epoch [15/100], Step [100/105], Loss: 0.2435
    Epoch [15/100], Val Loss: 2.7675
    Epoch [16/100], Step [100/105], Loss: 0.2585
    Epoch [16/100], Val Loss: 2.7752
    Epoch [17/100], Step [100/105], Loss: 0.1967
    Epoch [17/100], Val Loss: 2.8271
    Epoch [18/100], Step [100/105], Loss: 0.2871
    Epoch [18/100], Val Loss: 2.8210
    Epoch [19/100], Step [100/105], Loss: 0.2446
    Epoch [19/100], Val Loss: 2.7889
    Epoch [20/100], Step [100/105], Loss: 0.2468
    Epoch [20/100], Val Loss: 2.8230
    Epoch [21/100], Step [100/105], Loss: 0.2217
    Epoch [21/100], Val Loss: 2.8947
    Epoch [22/100], Step [100/105], Loss: 0.2478
    Epoch [22/100], Val Loss: 2.8540
    Epoch [23/100], Step [100/105], Loss: 0.1834
    Epoch [23/100], Val Loss: 2.8605
    Epoch [24/100], Step [100/105], Loss: 0.2072
    Epoch [24/100], Val Loss: 2.9674
    Epoch [25/100], Step [100/105], Loss: 0.2195
    Epoch [25/100], Val Loss: 2.9576
    Epoch [26/100], Step [100/105], Loss: 0.1978
    Epoch [26/100], Val Loss: 2.9416
    Epoch [27/100], Step [100/105], Loss: 0.2229
    Epoch [27/100], Val Loss: 3.0165
    Epoch [28/100], Step [100/105], Loss: 0.1985
    Epoch [28/100], Val Loss: 2.9364
    Epoch [29/100], Step [100/105], Loss: 0.2005
    Epoch [29/100], Val Loss: 2.8886
    Epoch [30/100], Step [100/105], Loss: 0.2291
    Epoch [30/100], Val Loss: 2.9716
    Epoch [31/100], Step [100/105], Loss: 0.2135
    Epoch [31/100], Val Loss: 3.0643
    Epoch [32/100], Step [100/105], Loss: 0.1961
    Epoch [32/100], Val Loss: 3.0557
    Epoch [33/100], Step [100/105], Loss: 0.2152
    Epoch [33/100], Val Loss: 3.0090
    Epoch [34/100], Step [100/105], Loss: 0.2292
    Epoch [34/100], Val Loss: 2.9589
    Epoch [35/100], Step [100/105], Loss: 0.1944
    Epoch [35/100], Val Loss: 3.0709
    Epoch [36/100], Step [100/105], Loss: 0.1860
    Epoch [36/100], Val Loss: 3.0415
    Epoch [37/100], Step [100/105], Loss: 0.1790
    Epoch [37/100], Val Loss: 3.0340
    Epoch [38/100], Step [100/105], Loss: 0.2028
    Epoch [38/100], Val Loss: 2.9581
    Epoch [39/100], Step [100/105], Loss: 0.2544
    Epoch [39/100], Val Loss: 2.9778
    Epoch [40/100], Step [100/105], Loss: 0.2133
    Epoch [40/100], Val Loss: 3.0251
    Epoch [41/100], Step [100/105], Loss: 0.2286
    Epoch [41/100], Val Loss: 3.0224
    Epoch [42/100], Step [100/105], Loss: 0.2776
    Epoch [42/100], Val Loss: 3.0671
    Epoch [43/100], Step [100/105], Loss: 0.2283
    Epoch [43/100], Val Loss: 3.0078
    Epoch [44/100], Step [100/105], Loss: 0.2175
    Epoch [44/100], Val Loss: 2.9865
    Epoch [45/100], Step [100/105], Loss: 0.2282
    Epoch [45/100], Val Loss: 3.0143
    Epoch [46/100], Step [100/105], Loss: 0.1744
    Epoch [46/100], Val Loss: 3.0430
    Epoch [47/100], Step [100/105], Loss: 0.2246
    Epoch [47/100], Val Loss: 3.0490
    Epoch [48/100], Step [100/105], Loss: 0.2155
    Epoch [48/100], Val Loss: 3.0196
    Epoch [49/100], Step [100/105], Loss: 0.2282
    Epoch [49/100], Val Loss: 3.0399
    Epoch [50/100], Step [100/105], Loss: 0.2742
    Epoch [50/100], Val Loss: 3.0428
    Epoch [51/100], Step [100/105], Loss: 0.2597
    Epoch [51/100], Val Loss: 3.0952
    Epoch [52/100], Step [100/105], Loss: 0.2356
    Epoch [52/100], Val Loss: 3.0977
    Epoch [53/100], Step [100/105], Loss: 0.2149
    Epoch [53/100], Val Loss: 3.0633
    Epoch [54/100], Step [100/105], Loss: 0.2004
    Epoch [54/100], Val Loss: 3.0569
    Epoch [55/100], Step [100/105], Loss: 0.2017
    Epoch [55/100], Val Loss: 3.1382
    Epoch [56/100], Step [100/105], Loss: 0.2279
    Epoch [56/100], Val Loss: 3.1224
    Epoch [57/100], Step [100/105], Loss: 0.1634
    Epoch [57/100], Val Loss: 3.1832
    Epoch [58/100], Step [100/105], Loss: 0.1819
    Epoch [58/100], Val Loss: 3.0691
    Epoch [59/100], Step [100/105], Loss: 0.2077
    Epoch [59/100], Val Loss: 3.0989
    Epoch [60/100], Step [100/105], Loss: 0.2611
    Epoch [60/100], Val Loss: 3.1244
    Epoch [61/100], Step [100/105], Loss: 0.2280
    Epoch [61/100], Val Loss: 3.1470
    Epoch [62/100], Step [100/105], Loss: 0.2210
    Epoch [62/100], Val Loss: 3.1471
    Epoch [63/100], Step [100/105], Loss: 0.1612
    Epoch [63/100], Val Loss: 3.1180
    Epoch [64/100], Step [100/105], Loss: 0.1810
    Epoch [64/100], Val Loss: 3.1014
    Epoch [65/100], Step [100/105], Loss: 0.2091
    Epoch [65/100], Val Loss: 3.1247
    Epoch [66/100], Step [100/105], Loss: 0.2179
    Epoch [66/100], Val Loss: 3.1365
    Epoch [67/100], Step [100/105], Loss: 0.1915
    Epoch [67/100], Val Loss: 3.1776
    Epoch [68/100], Step [100/105], Loss: 0.1709
    Epoch [68/100], Val Loss: 3.1189
    Epoch [69/100], Step [100/105], Loss: 0.2128
    Epoch [69/100], Val Loss: 3.1786
    Epoch [70/100], Step [100/105], Loss: 0.2307
    Epoch [70/100], Val Loss: 3.1153
    Epoch [71/100], Step [100/105], Loss: 0.2230
    Epoch [71/100], Val Loss: 3.0782
    Epoch [72/100], Step [100/105], Loss: 0.2456
    Epoch [72/100], Val Loss: 3.1329
    Epoch [73/100], Step [100/105], Loss: 0.2187
    Epoch [73/100], Val Loss: 3.0342
    Epoch [74/100], Step [100/105], Loss: 0.1891
    Epoch [74/100], Val Loss: 3.1082
    Epoch [75/100], Step [100/105], Loss: 0.2005
    Epoch [75/100], Val Loss: 3.0965
    Epoch [76/100], Step [100/105], Loss: 0.1799
    Epoch [76/100], Val Loss: 3.1355
    Epoch [77/100], Step [100/105], Loss: 0.2688
    Epoch [77/100], Val Loss: 3.0981
    Epoch [78/100], Step [100/105], Loss: 0.2098
    Epoch [78/100], Val Loss: 3.1766
    Epoch [79/100], Step [100/105], Loss: 0.2357
    Epoch [79/100], Val Loss: 3.0939
    Epoch [80/100], Step [100/105], Loss: 0.1836
    Epoch [80/100], Val Loss: 3.1668
    Epoch [81/100], Step [100/105], Loss: 0.1851
    Epoch [81/100], Val Loss: 3.0961
    Epoch [82/100], Step [100/105], Loss: 0.2273
    Epoch [82/100], Val Loss: 3.1469
    Epoch [83/100], Step [100/105], Loss: 0.1792
    Epoch [83/100], Val Loss: 3.1658
    Epoch [84/100], Step [100/105], Loss: 0.2357
    Epoch [84/100], Val Loss: 3.1158
    Epoch [85/100], Step [100/105], Loss: 0.2358
    Epoch [85/100], Val Loss: 3.1836
    Epoch [86/100], Step [100/105], Loss: 0.2012
    Epoch [86/100], Val Loss: 3.1276
    Epoch [87/100], Step [100/105], Loss: 0.2197
    Epoch [87/100], Val Loss: 3.1249
    Epoch [88/100], Step [100/105], Loss: 0.2392
    Epoch [88/100], Val Loss: 3.1583
    Epoch [89/100], Step [100/105], Loss: 0.2356
    Epoch [89/100], Val Loss: 3.1232
    Epoch [90/100], Step [100/105], Loss: 0.2009
    Epoch [90/100], Val Loss: 3.2113
    Epoch [91/100], Step [100/105], Loss: 0.2516
    Epoch [91/100], Val Loss: 3.1574
    Epoch [92/100], Step [100/105], Loss: 0.1560
    Epoch [92/100], Val Loss: 3.1261
    Epoch [93/100], Step [100/105], Loss: 0.1669
    Epoch [93/100], Val Loss: 3.1279
    Epoch [94/100], Step [100/105], Loss: 0.2275
    Epoch [94/100], Val Loss: 3.1300
    Epoch [95/100], Step [100/105], Loss: 0.1971
    Epoch [95/100], Val Loss: 3.1420
    Epoch [96/100], Step [100/105], Loss: 0.1913
    Epoch [96/100], Val Loss: 3.1417
    Epoch [97/100], Step [100/105], Loss: 0.2140
    Epoch [97/100], Val Loss: 3.1123
    Epoch [98/100], Step [100/105], Loss: 0.2357
    Epoch [98/100], Val Loss: 3.1433
    Epoch [99/100], Step [100/105], Loss: 0.1988
    Epoch [99/100], Val Loss: 3.1658
    Epoch [100/100], Step [100/105], Loss: 0.2057
    Epoch [100/100], Val Loss: 3.1937


**Evaluate your models**
========================

**Please do not make changes to the following cell**

Please submit your model experiment to Modlee for evaluation. We will
provide you with detailed feedback including accuracy, model size, and
additional metrics. Keep in mind that **there are no penalties for
submitting suboptimal solutions**, so feel free to submit multiple
models as needed.

.. code:: ipython3

    submit(api_key,exercise_id,model,None,modlee)
    print('As a reminder, your exercises model_size_restriction_MB is ',model_size_restriction_MB)


.. parsed-literal::

    Request was successful.
    Response: {'mean_square_error': '1.0811', 'model_size (MB)': '0.26', 'submission_id': 'TS-E-2024-08-25T19:24:48.179727_70738853129'}
    As a reminder, your exercises model_size_restriction_MB is  10


**Great work, now keep exploring more models!**
===============================================

Use the insights gained from each evaluation to iteratively refine your
model’s architecture. This process is designed to help you optimize your
solution effectively, utilizing real-world feedback to enhance your
approach.

Solution Requirements
---------------------

-  Maximize evaluation accuracy
-  Ensure your models don’t go over model size restrictions

Expectations
------------

-  Candidates typically experiment with at least **10** different model
   architectures during this exercise: defining the model, training, and
   submitting for evaluation

Evaluating your performance
---------------------------

-  We want to see your experimenation process so just work as you
   normally would
-  We will take your best model as your final submission, and do not
   penalize sub-optimal submissions. Feel free to submit as many models
   as you want to for evaluation.

