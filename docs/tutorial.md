![](https://github.com/mansiagr4/gifs/raw/main/new_small_logo.svg)

# First Project with Modlee

In this tutorial, we'll walk through a complete machine learning project using the Modlee package. 

## MNIST Image Classification with Modlee: An End-to-End Tutorial

We’ll walk through an end-to-end project using the Modlee package for image classification. We’ll use the MNIST dataset to demonstrate how to:

1. Use the Modlee recommender to get a recommended model.
2. Train and evaluate the recommended model on the MNIST dataset.
3. Implement a custom model, train, and evaluate it.
4. Compare the performance of the Modlee-recommended model with our custom model.

### MNIST Dataset

The MNIST dataset is a well-known benchmark in the field of machine learning. It consists of:

- **Images**: 28x28 grayscale images of handwritten digits (0 through 9).
- **Labels**: Corresponding labels for each image indicating the digit.
  
MNIST is used to test and compare classification algorithms. In this tutorial, we'll use it to evaluate the performance of models recommended by Modlee and a custom-built model.

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/modlee/modlee-mnist-image-classification-example)

### Prerequisites

Before starting, ensure you have the `modlee` package installed. You can install it using `pip`:

```shell
pip install modlee
```

### 1. Using Modlee to Recommend and Train a Model

We'll start by loading the MNIST dataset, using Modlee to recommend a model, and then training and evaluating it.

a. **Import Required Libraries**
  
    First, we import the necessary libraries.

    ```python
    import os
    import lightning.pytorch as pl
    import torchvision.transforms as transforms
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader
    import torch
    import modlee
    ```

b. **Initialize Modlee**
  
    We initialize the Modlee package with your API key.

    ```python
    os.environ['MODLEE_API_KEY'] = "replace-with-your-api-key"
    modlee.init(api_key=os.environ.get('MODLEE_API_KEY'))
    ```

c. **Define Data Transformations**

    We preprocess the images to be compatible with our models.

    ```python
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.Grayscale(num_output_channels=3),  
    transforms.ToTensor(),    
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])  
    ```

    *What We Are Doing*:
    We prepare our images for the model by resizing them to 224x224 pixels, which ensures consistent input size. We then convert the images into tensors for processing with `PyTorch`. Finally, we normalize the pixel values to a range of -1 to 1, which helps the model learn better by standardizing the data and reducing bias.

    *Why We Are Doing It*:
    We resize images to ensure they are all the same size for consistent input to the model. Converting them to tensors allows `PyTorch` to process the data, and normalizing pixel values helps the model learn effectively by standardizing the data range and reducing bias.

d. **Load the MNIST Dataset**
  
    We load the training and validation datasets.

    ```python
    train_dataset = MNIST( 
    root='./data',
    train=True, 
    download=True,
    transform=transform) 

    val_dataset = MNIST(
    root='./data',
    train=False, 
    download=True,
    transform=transform)
    ```

    *What We Are Doing*:
    We are loading the MNIST dataset by specifying the directory to store the data and setting whether we want the training or validation data. We also ensure that the dataset is downloaded if it's not already present and apply the previously defined transformations.

    *Why We Are Doing It*:
    Loading the dataset with transformations prepares the images for the model by resizing and normalizing them, ensuring that the data is ready for training and evaluation. This helps in standardizing the data input, which is crucial for effective model performance.

e. **Create DataLoaders**
  
    We create DataLoaders to handle mini-batch loading.

    ```python
    train_loader = DataLoader( 
        train_dataset,
        batch_size=4, 
        shuffle=True)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=4)

    ```

    *What We Are Doing*:
    We are creating `DataLoaders` for the training and validation datasets. The `DataLoader` manages how the data is batched and shuffled during training. We set the batch size to 4 samples per batch and enabled shuffling for the training data to improve model performance.

    *Why We Are Doing It*:
    Using `DataLoaders` ensures that data is processed in manageable chunks (mini-batches) and shuffled during training, which helps the model learn more effectively by exposing it to varied data in each training iteration.

f. **Initialize the Modlee Recommender**

    We use Modlee to get a recommended model for image classification.

    ```python
    recommender = modlee.recommender.ImageClassificationRecommender(
        num_classes=10  # MNIST has 10 classes (digits 0 to 9)
    )
    ```

    *What We Are Doing*:
    We are initializing the Modlee recommender to obtain a recommended model for image classification. By specifying the number of classes, we use Modlee to select a suitable model for our needs.

    *Why We Are Doing It*:
    Using Modlee's recommender simplifies the process of choosing a model by automatically selecting one that is well-suited for image classification tasks, saving time and ensuring a good starting point for our project.

g. **Fit the Recommender on Training Data**

    We fit the recommender on the training data to get the best model.

    ```python
    recommender.fit(train_loader)
    ```

    *What We Are Doing*:
    We are training the recommended model by fitting the recommender on our training data using the `fit` method.

    *Why We Are Doing It*:
    Training the model on the training data allows it to learn and adapt to the specific patterns in the data, ensuring it performs well on the task of image classification.

h. **Get and Print the Recommended Model**

    We get the model recommended by Modlee and print it.

    ```python
    modlee_model = recommender.model
    print(f"\nRecommended model: \n{modlee_model}")
    ```

    The `recommender.model` function retrieves the model recommended by Modlee.

i. **Train the Model**

    We train the recommended model using PyTorch Lightning.

    ```python
    with modlee.start_run() as run:
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(
            model=modlee_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_dataloader
        )
    ```

    *What We Are Doing*:
    We are training the recommended model using `PyTorch Lightning`. We start a new run for tracking, configure a trainer to manage the training process, and fit the model on both the training and validation datasets.

    *Why We Are Doing It*:
    Training the model with `PyTorch Lightning` simplifies and organizes the process, while tracking the run helps monitor performance and progress. Setting the number of epochs determines how long the model will train, ensuring it learns effectively from the data.

j. **Evaluate the Model**

    We evaluate the trained model on the validation set.

    ```python
    trainer.validate(model=modlee_model, dataloaders=val_dataloader)
    ```

    *What We Are Doing*: 
    We are evaluating the custom model on the validation set using the `validate` method of the trainer.

    *Why We Are Doing It*: 
    Running validation helps us assess how well the model performs on unseen data, providing insights into its accuracy and generalization. This step is crucial for understanding the model’s effectiveness and identifying any areas for improvement.

### 2. Custom Model Implementation

Now, we'll define a custom CNN model with Modlee's framework, train it, and evaluate its performance.

a. **Define the Custom Model**

    We define a custom Convolutional Neural Network (CNN) with Modlee.

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    import modlee

    # Subclass the ImageClassificationModleeModel class to enable automatic documentation
    class SimpleCNNModleeModel(modlee.model.ImageClassificationModleeModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Define model architecture 
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),  # First convolutional layer
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
                nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Second convolutional layer
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling
                nn.Flatten(),  # Flatten the tensor for fully connected layers
                nn.Linear(64 * 56 * 56, 128),  # Fully connected layer
                nn.ReLU(),
                nn.Linear(128, 10)  # Output layer with 10 classes
            )
            # Define the loss function as cross-entropy loss
            self.loss_fn = F.cross_entropy

        def forward(self, x):
            # Forward pass through the CNN model
            return self.model(x)

        # Define the training step
        def training_step(self, batch, batch_idx):
            x, y_target = batch  # Get input data and targets
            y_pred = self(x)  # Model predictions
            loss = self.loss_fn(y_pred, y_target)  # Calculate the loss
            return {"loss": loss}

        # Define the validation step
        def validation_step(self, val_batch, batch_idx):
            x, y_target = val_batch  # Get validation data and targets
            y_pred = self(x)  # Model predictions
            val_loss = self.loss_fn(y_pred, y_target)  # Calculate the validation loss
            # Calculate accuracy
            acc = torch.sum(torch.argmax(y_pred, dim=1) == y_target).float() / y_target.size(0)
            return {'val_loss': val_loss, 'val_acc': acc}

        # Set up the optimizer for training
        def configure_optimizers(self):
            optimizer = Adam(self.parameters(), lr=0.001)  # Adam optimizer
            return optimizer

    ```

    *What We Are Doing*:
    We are defining a custom Convolutional Neural Network (CNN) model. This model includes convolutional layers to extract features from images, followed by fully connected layers for classification.

    *Why We Are Doing It*:
    Defining a custom CNN allows us to tailor the architecture specifically for our task, in this case, classifying MNIST images. The convolutional layers help in extracting important features from the images, while the fully connected layers perform the final classification, enabling the model to accurately predict the digits.

b. **Create DataLoaders**

    We prepare DataLoaders for the custom model.

    ```python
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4)
    ```

    *What We Are Doing*:
    We are creating `DataLoaders` for our custom model to manage how data is batched and shuffled during training and validation.

    *Why We Are Doing It*:
    `DataLoaders` help process the dataset in manageable batches and shuffle the training data, which enhances model performance by providing varied data each epoch and speeding up the training process.

c. **Train the Custom Model**

    We train the custom model using PyTorch Lightning.

    ```python
    # Create an instance of the SimpleCNNModleeModel model
    modlee_model = SimpleCNNModleeModel()

    # Start the training process
    with modlee.start_run() as run:
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit( 
            model=modlee_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_dataloader
        )
    ```

    *What We Are Doing*:
    We are training the custom CNN model using `PyTorch Lightning`. We initialize the `SimpleCNNModleeModel` model, then configure a trainer to handle the training and validation processes, setting it to run for one epoch.

    *Why We Are Doing It*:
    PyTorch Lightning's `Trainer` simplifies the training and validation workflow, automating many of the repetitive tasks. This setup ensures our model is trained efficiently and allows for easy monitoring of performance across epochs.

d. **Evaluate the Custom Model**

    We evaluate the custom model on the validation set.

    ```python
    trainer.validate(model=modlee_model, dataloaders=val_dataloader)
    ```

    *What We Are Doing*:
    We are evaluating the custom model on the validation set using the `validate` method of the trainer.

    *Why We Are Doing It*:
    Running validation helps us assess how well the model performs on unseen data, providing insights into its accuracy and generalization. This step is crucial for understanding the model's effectiveness and identifying any areas for improvement.

### 3. Compare Models

Finally, compare the performance of the Modlee recommended model with the custom model by examining their accuracy on the test set.

### 4. View Saved Training Assets

We can view the saved assets from training. With Modlee, your training assets are automatically saved, preserving valuable insights for future reference and collaboration.

```python
last_run_path = modlee.last_run_path()
print(f"Run path: {last_run_path}")
artifacts_path = os.path.join(last_run_path, 'artifacts')
artifacts = sorted(os.listdir(artifacts_path))
print(f"Saved artifacts: {artifacts}")
```

### Conclusion

We have successfully walked through a complete machine learning project using the Modlee package for image classification. We demonstrated how to:

- Use Modlee to recommend and train a model for MNIST image classification.
- Implement and train a custom CNN model.
- Evaluate and compare the performance of both models.
  
By following these steps, you should now have a solid understanding of how to leverage Modlee for model recommendation and how to build and train custom models. The comparison between the recommended and custom models will help you understand the strengths and weaknesses of each approach.

### Recommended Next Steps

To build on your progress, consider these next steps:

1. **[Check Out the Guides](https://docs.modlee.ai/guides.html)**: Explore Modlee's detailed guides to gain deeper insights into advanced features and functionalities. These guides offer step-by-step instructions and practical examples to enhance your understanding.

2. **[Review Examples](https://docs.modlee.ai/notebooks/recommend.html)**: Look through our collection of examples to see Modlee in action across various tasks. These examples can inspire and help you apply Modlee to your projects effectively.

3. **Experiment with Your Projects**: Use the knowledge you've gained to experiment with Modlee on new datasets and challenges. This will help you refine your skills and develop innovative solutions.

4. **[Engage with the Community](https://docs.modlee.ai/support.html)**: Join discussions and forums to connect with other users, seek advice, and share your experiences.