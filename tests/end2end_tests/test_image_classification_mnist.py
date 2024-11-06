import torch
import os
import modlee
import pytest
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from utils import check_artifacts, get_device

import torchvision.transforms as transforms
from torchvision.datasets import MNIST

device = get_device()

modlee.init(api_key=os.getenv("MODLEE_API_KEY"), run_path= '/home/ubuntu/efs/modlee_pypi_testruns')

class ModleeImageClassification(modlee.model.ImageClassificationModleeModel):
    def __init__(self, num_classes=2, img_size=(3, 32, 32)):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        input_dim = img_size[0] * img_size[1] * img_size[2]  # 3 * 32 * 32 = 3072
        
        # Define the model as a sequential set of layers
        self.model = nn.Sequential(
            nn.Flatten(),  # Flatten [batch_size, 3, 32, 32] -> [batch_size, 3072]
            nn.Linear(input_dim, 128),  # Linear layer with input 3072, output 128
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Output layer with input 128, output num_classes
        )
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # print(f"Original input shape: {x.shape}, total size: {x.numel()}")
        # print(f"Type of data being passed: {x.dtype}")
        
        # If the shape is `[2, 32, 3, 32, 32]`, reshape it correctly
        if x.dim() == 5 and x.size(1) == 32:
            # Reshape to collapse the second dimension (32) into the batch dimension
            shape = x.shape
            x = x.view(-1, *shape)  # Adjust to correct shape: [batch_size, channels, height, width]
            # print(f"Reshaped input shape: {x.shape}")

        # Forward pass through the model
        x = self.model(x)
        
        # print(f"Shape after forward pass: {x.shape}")
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # print(f"Batch x shape: {x.shape}, Batch y shape: {y.shape}")
        # Forward pass
        logits = self.forward(x)
        # Calculate los
        loss = self.loss_fn(logits, y)
        # self.log('loss', loss, on_epoch=True, prog_bar=True)

        # print(f"Loss: {loss.item()}")
        
        return {'loss':loss}
    
    def validation_step(self, val_batch):
        x, y_target = val_batch  # Get validation data and targets
        y_pred = self.forward(x)  # Model predictions
        val_loss = self.loss_fn(y_pred, y_target)  # Calculate the validation loss
        # self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        return {'val_loss': val_loss}
    
    def configure_optimizers(self):
        # Make sure to refer to self.model for the optimizer
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

img_size_list = [(3, 32, 32),(1, 64, 64)]
recommended_model_list = [True,False]
modlee_trainer_list = [True,False]

@pytest.mark.parametrize("img_size", img_size_list)
@pytest.mark.parametrize("recommended_model", recommended_model_list)
@pytest.mark.parametrize("modlee_trainer", modlee_trainer_list)
def test_image_classifer(img_size,recommended_model,modlee_trainer):

    transform = transforms.Compose([
        transforms.Resize((img_size[1], img_size[2])),
        transforms.Grayscale(num_output_channels=img_size[0]),  # Convert images to RGB format
        transforms.ToTensor(),          # Convert images to tensors (PyTorch format)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images with mean and std deviation
    ])

    train_dataset = MNIST( #this command gets the MNIST images
        root='./data',
        train=True, #loading the training split of the dataset
        download=True,
        transform=transform) #applies transformations defined earlier

    val_dataset = MNIST(
        root='./data',
        train=False, #loading the validation split of the dataset
        download=True,
        transform=transform)

    num_classes = 10#Hardcoded to match MNIST

    train_dataloader = DataLoader( #this tool loads the data
        train_dataset,
        batch_size=32, #we will load the images in groups of 4
        shuffle=True)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32)


    if recommended_model == True:
        recommender = modlee.recommender.ImageClassificationRecommender(num_classes=num_classes)
        recommender.fit(train_dataloader)
        modlee_model = recommender.model.to(device)
        print(f"\nRecommended model: \n{modlee_model}")
    else:
        modlee_model = ModleeImageClassification(num_classes=num_classes, img_size=img_size).to(device)
    
    # modlee_trainer = True
    if modlee_trainer == True:
        trainer = modlee.model.trainer.AutoTrainer(max_epochs=5)
        trainer.fit(
            model=modlee_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
    else:
        with modlee.start_run() as run:
            trainer = pl.Trainer(max_epochs=10)
            trainer.fit(
                model=modlee_model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )


    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    check_artifacts(artifacts_path)


if __name__ == "__main__":

    test_image_classifer((3, 32, 32),True,True)