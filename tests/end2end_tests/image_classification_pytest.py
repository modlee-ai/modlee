import torch
import os
import modlee
import pytest
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

device = torch.device('cpu')
modlee.init(api_key=os.getenv("MODLEE_API_KEY"))

def generate_dummy_data(num_samples=100, num_classes=2, img_size=(3, 32, 32)):
    X = torch.randn(num_samples, *img_size, device=device, dtype=torch.float32)  
    y = torch.randint(0, num_classes, (num_samples,), device=device, dtype=torch.long) 
    return X, y


class ModleeImageClassification(modlee.model.ImageClassificationModleeModel):
    def __init__(self, num_classes=2, img_size=(3, 32, 32)):
        super().__init__(num_classes=num_classes)
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
        print(f"Original input shape: {x.shape}, total size: {x.numel()}")
        print(f"Type of data being passed: {x.dtype}")
        
        # If the shape is `[2, 32, 3, 32, 32]`, reshape it correctly
        if x.dim() == 5 and x.size(1) == 32:
            # Reshape to collapse the second dimension (32) into the batch dimension
            shape = x.shape
            x = x.view(-1, *shape)  # Adjust to correct shape: [batch_size, channels, height, width]
            print(f"Reshaped input shape: {x.shape}")

        # Forward pass through the model
        x = self.model(x)
        
        print(f"Shape after forward pass: {x.shape}")
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        print(f"Batch x shape: {x.shape}, Batch y shape: {y.shape}")
        
        # Forward pass
        logits = self.forward(x)
        
        # Calculate loss
        loss = self.loss_fn(logits, y)
        print(f"Loss: {loss.item()}")
        
        return loss
    
    def configure_optimizers(self):
        # Make sure to refer to self.model for the optimizer
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

@pytest.mark.parametrize("img_size num_classes", [
                                        ((3, 32, 32),2)
                                    #   , ((3, 64, 64),3)
                                    #   , ((3, 128, 128),4)
                                    #   , ((1, 32, 32),4)
                                    #   , ((4, 32, 32),4)
                                    #   , ((4, 64, 64),4)
                                    #   , ((5, 32, 32),4)
                                    #   , ((5, 128, 128),4)
                                    #   , ((6, 128, 128),4)
                                    #   , ((6, 256, 256),4)
                                      ]
                                      )
def test_model_training(img_size,num_classes):
    X_train, y_train = generate_dummy_data(num_samples=100, num_classes=num_classes, img_size=img_size)
    X_test, y_test = generate_dummy_data(num_samples=20, num_classes=num_classes, img_size=img_size)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    modlee_model = ModleeImageClassification(num_classes=num_classes, img_size=img_size).to(device)


    with modlee.start_run() as run:
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(
            model=modlee_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=test_dataloader
        )

    last_run_path = modlee.last_run_path()
    artifacts_path = os.path.join(last_run_path, 'artifacts')
    artifacts = os.listdir(artifacts_path)
    assert artifacts, "No artifacts found in the artifacts path."

    print(f"Run path: {last_run_path}")
    print(f"Saved artifacts: {artifacts}")

if __name__ == "__main__":

    test_model_training((3, 32, 32),3)