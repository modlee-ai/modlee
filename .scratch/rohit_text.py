#%%
import os, sys
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import modlee
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Set the API key to an environment variable,
# to simulate setting this in your shell profile
# Modlee-specific imports
import modlee
modlee.init(api_key=os.environ['MODLEE_API_KEY'])

print('Stage 1 Successful')

### Importing Dataset and creating a dataframe
file_path = './data/ecommerceDataset.csv'

# Read the dataset
df = pd.read_csv(file_path, header=None)

# Set the first row as column headers and drop the first row
df.columns = df.iloc[0]
df = df[1:]

# Reset index to keep the DataFrame clean
df.reset_index(drop=True, inplace=True)

# Display the first few rows of the DataFrame to verify changes
print(df.head())

# Convert labels to numerical values
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Ensure no NaN values in the text column and convert to string
df['Text'] = df['Text'].fillna('').astype(str)

# Assuming 'Label' is the target column and should be separated
X = df['Text']
y = df['Label']

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine X and y for DataLoader creation
train_df = pd.DataFrame({'Text': X_train, 'Label': y_train})
val_df = pd.DataFrame({'Text': X_val, 'Label': y_val})
#%%
# Tokenizer and model from transformers
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(np.unique(y)))

# Dataset class for text data
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts.iloc[index]
        label = self.labels.iloc[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Create DataLoader
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = TextDataset(
        texts=df['Text'],
        labels=df['Label'],
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)

# Adjust these parameters as necessary
MAX_LEN = 128
BATCH_SIZE = 16

train_dataloader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
val_dataloader = create_data_loader(val_df, tokenizer, MAX_LEN, BATCH_SIZE)

print('Stage 2 Successful')

# Subclass the ModleeModel class to enable automatic documentation
class ModleeClassifier(modlee.model.ModleeModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss_fn(outputs.logits, labels)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        input_ids = val_batch['input_ids']
        attention_mask = val_batch['attention_mask']
        labels = val_batch['label']
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        val_loss = self.loss_fn(outputs.logits, labels)
        return {'val_loss': val_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        return optimizer

# Create the model object
modlee_model = ModleeClassifier()

print("Stage 3 Successful")

# Create a dummy input for ONNX export
dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, MAX_LEN))
dummy_attention_mask = torch.ones(1, MAX_LEN)
dummy_input = (dummy_input_ids, dummy_attention_mask)

with modlee.start_run() as run:
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(
        model=modlee_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

print("Stage 4 Successful")

# Export the model to ONNX
torch.onnx.export(
    modlee_model,
    dummy_input,
    "model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
        "output": {0: "batch_size"}
    }
)

last_run_path = modlee.last_run_path()
print(f"Run path: {last_run_path}")

artifacts_path = os.path.join(last_run_path, 'artifacts')
artifacts = os.listdir(artifacts_path)
print(f"Saved artifacts: {artifacts}")

os.environ['ARTIFACTS_PATH'] = artifacts_path
# Add the artifacts directory to the path,
# so we can import the model
sys.path.insert(0, artifacts_path)

print("Stage 5 Successful")

