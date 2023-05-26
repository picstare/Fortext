import streamlit as st
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

access_token = "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"
tokenizer = AutoTokenizer.from_pretrained('indolem/indobert-large-uncased', use_auth_token=access_token)


# Define the dataset class
class TwitterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'label': torch.tensor(label, dtype=torch.long)
        }

# Load the IndoBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-large-uncased")
model = AutoModelForSequenceClassification.from_pretrained("indolem/indobert-large-uncased", num_labels=3)

# assume that you have a pandas DataFrame containing training data
df = pd.read_csv('tdset/tsds.csv')

# split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Define the training and validation datasets
train_dataset = TwitterDataset(train_df['text'].to_numpy(), train_df['label'].to_numpy(), tokenizer, max_length=128)
val_dataset = TwitterDataset(val_df['text'].to_numpy(), val_df['label'].to_numpy(), tokenizer, max_length=128)

# Define the optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Define the data loaders
train_loader = DataLoader(train_dataset, batch_size=16, num_workers=2, sampler=RandomSampler(train_dataset))
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=2, sampler=SequentialSampler(val_dataset))

# Define the device to run the model on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train the model
epochs = 3

for epoch in range(epochs):
    model.train()
    
    for batch in tqdm(train_loader, desc=f"Training epoch {epoch}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)