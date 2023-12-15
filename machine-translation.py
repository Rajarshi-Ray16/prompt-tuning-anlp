#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import torch
from torch import nn
import random
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

def load_europarl_data(en_file, de_file):
    with open(en_file, 'r', encoding='utf-8') as file_en, open(de_file, 'r', encoding='utf-8') as file_de:
        texts_en = file_en.readlines()
        texts_de = file_de.readlines()

    # Ensure both lists have the same length
    assert len(texts_en) == len(texts_de), "Mismatch in line count between English and German files."
    
    return texts_en, texts_de

texts_en, texts_de = load_europarl_data('/kaggle/input/europart-dataset-for-task-3/de-en/europarl-v7.de-en.en', '/kaggle/input/europart-dataset-for-task-3/de-en/europarl-v7.de-en.de')

class TranslationDataset(Dataset):
    def __init__(self, texts_en, texts_de, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.texts_en = texts_en  # List of English sentences
        self.texts_de = texts_de  # List of corresponding German translations
        self.max_length = max_length

    def __len__(self):
        return len(self.texts_en)

    def __getitem__(self, idx):
        source_text = f"[TRANSLATE EN DE] {self.texts_en[idx]}"
        target_text = self.texts_de[idx]

        # Tokenize source and target texts
        source_encoding = self.tokenizer(
            source_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            target_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt"
        )

        return {
            'input_ids': source_encoding['input_ids'].flatten(),
            'attention_mask': source_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

# Split the data into training and validation sets
train_texts_en, val_texts_en, train_texts_de, val_texts_de = train_test_split(
    texts_en, texts_de, test_size=0.1  # 10% for validation
)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.pad_token = tokenizer.eos_token

# Add special tokens and resize token embeddings in the model
special_tokens_dict = {'additional_special_tokens': ['[TRANSLATE EN DE]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
# model.resize_token_embeddings(len(tokenizer))

# Create datasets
train_dataset = TranslationDataset(train_texts_en, train_texts_de, tokenizer, max_length=512)
val_dataset = TranslationDataset(val_texts_en, val_texts_de, tokenizer, max_length=512)
batch_size = 8

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size= batch_size)


model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Move model to GPU
model.to(device)
# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)
number_of_epochs = 1

for epoch in range(number_of_epochs):
    model.train()
    total_train_loss = 0

    for batch in train_dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_train_loss = total_train_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{number_of_epochs}, Training Loss: {average_train_loss:.4f}')

# Validation step
model.eval()
total_val_loss = 0
with torch.no_grad():
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_val_loss += loss.item()

average_val_loss = total_val_loss / len(val_dataloader)
print(f'Epoch {epoch + 1}/{number_of_epochs}, Validation Loss: {average_val_loss:.4f}')