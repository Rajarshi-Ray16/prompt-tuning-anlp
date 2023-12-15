#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

import torch
from torch import nn

class CSVDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length, num_soft_prompt_tokens, subset_size = None):
        self.dataframe = pd.read_csv(filename)
        if subset_size is not None:
            self.dataframe = self.dataframe.sample(n = subset_size, random_state = 42).reset_index(drop= True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_soft_prompt_tokens = num_soft_prompt_tokens  # Number of soft prompt tokens to be added

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Assuming CSV has 'input' and 'target' columns
        article_text = self.dataframe.iloc[idx]['article']
        highlights_text = self.dataframe.iloc[idx]['highlights']

        # Encode the inputs and targets
        article_encoding = self.tokenizer.encode_plus(
            article_text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        highlights_encoding = self.tokenizer.encode_plus(
            highlights_text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Pad the labels to account for the soft prompts
        padded_labels = F.pad(
            highlights_encoding['input_ids'].flatten(), 
            pad=(self.num_soft_prompt_tokens, 0), 
            value=-100  # Use the padding token ID of your tokenizer if different
        )

        return {
            'input_ids': article_encoding['input_ids'].flatten(),
            'attention_mask': article_encoding['attention_mask'].flatten(),
            'labels': padded_labels
        }

train_path = '/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/train.csv'
max_length = 512  # or the max sequence length for your model

num_soft_prompt_tokens = 10 

# Load the tokenizer and set the pad_token
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
subset_size = 1000

dataset = CSVDataset(train_path, tokenizer, max_length,num_soft_prompt_tokens,subset_size)

# Create a DataLoader
batch_size = 5 
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

len(dataset)
validation_data = '/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/validation.csv'

validation_dataset = CSVDataset(validation_data, tokenizer, max_length, num_soft_prompt_tokens,subset_size)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

# Load the pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')

class SoftPromptEmbedding(nn.Module):
    def __init__(self, num_prompts, embedding_size):
        super(SoftPromptEmbedding, self).__init__()
        self.num_embeddings = num_prompts  # Add this line
        self.embedding = nn.Embedding(num_prompts, embedding_size)

    def forward(self, prompt_ids):
        return self.embedding(prompt_ids)


class GPT2WithSoftPrompt(GPT2LMHeadModel):
    def __init__(self, config, soft_prompt_embedding):
        super().__init__(config)
        self.soft_prompt_embedding = soft_prompt_embedding
        
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Generate a range tensor for soft prompts and repeat it for each item in the batch
        soft_prompt_ids = torch.arange(self.soft_prompt_embedding.num_embeddings, device=device).unsqueeze(0).repeat(batch_size, 1)

        # Get soft prompt embeddings
        soft_prompts = self.soft_prompt_embedding(soft_prompt_ids)

        # Concatenate the soft prompt embeddings with the original input embeddings
        # We first need to get the embeddings for the input_ids
        input_embeddings = self.transformer.wte(input_ids)

        # Concatenate embeddings
        embeddings = torch.cat((soft_prompts, input_embeddings), dim=1)

        # Adjust the attention mask for the soft prompts
        if attention_mask is not None:
            # Create an attention mask for the soft prompts
            soft_prompts_attention_mask = torch.ones(batch_size, self.soft_prompt_embedding.num_embeddings, device=device)
            # Concatenate the soft prompts attention mask with the original attention mask
            attention_mask = torch.cat((soft_prompts_attention_mask, attention_mask), dim=1)

        # Pass the concatenated embeddings and attention_mask to the GPT-2 model
        outputs = super().forward(inputs_embeds=embeddings, attention_mask=attention_mask, **kwargs)
        return outputs



# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

num_prompts = 10  
embedding_size = 768 
soft_prompt_embedding = SoftPromptEmbedding(num_prompts, embedding_size).to(model.device)
model_with_soft_prompt = GPT2WithSoftPrompt(model.config, soft_prompt_embedding)
model_with_soft_prompt.to(device)

for param in model.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(soft_prompt_embedding.parameters(), lr=0.01)
number_of_epochs = 2

for epoch in range(number_of_epochs):
    model_with_soft_prompt.train()  # Set the model to training mode
    running_loss = 0.0
    for i, batch in enumerate(dataloader):  # Enumerate the DataLoader for batch index
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device) if 'labels' in batch else None

        optimizer.zero_grad()

        # Forward pass
        outputs = model_with_soft_prompt(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Compute the loss
        loss = outputs.loss
        running_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        nn.utils.clip_grad_norm_(soft_prompt_embedding.parameters(), 1.0)
        optimizer.step()

        # Print statistics
        if (i + 1) % 10 == 0:  # Print every 10 batches
            print(f'Epoch [{epoch + 1}/{number_of_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    # Print the average loss after each epoch
    average_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{number_of_epochs}] finished with average loss: {average_loss:.4f}')

torch.cuda.empty_cache()
model_save_path = '/kaggle/working/model.pth'

torch.save(model_with_soft_prompt, model_save_path)
print(f'Entire model saved to {model_save_path}')

model_with_soft_prompt.eval()

valid_losses = []
predictions = []
actuals = []

# Disable gradient calculations
with torch.no_grad():
    for batch in validation_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model_with_soft_prompt(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Calculate the loss
        loss = outputs.loss
        valid_losses.append(loss.item())
        logits = outputs.logits
        predictions.extend(logits.argmax(dim=-1).tolist())
        actuals.extend(labels.tolist())
avg_validation_loss = sum(valid_losses) / len(valid_losses)
print(f'Validation loss: {avg_validation_loss}')