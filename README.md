Using the method of prompt tuning to finetune for various NLP tasks.

## Running the files

### Summarisation
    summarization.py
### Question-Answering
    question-answering.py
### Machine Translation
    machine_translation.py

# Analysis
## Hyperparameters Used:
1. `batch_size`: 2 or 4 or 8. CUDA goes out of memory after this. **Gradient accumulation** has been used.
2. `num_epochs`: 10. However, **early stopping** has also been used.
3. `gradient clipping`: Gradients have been clipped with a norm of 1.0 to prevent exploding gradients.
4. `learning_rate`: 0.01
5. `optimiser`: AdamW , 
6. `metric`: Bleu score

## Results achieved
### 1. Summarisation
- Best Loss achieved: 3.2
### 2. Question Answering
- Best Loss achieved: 1.32
- Bleu score on test set: 0.03

### 3. Machine Translation

- Best Loss achieved: 4.56  
- Bleu score on test set: 

---
