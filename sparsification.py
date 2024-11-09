
# Clone the nanoGPT repo and install dependencies
!git clone https://github.com/karpathy/nanoGPT.git
%cd nanoGPT
!pip install torch numpy transformers datasets tiktoken tqdm

# Download Shakespeare dataset and prepare it
!cd data/shakespeare && python prepare.py

# Import necessary libraries
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
# Load pre-trained GPT-2 medium model and tokenizer from Hugging Face
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model.eval()
print("Loaded pre-trained GPT-2 medium model.")

# Verify the model and tokenizer are loaded
print("GPT-2 model and tokenizer successfully loaded.")

# Define sparsity calculation and pruning functions
def calculate_sparsity(tensor):
    non_zero = torch.count_nonzero(tensor).item()
    total_elements = tensor.numel()
    return 1 - non_zero / total_elements

# Revised prune_model function to avoid memory issues with large tensors
def prune_model(model, sparsity_target):
    pruned_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')  # Initialize a new model with the same weights
    pruned_model.load_state_dict(model.state_dict())  # Copy weights

    for name, param in pruned_model.named_parameters():
        if 'weight' in name:
            # Determine the number of elements to keep based on sparsity_target
            num_elements_to_keep = int((1 - sparsity_target) * param.numel())
            
            # Flatten the tensor to rank by absolute value, then create a mask to zero out others
            if num_elements_to_keep > 0:
                _, indices = torch.topk(param.abs().flatten(), num_elements_to_keep)
                mask = torch.zeros_like(param).flatten()
                mask[indices] = 1
                mask = mask.view(param.shape)
                
                # Apply the mask to zero out parameters below the threshold
                param.data *= mask

    return pruned_model

# Define functions for evaluation
def evaluate_perplexity(model, text="To be or not to be, that is the question."):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs['input_ids']
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss)
    return perplexity.item()

def fine_tune_loss(model, text="To be or not to be, that is the question."):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs['input_ids']
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return loss.item()

# Baseline evaluation before pruning
baseline_perplexity = evaluate_perplexity(model)
baseline_loss = fine_tune_loss(model)
print("Baseline model - Fine-tune Loss:", baseline_loss)
print("Baseline model - Zero-shot Perplexity:", baseline_perplexity)

# Define sparsity levels to test
sparsity_levels = [0.1, 0.5, 0.9, 0.95, 0.99]
results = []

# Evaluate the model performance at different sparsity levels
for sparsity_level in sparsity_levels:
    # Prune model
    pruned_model = prune_model(model, sparsity_level)
    
    # Calculate average sparsity
    pruned_sparsity = {name: calculate_sparsity(param) for name, param in pruned_model.named_parameters() if 'weight' in name}
    avg_sparsity = np.mean(list(pruned_sparsity.values()))
    
    # Evaluate fine-tune loss and zero-shot perplexity
    pruned_loss = fine_tune_loss(pruned_model)
    pruned_perplexity = evaluate_perplexity(pruned_model)
    
    # Store results
    results.append({
        "sparsity_level": sparsity_level,
        "avg_sparsity": avg_sparsity,
        "fine_tune_loss": pruned_loss,
        "zero_shot_perplexity": pruned_perplexity
    })
    print(f"Sparsity Level {sparsity_level * 100}% - Avg Sparsity: {avg_sparsity:.4f}, Fine-tune Loss: {pruned_loss:.4f}, Zero-shot Perplexity: {pruned_perplexity:.4f}")

# Display results in DataFrame
import pandas as pd
results_df = pd.DataFrame(results)
print(results_df)

# Plot results for visual analysis
import matplotlib.pyplot as plt

# Fine-tune Loss vs. Sparsity Level
plt.figure(figsize=(10, 5))
plt.plot(results_df["sparsity_level"], results_df["fine_tune_loss"], marker='o', label='Fine-tune Loss')
plt.xlabel('Sparsity Level')
plt.ylabel('Fine-tune Loss')
plt.title('Fine-tune Loss at Different Sparsity Levels')
plt.grid()
plt.legend()
plt.show()

# Zero-shot Perplexity vs. Sparsity Level
plt.figure(figsize=(10, 5))
plt.plot(results_df["sparsity_level"], results_df["zero_shot_perplexity"], marker='o', color='red', label='Zero-shot Perplexity')
plt.xlabel('Sparsity Level')
plt.ylabel('Zero-shot Perplexity')
plt.title('Zero-shot Perplexity at Different Sparsity Levels')
plt.grid()
plt.legend()
plt.show()
