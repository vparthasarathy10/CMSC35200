# Block 1: Fine-tuning the Model

# Clone the nanoGPT repo and install dependencies
!git clone https://github.com/karpathy/nanoGPT.git
%cd nanoGPT
!pip install torch numpy transformers datasets tiktoken tqdm

# Import necessary libraries
import torch
import time

# Download the Shakespeare dataset and prepare it
!cd data/shakespeare && python prepare.py

# Set the device for training (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
train_command = f"python train.py --dataset=shakespeare --n_layer=4 --n_head=4 --n_embd=64 " \
                f"--compile=False --block_size=64 --batch_size=8 --init_from=gpt2-medium " \
                f"--dtype=float16 --eval_interval=100 --eval_iters=100 --max_iters=300 --bias=True " \
                f"--device={device}"
print(f"Running training with device: {device}")
!{train_command}

# Check the output directory for the saved model checkpoint
!ls ./out/

# Block 2: Model Loading, Sparsification, Evaluation, and Plotting

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer
from model import GPTConfig, GPT
import torch

# Define paths and device
out_dir = "./out"
fine_tuned_model_path = os.path.join(out_dir, "ckpt.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

# Load the fine-tuned GPT model checkpoint
print("Loading the fine-tuned model...")
checkpoint = torch.load(fine_tuned_model_path, map_location=device)

# Create the model configuration from the checkpoint
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)

# Load the model state dictionary, handling any unwanted prefixes
state_dict = checkpoint['model']
unwanted_prefix = "_orig_mod."
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Fine-tuned model loaded successfully.")

# Define a set of diverse prompts for evaluation
prompts = [
    "Once upon a time, in a distant kingdom,",
    "The recent economic downturn has led to",
    "In the year 2050, humans have discovered",
    "Advancements in AI technology enable us to",
    "To be or not to be, that is the question."
]

# Function to calculate sparsity
def calculate_sparsity(tensor):
    non_zero = torch.count_nonzero(tensor).item()
    total_elements = tensor.numel()
    return 1 - non_zero / total_elements

# Function to prune the model
def prune_model(model, sparsity_target):
    pruned_model = GPT(gptconf)
    pruned_model.load_state_dict(model.state_dict())

    for name, param in pruned_model.named_parameters():
        if 'weight' in name:
            num_elements_to_keep = int((1 - sparsity_target) * param.numel())
            if num_elements_to_keep > 0:
                _, indices = torch.topk(param.abs().flatten(), num_elements_to_keep)
                mask = torch.zeros_like(param).flatten()
                mask[indices] = 1
                mask = mask.view(param.shape)
                param.data *= mask

    return pruned_model

# Function to evaluate the model across multiple prompts
def evaluate_model_on_prompts(model, prompts):
    total_loss = 0
    total_perplexity = 0

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs['input_ids']
        targets = input_ids.clone()

        with torch.no_grad():
            logits, loss = model(input_ids, targets)
            perplexity = torch.exp(loss)

        total_loss += loss.item()
        total_perplexity += perplexity.item()

    avg_loss = total_loss / len(prompts)
    avg_perplexity = total_perplexity / len(prompts)

    return avg_loss, avg_perplexity

# Baseline evaluation
baseline_loss, baseline_perplexity = evaluate_model_on_prompts(model, prompts)
print("Baseline (Fine-tuned) model - Average Fine-tune Loss:", baseline_loss)
print("Baseline (Fine-tuned) model - Average Zero-shot Perplexity:", baseline_perplexity)

# Define sparsity levels to test
sparsity_levels = [0.1, 0.5, 0.9, 0.95, 0.99]
results = []

# Evaluate the performance of sparsified versions of the fine-tuned model
for sparsity_level in sparsity_levels:
    pruned_model = prune_model(model, sparsity_level)
    pruned_model.to(device)
    pruned_sparsity = {name: calculate_sparsity(param) for name, param in pruned_model.named_parameters() if 'weight' in name}
    avg_sparsity = np.mean(list(pruned_sparsity.values()))

    pruned_loss, pruned_perplexity = evaluate_model_on_prompts(pruned_model, prompts)

    results.append({
        "sparsity_level": sparsity_level,
        "avg_sparsity": avg_sparsity,
        "fine_tune_loss": pruned_loss,
        "zero_shot_perplexity": pruned_perplexity
    })
    print(f"Sparsity Level {sparsity_level * 100}% - Avg Sparsity: {avg_sparsity:.4f}, "
          f"Average Fine-tune Loss: {pruned_loss:.4f}, Average Zero-shot Perplexity: {pruned_perplexity:.4f}")

# Display results in a DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Plotting results
# Fine-tune Loss vs. Sparsity Level
plt.figure(figsize=(10, 5))
plt.plot(results_df["sparsity_level"], results_df["fine_tune_loss"], marker='o', label='Fine-tune Loss')
plt.xlabel('Sparsity Level')
plt.ylabel('Average Fine-tune Loss')
plt.title('Average Fine-tune Loss at Different Sparsity Levels')
plt.grid()
plt.legend()
plt.show()

# Zero-shot Perplexity vs. Sparsity Level
plt.figure(figsize=(10, 5))
plt.plot(results_df["sparsity_level"], results_df["zero_shot_perplexity"], marker='o', color='red', label='Zero-shot Perplexity')
plt.xlabel('Sparsity Level')
plt.ylabel('Average Zero-shot Perplexity')
plt.title('Average Zero-shot Perplexity at Different Sparsity Levels')
plt.grid()
plt.legend()
plt.show()

# Sample generation to verify text coherence for sparsified models
print("Generating samples for each sparsity level...")
for sparsity_level in sparsity_levels:
    pruned_model = prune_model(model, sparsity_level)
    pruned_model.to(device)
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs['input_ids']
        with torch.no_grad():
            output = pruned_model.generate(input_ids, max_new_tokens=20)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Sparsity Level {sparsity_level * 100}% - Prompt: {prompt}")
        print("Generated Text:", generated_text)
        print('-' * 50)
