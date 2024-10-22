import json
import random
import requests

# Example gene list
genes = [
    "BRCA1", "TP53", "EGFR", "VEGFA", "APOE", "TNF", "IL6", "MTHFR", "PTEN", 
    "KRAS", "ESR1", "ABL1", "MYC", "CDKN2A", "FGFR3", "PIK3CA", "NOTCH1", "FLT3", 
    "IDH1", "SMAD4", "NTRK1", "ALK", "ROS1", "RET", "BRAF"
]

# Randomly select 20 genes from the pool
selected_genes = random.sample(genes, 20)

# Function to query LM Studio's local model API with a specific model id
def model_query_function(prompt, model_id):
    url = "http://127.0.0.1:1234/v1/completions"  # Localhost URL for LM Studio
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer lm-studio",  # Default API key for LM Studio
    }
    payload = {
        "model": model_id,  # Specify the model's id
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 100  # I adjust this as needed, 100 seems pretty good
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    result = response.json()["choices"][0]["text"].strip() 
    return result

# Function to query disease association using the Gemma model
def query_disease_association_gemma(gene):
    prompt = f"In a brief sentence, is the gene {gene} associated with cancer, heart disease, diabetes, or dementia? Provide a concise and brief explanation."
    response = model_query_function(prompt, "gemma-2-2b-instruct")  # Use Gemma model's id
    return response

# Function to query relationships between all selected genes using the Gemma model
def query_gene_relationships_gemma(genes):
    gene_list = ", ".join(genes)
    prompt = f"Describe any significant interactions or links between the following genes: {gene_list}."
    response = model_query_function(prompt, "gemma-2-2b-instruct")  # Use Gemma model's id
    return response

# Function to query disease association using the Llama model
def query_disease_association_llama(gene):
    prompt = f"In a brief sentence, is the gene {gene} associated with cancer, heart disease, diabetes, or dementia? Provide a concise and brief explanation."
    response = model_query_function(prompt, "llama-3.2-3b-instruct")  # Use Llama model's id
    return response

# Function to query relationships between all selected genes using the Llama model
def query_gene_relationships_llama(genes):
    gene_list = ", ".join(genes)
    prompt = f"Describe any significant interactions or links between the following genes: {gene_list}."
    response = model_query_function(prompt, "llama-3.2-3b-instruct")  # Use Llama model's id
    return response

# Step 1: Using the Gemma model to check disease associations for selected genes
gene_disease_connections_gemma = {}
for gene in selected_genes:
    response = query_disease_association_gemma(gene)
    gene_disease_connections_gemma[gene] = response

# Step 2: Using the Llama model to check disease associations for selected genes
gene_disease_connections_llama = {}
for gene in selected_genes:
    response = query_disease_association_llama(gene)
    gene_disease_connections_llama[gene] = response

# Step 3: Using the Gemma model to check relationships for selected genes (all interactions queried at once)
relationship_response_gemma = query_gene_relationships_gemma(selected_genes)

# Step 4: Using the Llama model to check relationships for selected genes (all interactions queried at once)
relationship_response_llama = query_gene_relationships_llama(selected_genes)

# Output disease association results from Gemma
print("Gene-Disease Connections (Gemma):")
for gene, association in gene_disease_connections_gemma.items():
    print(f"{gene}: {association}")

# Output disease association results from Llama
print("\nGene-Disease Connections (Llama):")
for gene, association in gene_disease_connections_llama.items():
    print(f"{gene}: {association}")

# Output gene-gene interactions from Gemma
print("\nGene-Gene Relationships (Gemma):")
print(relationship_response_gemma)

# Output gene-gene interactions from Llama
print("\nGene-Gene Relationships (Llama):")
print(relationship_response_llama)
