
import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from itertools import islice
import wandb
from datetime import datetime   

# Import from your custom scripts
from scripts.data import get_batch, TokenDataset
from scripts.model import TransformerLM
from scripts.optimizer import AdamW,CosineScheduleLR
from scripts.serialization import save_checkpoint, load_checkpoint
from scripts.tokenizer import Tokenizer
from scripts.inference import generate

def initialize_weights(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    device: str
) -> dict[str, torch.Tensor]:
    """
    Initializes weights for the TransformerLM model from scratch.
    This is necessary because the provided model.py expects a weights dictionary.
    """
    def _xavier_init(tensor):
        if tensor.dim() > 1:
            nn.init.xavier_uniform_(tensor)
        else:
            nn.init.zeros_(tensor)
        return tensor

    weights = {}
    
    # Token embeddings
    weights["token_embeddings.weight"] = _xavier_init(torch.empty(vocab_size, d_model, device=device))
    
    # Transformer blocks
    for layer in range(num_layers):
        prefix = f"layers.{layer}."
        # Attention weights
        weights[prefix + "attn.q_proj.weight"] = _xavier_init(torch.empty(d_model, d_model, device=device))
        weights[prefix + "attn.k_proj.weight"] = _xavier_init(torch.empty(d_model, d_model, device=device))
        weights[prefix + "attn.v_proj.weight"] = _xavier_init(torch.empty(d_model, d_model, device=device))
        weights[prefix + "attn.output_proj.weight"] = _xavier_init(torch.empty(d_model, d_model, device=device))
        
        # RMSNorm weights
        weights[prefix + "ln1.weight"] = torch.ones(d_model, device=device)
        weights[prefix + "ln2.weight"] = torch.ones(d_model, device=device)
        
        # FFN weights
        weights[prefix + "ffn.w1.weight"] = _xavier_init(torch.empty(d_ff, d_model, device=device))
        weights[prefix + "ffn.w2.weight"] = _xavier_init(torch.empty(d_model, d_ff, device=device))
        weights[prefix + "ffn.w3.weight"] = _xavier_init(torch.empty(d_ff, d_model, device=device))

    # Final RMSNorm
    weights["ln_final.weight"] = torch.ones(d_model, device=device)
    
    # LM Head
    weights["lm_head.weight"] = _xavier_init(torch.empty(vocab_size, d_model, device=device))
    
    return weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")
    # Data and paths
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to training data file.')
    # parser.add_argument('--val_dataset', type=str, required=True, help='Path to training data file.')
    parser.add_argument('--output_dir', type=str, default='out', help='Directory to save tokenizer and model checkpoints.')
    
    # Tokenizer
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size for the tokenizer.')
    
    # Model hyperparameters
    parser.add_argument('--context_length', type=int, default=256, help='Maximum sequence length.')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers.')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads.')
    parser.add_argument('--d_ff', type=int, default=1344, help='Dimension of the feed-forward network.')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta value.')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--max_iters', type=int, default=500000, help='Total training iterations.')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluate every N iterations.')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed.')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vocab_path = os.path.join(args.dataset_path , "vocab.json")
    merges_path = os.path.join(args.dataset_path , "merges.txt")
    special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)
    
    
    # --- Model Initialization ---
    print("Initializing model...")
    model_weights = initialize_weights(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=device
    )
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        weights=model_weights
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
     
    # --- Inference Demo ---
    print("\n--- Generating text from the best model ---")
    # Load the best model
    best_model_path = os.path.join(args.output_dir , "model_best.pt")
    if os.path.exists(best_model_path):
        _ = load_checkpoint(best_model_path, model, optimizer)
        model.to(device)
        
        prompt = "Once upon a time"
        print(f"Prompt: '{prompt}'")
        generated_text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=100,
            device=device
        )
        print("\nGenerated Text:")
        print(generated_text)
    else:
        print("No best model checkpoint found to run inference.")