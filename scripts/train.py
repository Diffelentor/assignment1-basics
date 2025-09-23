
import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import json

# Import from your custom scripts
from data import get_batch
from model import TransformerLM
from optimizer import AdamW
from serialization import save_checkpoint, load_checkpoint
from tokenizer import Tokenizer
from inference import generate
from utils import count_parameters

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

@torch.no_grad()
def evaluate(model, dataset: np.ndarray, context_length: int, batch_size: int, device: str, eval_iters: int = 100):
    """
    Evaluates the model's performance on the validation set.
    """
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        # Reshape for cross_entropy
        b, t, c = logits.shape
        logits_view = logits.view(b * t, c)
        y_view = y.view(b * t)
        loss = nn.functional.cross_entropy(logits_view, y_view)
        losses.append(loss.item())
    model.train()
    return np.mean(losses)

def main():
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")
    # Data and paths
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data file.')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data file.')
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
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--max_iters', type=int, default=5000, help='Total training iterations.')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--eval_interval', type=int, default=250, help='Evaluate every N iterations.')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed.')

    args = parser.parse_args()

    # --- Setup ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Tokenizer ---
    print("Initializing tokenizer...")
    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    special_tokens = ["<|endoftext|>"]

    if vocab_path.exists() and merges_path.exists():
        print("Loading existing tokenizer.")
        tokenizer = Tokenizer.from_files(str(vocab_path), str(merges_path), special_tokens=special_tokens)
    else:
        print("Training tokenizer from scratch...")
        tokenizer = Tokenizer.from_txt(
            input_path=args.train_data,
            vocab_size=args.vocab_size,
            special_tokens=special_tokens
        )
        # Save the trained tokenizer
        # Vocab saving
        gpt2_vocab = {
            "".join([chr(b) for b in tokenizer.vocab[i]]): i
            for i in range(tokenizer.vocab_size)
        }
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(gpt2_vocab, f, ensure_ascii=False, indent=2)
        
        # Merges saving
        with open(merges_path, "w", encoding="utf-8") as f:
            for p1, p2 in tokenizer.merges:
                f.write(f"$" + "".join([chr(b) for b in p1]) + " " + "".join([chr(b) for b in p2]) + "\n")
        print(f"Tokenizer saved to {output_dir}")

    # --- Data Loading ---
    print("Loading and tokenizing data...")
    with open(args.train_data, 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(args.val_data, 'r', encoding='utf-8') as f:
        val_text = f.read()

    train_data = np.array(tokenizer.encode(train_text), dtype=np.int64)
    val_data = np.array(tokenizer.encode(val_text), dtype=np.int64)
    print(f"Train data has {len(train_data):,} tokens.")
    print(f"Validation data has {len(val_data):,} tokens.")

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
    
    print(f"Model has {count_parameters(model):,} parameters.")

    # --- Optimizer ---
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # --- Training Loop ---
    print("Starting training...")
    best_val_loss = float('inf')
    start_time = time.time()

    for i in range(args.max_iters):
        # Get a batch of data
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        # Forward pass
        logits = model(x)
        
        # Calculate loss
        b, t, c = logits.shape
        logits_view = logits.view(b * t, c)
        y_view = y.view(b * t)
        loss = nn.functional.cross_entropy(logits_view, y_view)

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Evaluate and log
        if i % args.eval_interval == 0 or i == args.max_iters - 1:
            val_loss = evaluate(model, val_data, args.context_length, args.batch_size, device)
            duration = time.time() - start_time
            print(f"Iter {i:5d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Time: {duration:.2f}s")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = output_dir / "best_model.pt"
                print(f"New best validation loss. Saving model to {checkpoint_path}")
                save_checkpoint(model, optimizer, i, str(checkpoint_path))
    
    print("Training finished.")

    # --- Inference Demo ---
    print("\n--- Generating text from the best model ---")
    # Load the best model
    best_model_path = str(output_dir / "best_model.pt")
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

if __name__ == "__main__":
    main()
