
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
from scripts.utils import CrossEntropyLoss


# 生成时间戳实验名
run_name = f"experiment-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# 初始化 wandb
wandb.init(
    project="cs336_spring2025_assignment1",   # 换成你的项目名
    name=run_name
)


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

def getDataset(dataset_path, context_length, vocab_size ,split="train"): 
    text_path = dataset_path + ".txt"
    token_ids_path = os.path.join(dataset_path , "token_ids_train.npy")
    vocab_path = os.path.join(dataset_path , "vocab.json")
    merges_path = os.path.join(dataset_path , "merges.txt")
    special_tokens = ["<|endoftext|>"]
    
    text_path.replace("train",split)
    token_ids_path.replace("train",split)
    if split!="train":
        assert os.path.exists(vocab_path) and os.path.exists(merges_path), "验证集或测试集需要现有的词汇表和合并表"
    
    if os.path.exists(token_ids_path):
        print(f"✅ 已找到现有切分数据 {token_ids_path}")
        tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)
        
    elif os.path.exists(vocab_path) and os.path.exists(merges_path):
        print(f"✅ 已找到现有词汇表和合并表 {vocab_path} {merges_path}")
        tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)
        print(f"✅ 保存解码的Token_ids到 {token_ids_path}")
        tokenizer.save_token_ids_from_text_path(token_ids_path, text_path)
        
    else:
        print(f"✅ 从头开始构建词汇表和合并表 {vocab_path} {merges_path}")
        tokenizer = Tokenizer.from_txt(
            input_path=text_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )
        # Save the trained tokenizer
        print(f"✅ 保存词汇表和合并表 {vocab_path} {merges_path}")
        tokenizer.save_vocab_merges(vocab_path,merges_path)
        print(f"✅ 保存解码的Token_ids到 {token_ids_path}")
        tokenizer.save_token_ids_from_text_path(token_ids_path, text_path)
    print(f"获取数据集")
    return TokenDataset(token_ids_path, context_length),tokenizer
    

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
    parser.add_argument('--train_dataset', type=str, required=True, help='Path to training data file.')
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
    parser.add_argument('--eval_interval', type=int, default=250, help='Evaluate every N iterations.')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed.')

    args = parser.parse_args()

    # --- Setup ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.train_dataset, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset,tokenizer = getDataset(args.train_dataset, args.context_length, args.vocab_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset,tokenizer = getDataset(args.train_dataset, args.context_length, args.vocab_size, split="val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train data has {len(train_dataset)} tokens.")

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
    
    # print(f"Model has {count_parameters(model):,} parameters.")

    # --- Optimizer ---
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineScheduleLR(
        optimizer,
        max_lr=3e-4,
        min_lr=1e-5,
        warmup_iters=1000,
        cosine_cycle_iters=10000
    )
    
    # --- Training Loop ---
    print("Starting training...")
    best_val_loss = float('inf')
    start_time = time.time()

    iteration = 0
    for epoch in range(args.max_iters//len(train_dataloader) + 1):
        # Get a batch of data
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.max_iters//len(train_dataloader) + 1}", unit="iter") as trainbar:
            for x, y in trainbar:
                # Forward pass
                logits = model(x)
                
                # Calculate loss
                b, t, c = logits.shape
                logits_view = logits.view(b * t, c)
                y_view = y.view(b * t)
                loss = CrossEntropyLoss()(logits_view, y_view)

                # Backward pass and optimization
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                # 更新学习率
                scheduler.step()
                
                
                trainbar.set_postfix({"loss": f"{loss.item():.4f}"})
                iteration+=1
                
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "step": iteration
                })

                # Evaluate and log
                if iteration % args.eval_interval == 0 or iteration == args.max_iters - 1:
                    val_losses = []
                    with torch.no_grad():
                        model.eval()
                        with tqdm(islice(val_dataloader, 50), total=50, desc=f"Val {epoch+1}/{args.max_iters//len(train_dataloader) + 1}", unit="iter") as valbar:
                            for x, y in valbar:
                                # Forward pass
                                logits = model(x)
                                
                                # Calculate loss
                                b, t, c = logits.shape
                                logits_view = logits.view(b * t, c)
                                y_view = y.view(b * t)
                                val_loss = CrossEntropyLoss()(logits_view, y_view)
                                val_losses.append(val_loss.item())
                        model.train()
                    val_loss = np.mean(val_losses)
                    duration = time.time() - start_time
                    print(f"\nIter {iteration:5d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Time: {duration:.2f}s")
                    checkpoint_path = os.path.join(args.output_dir , f"model_{iteration}.pt")
                    save_checkpoint(model, optimizer, iteration, checkpoint_path)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        checkpoint_path = os.path.join(args.output_dir , "model_best.pt")
                        print(f"New best validation loss. Saving model to {checkpoint_path}")
                        save_checkpoint(model, optimizer, iteration, checkpoint_path)
    
    wandb.finish()
    print("Training finished.")

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

if __name__ == "__main__":
    main()
