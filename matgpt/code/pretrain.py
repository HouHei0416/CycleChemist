import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Resolve important project paths relative to this file so that data and checkpoints
# are found correctly no matter where the script is launched from.
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CODE_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Setup paths to avoid namespace conflicts
_MATGPT_MODEL_DIR = os.path.join(CODE_DIR, 'model')
sys.path.insert(0, _MATGPT_MODEL_DIR)
from MatGPT import MatGPT, MatGPTConfig
sys.path.pop(0)

from vocabulary import SMILESTokenizer, read_vocabulary, create_vocabulary, save_vocabulary
from dataset import Dataset
from utils import model_validity, set_seed, filter_valid_smiles, freeze_parameters, make_collate_fn

def get_args():
    """Parse command line arguments (config file + lightweight overrides)."""
    parser = argparse.ArgumentParser(description="MatGPT pretraining")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional override for run_name in the config.",
    )

    return parser.parse_args()

def get_lr(it, total_it, config):
    """Calculate learning rate with warmup and cosine decay."""
    warmup_iters = config["warmup"] * total_it
    if it < warmup_iters:
        lr_mult = it / warmup_iters
    else:
        decay_ratio = (it - warmup_iters) / (total_it - warmup_iters)
        lr_mult = max(0.1, 0.5 * (1.0 + np.cos(np.pi * decay_ratio)))
    return config["learning_rate"] * lr_mult

def main():
    # Load configuration from JSON and apply any CLI overrides
    args = get_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    if args.run_name is not None:
        config["run_name"] = args.run_name

    # Setup directories
    ckpt_base_dir = os.path.join(PROJECT_ROOT, config["ckpt_save_path"])
    ckpt_dir = os.path.join(ckpt_base_dir, config["run_name"])
    os.makedirs(ckpt_dir, exist_ok=True)
    
    writer = None
    if config.get("use_tensorboard", False):
        log_dir = os.path.join(PROJECT_ROOT, "matgpt", "runs", config["run_name"])
        writer = SummaryWriter(log_dir=log_dir)
    
    # Save config
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Set device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    set_seed(42)
    
    # Load dataset
    dataset_name = config["dataset"]
    data_df = pd.read_csv(os.path.join(DATA_DIR, f"{dataset_name}.csv"))

    # Determine SMILES column(s)
    if "SMILES" in data_df.columns:
        data = data_df["SMILES"]
    elif "smiles" in data_df.columns:
        data = data_df["smiles"]
    elif "Donor SMILES" in data_df.columns or "Acceptor SMILES" in data_df.columns:
        donor_series = (
            data_df["Donor SMILES"].dropna()
            if "Donor SMILES" in data_df.columns
            else pd.Series([], dtype=str)
        )
        acceptor_series = (
            data_df["Acceptor SMILES"].dropna()
            if "Acceptor SMILES" in data_df.columns
            else pd.Series([], dtype=str)
        )
        data = pd.concat([donor_series, acceptor_series], ignore_index=True)
    else:
        raise KeyError(
            "Could not find a SMILES column. Expected one of "
            "'SMILES', 'smiles', or 'Donor SMILES'/'Acceptor SMILES'."
        )
    
    print("Total data samples:", len(data))
    
    # Load or create vocabulary
    if config["vocab_path"]:
        voc = read_vocabulary(config["vocab_path"])
        print("Read vocabulary from:", config["vocab_path"])
    else:
        voc = create_vocabulary(data, tokenizer=SMILESTokenizer())
        print("Created vocabulary from dataset.")
        # Save vocabulary to data directory
        vocab_filename = f"vocab_{dataset_name}.txt"
        vocab_path = os.path.join(DATA_DIR, vocab_filename)
        save_vocabulary(voc, vocab_path)
        print(f"Saved vocabulary to: {vocab_path}")
    
    # Split data
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]
    
    train_dataset = Dataset(train_data, voc, SMILESTokenizer(), aug_prob=config["aug_prob"])
    val_dataset = Dataset(val_data, voc, SMILESTokenizer())
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, 
        pin_memory=False, collate_fn=make_collate_fn(config["block_size"])
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, 
        pin_memory=False, collate_fn=make_collate_fn(config["block_size"])
    )
    
    # Create model
    diversity_params = {
        'use_diversity_loss', 'diversity_weight', 'use_nucleus_sampling', 
        'nucleus_p', 'use_temperature_annealing', 'min_temperature', 'max_temperature'
    }
    model_config = MatGPTConfig(
        vocab_size=len(voc),
        **{k: v for k, v in config.items() if k in [
            'n_layer', 'n_head', 'n_embd', 'block_size',
            'use_rotary', 'use_rel_pos_bias', 'use_gated_mlp'] or k in diversity_params}
    )
    model = MatGPT(model_config).to(device)
    
    # Load checkpoint
    if config["ckpt_load_path"] and config["continued_pretrain"]:
        model.load_state_dict(torch.load(config["ckpt_load_path"]), strict=True)
        print("Loaded checkpoint from:", config["ckpt_load_path"])

    if config["continued_pretrain"] and config.get("freeze_layers", 0) > 0:
        freeze_parameters(model, config)

    # Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=config["weight_decay"], 
        learning_rate=config["learning_rate"], 
        betas=(0.9, 0.95)
    )
    
    # AMP
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else torch.amp.GradScaler()
    model = torch.nn.DataParallel(model)
    
    num_batches = len(train_loader)
    total_iterations = num_batches * config["max_epochs"]
    
    # Training loop
    for epoch in tqdm(range(config["max_epochs"]), desc="Epochs"):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=num_batches, leave=False, desc=f"Epoch {epoch+1}")
        
        for iter_num, (x, y) in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            global_iter = iter_num + num_batches * epoch
            
            lr = get_lr(global_iter, total_iterations, config) if config["lr_decay"] else config["learning_rate"]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits, loss, diversity_loss = model(x, y, use_causal_mask=True, compute_diversity_loss=True)
                total_loss = loss.mean()
                if diversity_loss is not None:
                    total_loss += config.get("diversity_weight", 0.1) * diversity_loss.mean()

            scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_norm_clip"])
            scaler.step(optimizer)
            scaler.update()

            if writer is not None:
                writer.add_scalar("train/loss", total_loss.item(), global_iter)
                writer.add_scalar("train/lr", lr, global_iter)
                if diversity_loss is not None:
                    writer.add_scalar("train/diversity_loss", diversity_loss.mean().item(), global_iter)

            pbar.set_description(f"Epoch {epoch+1}, Iter {iter_num}: loss {total_loss.item():.5f}, lr {lr:e}")

            # Intermediate validation
            if global_iter > 0 and global_iter % config["val_inter"] == 0:
                print(f"Step {global_iter}: Starting intermediate validation...")
                if config["vocab_path"]:
                    validity = model_validity(model, vocab_path=config["vocab_path"], block_size=config["block_size"])
                else:
                    validity = model_validity(model, vocab=voc, block_size=config["block_size"])
                print(f"Step {global_iter} Validity: {validity:.4f}")
                if writer is not None:
                    writer.add_scalar("val/validity", validity, global_iter)
                model.train()

        # End of epoch validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                _, loss, _ = model(x_val, y_val, use_causal_mask=True)
                val_losses.append(loss.mean().item())
        
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")
        if writer is not None:
            writer.add_scalar("val/loss", avg_val_loss, epoch)

        # Save checkpoint
        if (epoch + 1) % config["save_epoch"] == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch{epoch+1}.pt")
            torch.save(model.module.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
    
    # Save final model
    final_ckpt_path = os.path.join(ckpt_dir, "final_model.pt")
    torch.save(model.module.state_dict(), final_ckpt_path)
    print(f"Saved final model: {final_ckpt_path}")
    
    if writer is not None:
        writer.close()

if __name__ == '__main__':
    main()
