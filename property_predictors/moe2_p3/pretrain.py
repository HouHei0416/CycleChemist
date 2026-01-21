import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import csv
import os
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from model import MOE2
from dataset import MLMDataset, HomoLumoDataset
from config import (
    IN_CHANNELS, EDGE_DIM, HIDDEN_CHANNELS, OUT_CHANNELS,
    REGRESSION_TARGETS, HEADS, DROPOUT_RATE, BATCH_SIZE,
    MLM_EPOCHS, HOMO_EPOCHS, MLM_LEARNING_RATE, HOMO_LEARNING_RATE,
    MLM_DATA_PATH, HOMO_DATA_PATH, HOMO_EXP_DATA_PATH, EARLY_STOPPING_PATIENCE,
    VAL_RATIO, SEED, MLM_MODEL_PATH, HOMOLUMO_MODEL_PATH, HOMOLUMO_EXP_MODEL_PATH
)

def get_args():
    parser = argparse.ArgumentParser(description='Pretraining GNN')
    parser.add_argument('--hidden_channels', type=int, default=HIDDEN_CHANNELS)
    parser.add_argument('--out_channels', type=int, default=OUT_CHANNELS)
    parser.add_argument('--heads', type=int, default=HEADS)
    parser.add_argument('--dropout', type=float, default=DROPOUT_RATE)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--mlm_lr', type=float, default=MLM_LEARNING_RATE)
    parser.add_argument('--homo_lr', type=float, default=HOMO_LEARNING_RATE)
    parser.add_argument('--mlm_epochs', type=int, default=MLM_EPOCHS)
    parser.add_argument('--homo_epochs', type=int, default=HOMO_EPOCHS)
    return parser.parse_args()

def train_mlm_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for data in tqdm(loader, desc="MLM Training"):
        data = data.to(device)
        optimizer.zero_grad()
        
        pred = model(data.x, data.edge_index, data.edge_attr, data.batch, "mlm")
        loss = criterion(pred.view(-1, pred.size(-1)), data.mask_labels.view(-1))
        
        loss.backward()
        optimizer.step()
        
        mask = data.mask_labels != -100
        correct += (pred.argmax(-1).view(-1)[mask] == data.mask_labels.view(-1)[mask]).sum().item()
        total += mask.sum().item()
        total_loss += loss.item()
    
    return {
        "train/mlm_loss": total_loss / len(loader),
        "train/mlm_acc": correct / total if total > 0 else 0
    }

def evaluate_mlm(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch, "mlm")
            loss = criterion(pred.view(-1, pred.size(-1)), data.mask_labels.view(-1))
            
            mask = data.mask_labels != -100
            correct += (pred.argmax(-1).view(-1)[mask] == data.mask_labels.view(-1)[mask]).sum().item()
            total += mask.sum().item()
            total_loss += loss.item()
    
    return {
        "val/mlm_loss": total_loss / len(loader),
        "val/mlm_acc": correct / total if total > 0 else 0
    }

def train_homo_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="HOMO Training"):
        data = data.to(device)
        optimizer.zero_grad()
        
        pred = model(data.x, data.edge_index, data.edge_attr, data.batch, "homo_lumo")
        pred = pred.view(-1)
        loss = criterion(pred, data.y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return {"train/homo_loss": total_loss / len(loader)}

def evaluate_homo(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch, "homo_lumo")
            pred = pred.view(-1)
            loss = criterion(pred, data.y)
            total_loss += loss.item()
    
    return {"val/homo_loss": total_loss / len(loader)}

def save_metrics_to_csv(metrics_list, csv_path, fieldnames):

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for metrics in metrics_list:
            writer.writerow(metrics)

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create runs directory in the moe2_p3 folder
    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
    os.makedirs(runs_dir, exist_ok=True)
    writer_mlm = SummaryWriter(log_dir=os.path.join(runs_dir, 'MLM-Pretrain'))
    
    mlm_dataset = MLMDataset(MLM_DATA_PATH)
    mlm_val_size = int(len(mlm_dataset) * VAL_RATIO)
    mlm_train_size = len(mlm_dataset) - mlm_val_size
    mlm_train, mlm_val = random_split(mlm_dataset, [mlm_train_size, mlm_val_size],
                                     generator=torch.Generator().manual_seed(SEED))
    
    mlm_train_loader = DataLoader(mlm_train, args.batch_size, shuffle=True)
    mlm_val_loader = DataLoader(mlm_val, args.batch_size, shuffle=False)
    
    model = MOE2(
        in_channels=IN_CHANNELS,
        edge_dim=EDGE_DIM,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        regression_targets=REGRESSION_TARGETS,
        heads=args.heads,
        dropout_rate=args.dropout
    ).to(device)
    
    mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=args.mlm_lr)
    
    best_mlm_loss = float('inf')
    mlm_metrics_list = []
    # Save CSV in the same directory as the model, with a clean name
    mlm_csv_path = os.path.join(os.path.dirname(MLM_MODEL_PATH), "MLM_metrics.csv")
    mlm_fieldnames = ["epoch", "train/mlm_loss", "train/mlm_acc", "val/mlm_loss", "val/mlm_acc"]
    for epoch in range(1, args.mlm_epochs+1):
        train_metrics = train_mlm_epoch(model, mlm_train_loader, mlm_criterion, optimizer, device)
        val_metrics = evaluate_mlm(model, mlm_val_loader, mlm_criterion, device)
        
        # Log metrics to tensorboard
        writer_mlm.add_scalar('Train/MLM_Loss', train_metrics["train/mlm_loss"], epoch)
        writer_mlm.add_scalar('Train/MLM_Acc', train_metrics["train/mlm_acc"], epoch)
        writer_mlm.add_scalar('Val/MLM_Loss', val_metrics["val/mlm_loss"], epoch)
        writer_mlm.add_scalar('Val/MLM_Acc', val_metrics["val/mlm_acc"], epoch)
        
        # Save metrics for this epoch
        metrics_row = {"epoch": epoch}
        metrics_row.update(train_metrics)
        metrics_row.update(val_metrics)
        mlm_metrics_list.append(metrics_row)
        
        if val_metrics["val/mlm_loss"] < best_mlm_loss:
            best_mlm_loss = val_metrics["val/mlm_loss"]
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(MLM_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), MLM_MODEL_PATH)
        
        print(f"Epoch {epoch}: MLM Loss {val_metrics['val/mlm_loss']:.4f}, Acc {val_metrics['val/mlm_acc']:.4f}")
    
    # Save all MLM metrics to CSV
    os.makedirs(os.path.dirname(mlm_csv_path), exist_ok=True)
    save_metrics_to_csv(mlm_metrics_list, mlm_csv_path, mlm_fieldnames)
    writer_mlm.close()

    writer_homo = SummaryWriter(log_dir=os.path.join(runs_dir, 'HOMO-Finetune'))
    
    homo_dataset = HomoLumoDataset(HOMO_DATA_PATH)
    homo_val_size = int(len(homo_dataset) * VAL_RATIO)
    homo_train_size = len(homo_dataset) - homo_val_size
    homo_train, homo_val = random_split(homo_dataset, [homo_train_size, homo_val_size],
                                       generator=torch.Generator().manual_seed(SEED+1))
    
    homo_train_loader = DataLoader(homo_train, args.batch_size, shuffle=True)
    homo_val_loader = DataLoader(homo_val, args.batch_size, shuffle=False)

    model = MOE2(
        in_channels=IN_CHANNELS,
        edge_dim=EDGE_DIM,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        regression_targets=REGRESSION_TARGETS,
        heads=args.heads,
        dropout_rate=args.dropout
    ).to(device)
    model.load_state_dict(torch.load(MLM_MODEL_PATH))

    print("Freezing GNN backbone for HOMO finetuning...")
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False
    for param in model.conv3.parameters():
        param.requires_grad = False
    
    homo_criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.homo_lr)
    
    best_homo_loss = float('inf')
    homo_metrics_list = []
    # Save CSV in the same directory as the model, with a clean name
    homo_csv_path = os.path.join(os.path.dirname(HOMOLUMO_MODEL_PATH), "homolumo_metrics.csv")
    homo_fieldnames = ["epoch", "train/homo_loss", "val/homo_loss"]
    for epoch in range(1, args.homo_epochs+1):
        train_metrics = train_homo_epoch(model, homo_train_loader, homo_criterion, optimizer, device)
        val_metrics = evaluate_homo(model, homo_val_loader, homo_criterion, device)
        
        # Log metrics to tensorboard
        writer_homo.add_scalar('Train/HOMO_Loss', train_metrics["train/homo_loss"], epoch)
        writer_homo.add_scalar('Val/HOMO_Loss', val_metrics["val/homo_loss"], epoch)
        
        metrics_row = {"epoch": epoch}
        metrics_row.update(train_metrics)
        metrics_row.update(val_metrics)
        homo_metrics_list.append(metrics_row)
        
        if val_metrics["val/homo_loss"] < best_homo_loss:
            best_homo_loss = val_metrics["val/homo_loss"]
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(HOMOLUMO_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), HOMOLUMO_MODEL_PATH)
        
        print(f"Epoch {epoch}: HOMO Loss {val_metrics['val/homo_loss']:.4f}")

    os.makedirs(os.path.dirname(homo_csv_path), exist_ok=True)
    save_metrics_to_csv(homo_metrics_list, homo_csv_path, homo_fieldnames)
    writer_homo.close()

    writer_homo_exp = SummaryWriter(log_dir=os.path.join(runs_dir, 'HOMO-EXP-Finetune'))

    homo_dataset = HomoLumoDataset(HOMO_EXP_DATA_PATH)
    homo_val_size = int(len(homo_dataset) * VAL_RATIO)
    homo_train_size = len(homo_dataset) - homo_val_size
    homo_train, homo_val = random_split(homo_dataset, [homo_train_size, homo_val_size],
                                       generator=torch.Generator().manual_seed(SEED+1))
    
    homo_train_loader = DataLoader(homo_train, args.batch_size, shuffle=True)
    homo_val_loader = DataLoader(homo_val, args.batch_size, shuffle=False)

    model = MOE2(
        in_channels=IN_CHANNELS,
        edge_dim=EDGE_DIM,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
        regression_targets=REGRESSION_TARGETS,
        heads=args.heads,
        dropout_rate=args.dropout
    ).to(device)
    model.load_state_dict(torch.load(HOMOLUMO_MODEL_PATH))

    print("Freezing GNN backbone for HOMO_EXP finetuning...")
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False
    for param in model.conv3.parameters():
        param.requires_grad = False
    
    homo_criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.homo_lr)

    best_homo_loss = float('inf')
    homo_exp_metrics_list = []
    # Save CSV in the same directory as the model, with a clean name
    homo_exp_csv_path = os.path.join(os.path.dirname(HOMOLUMO_EXP_MODEL_PATH), "homolumo_exp_metrics.csv")
    homo_exp_fieldnames = ["epoch", "train/homo_loss", "val/homo_loss"]
    for epoch in range(1, args.homo_epochs+1):
        train_metrics = train_homo_epoch(model, homo_train_loader, homo_criterion, optimizer, device)
        val_metrics = evaluate_homo(model, homo_val_loader, homo_criterion, device)
        
        # Log metrics to tensorboard
        writer_homo_exp.add_scalar('Train/HOMO_Loss', train_metrics["train/homo_loss"], epoch)
        writer_homo_exp.add_scalar('Val/HOMO_Loss', val_metrics["val/homo_loss"], epoch)

        metrics_row = {"epoch": epoch}
        metrics_row.update(train_metrics)
        metrics_row.update(val_metrics)
        homo_exp_metrics_list.append(metrics_row)
        
        if val_metrics["val/homo_loss"] < best_homo_loss:
            best_homo_loss = val_metrics["val/homo_loss"]
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(HOMOLUMO_EXP_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), HOMOLUMO_EXP_MODEL_PATH)
        
        print(f"Epoch {epoch}: HOMO Loss {val_metrics['val/homo_loss']:.4f}")

    os.makedirs(os.path.dirname(homo_exp_csv_path), exist_ok=True)
    save_metrics_to_csv(homo_exp_metrics_list, homo_exp_csv_path, homo_exp_fieldnames)
    writer_homo_exp.close()

if __name__ == "__main__":
    main()
