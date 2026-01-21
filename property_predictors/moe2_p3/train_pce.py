import argparse
import json
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset import OPVDataset
from model import P3
import pandas as pd
import numpy as np
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.metrics import r2_score
from config import (
    IN_CHANNELS,
    EDGE_DIM,
    HIDDEN_CHANNELS,
    OUT_CHANNELS,
    PCE_TARGETS,
    HOMOLUMO_TARGETS,
    HEADS,
    DROPOUT_RATE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    MIN_DELTA,
    PREDICT_DATA_PATH,
    HOMOLUMO_MODEL_PATH,
    HOMOLUMO_EXP_MODEL_PATH,
    MLM_MODEL_PATH,
    PCE_MODEL_PATH
)

def get_args():
    parser = argparse.ArgumentParser(description='PCE Prediction Training')
    parser.add_argument('--hidden_channels', type=int, default=HIDDEN_CHANNELS)
    parser.add_argument('--out_channels', type=int, default=OUT_CHANNELS)
    parser.add_argument('--heads', type=int, default=HEADS)
    parser.add_argument('--dropout', type=float, default=DROPOUT_RATE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--output_file', type=str, default='cv_stats.json')
    return parser.parse_args()

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, epochs, scheduler, fold, writer, y_mean, y_std, pce_model_path):
    best_loss = float('inf')
    patience = 0
    # P3 models are saved directly in the p3 directory
    os.makedirs(pce_model_path, exist_ok=True)
    best_model_path = os.path.join(pce_model_path, f'PCE_model_fold{fold+1}.pth')
    size_mismatch_count = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Prepare donor and acceptor batches
            donor_batch = Batch.from_data_list(batch.donor).to(device)
            acceptor_batch = Batch.from_data_list(batch.acceptor).to(device)
            
            # Get actual batch sizes
            donor_batch_size = donor_batch.batch.max() + 1 if donor_batch.batch.numel() > 0 else 0
            acceptor_batch_size = acceptor_batch.batch.max() + 1 if acceptor_batch.batch.numel() > 0 else 0
            expected_batch_size = len(batch.donor)  # This should be the actual batch size
            
            # Forward pass
            out = model(donor_batch, acceptor_batch)
            
            # Debug: Print tensor sizes for first few iterations
            if epoch == 0 and total_train_loss == 0:
                print(f"Debug - Expected batch size: {expected_batch_size}")
                print(f"Debug - Donor batch size: {donor_batch_size}, Acceptor batch size: {acceptor_batch_size}")
                print(f"Debug - Model output size: {out.size()}")
                print(f"Debug - batch.y size: {batch.y.size()}")
            
            out_pce = out[:, 0]
            y_pce = batch.y.view(-1, PCE_TARGETS)[:, 0]
            
            # Scale target
            y_pce_scaled = (y_pce - y_mean) / y_std
            
            # Compute loss
            loss = criterion(out_pce, y_pce_scaled)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        avg_val_loss = evaluate(model, val_loader, criterion, device, y_mean, y_std)
        
        print("Epoch", epoch+1, "Train Loss:", avg_train_loss, "Val Loss:", avg_val_loss)
        
        # Log metrics to tensorboard
        writer.add_scalar(f'Fold{fold+1}/Train_Loss', avg_train_loss, epoch)
        writer.add_scalar(f'Fold{fold+1}/Val_Loss', avg_val_loss, epoch)
        writer.add_scalar(f'Fold{fold+1}/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Calculate R2 score for this epoch
        model.eval()
        epoch_predictions = []
        epoch_actuals = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                donor_batch = Batch.from_data_list(batch.donor).to(device)
                acceptor_batch = Batch.from_data_list(batch.acceptor).to(device)
                
                out = model(donor_batch, acceptor_batch)
                out_pce = out[:, 0]
                y_pce = batch.y.view(-1, PCE_TARGETS)[:, 0]
                
                # Inverse scale predictions
                out_pce_original = out_pce * y_std + y_mean
                
                epoch_predictions.extend(out_pce_original.cpu().numpy())
                epoch_actuals.extend(y_pce.cpu().numpy())
        
        epoch_r2 = r2_score(epoch_actuals, epoch_predictions)
        print(f"Epoch {epoch+1} R2 Score: {epoch_r2:.4f}")
        writer.add_scalar(f'Fold{fold+1}/Epoch_R2', epoch_r2, epoch)

        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss + MIN_DELTA < best_loss:
            best_loss = avg_val_loss
            patience = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print(f'Early stopping at epoch {epoch+1}')
                break
                
    # Load best model for this fold
    model.load_state_dict(torch.load(best_model_path))
    return model

def evaluate(model, dataloader, criterion, device, y_mean, y_std):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            donor_batch = Batch.from_data_list(batch.donor).to(device)
            acceptor_batch = Batch.from_data_list(batch.acceptor).to(device)
            
            # Get actual batch sizes
            expected_batch_size = len(batch.donor)  # This should be the actual batch size
            
            out = model(donor_batch, acceptor_batch)
            
            out_pce = out[:, 0]
            y_pce = batch.y.view(-1, PCE_TARGETS)[:, 0]
            
            # Scale target for loss calculation
            y_pce_scaled = (y_pce - y_mean) / y_std
                
            loss = criterion(out_pce, y_pce_scaled)
            total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create runs directory in the moe2_p3 folder
    runs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs')
    os.makedirs(runs_dir, exist_ok=True)
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(runs_dir, 'PCE-Predict'))
    
    # Create results directory in the moe2_p3 folder
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load dataset
    dataset = OPVDataset(PREDICT_DATA_PATH)
    kf = KFold(n_splits=5, shuffle=True, random_state=3407)
    
    all_predictions = []
    all_actuals = []
    fold_r2_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n=== Processing Fold {fold+1}/5 ===")
        
        # Calculate mean and std for target scaling
        train_subset = Subset(dataset, train_idx)
        # Accessing data directly from dataframe for speed
        all_train_y = dataset.data.iloc[train_idx]['PCE'].values
        y_mean = torch.tensor(np.mean(all_train_y), dtype=torch.float32, device=device)
        y_std = torch.tensor(np.std(all_train_y), dtype=torch.float32, device=device)
        print(f"Fold {fold+1} Target Stats - Mean: {y_mean.item():.4f}, Std: {y_std.item():.4f}")

        # Create data loaders
        train_loader = DataLoader(train_subset, 
                                batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(Subset(dataset, val_idx),
                                batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
        
        # Initialize model and optimizer
        model = P3(
            IN_CHANNELS,
            EDGE_DIM,
            args.hidden_channels,
            args.out_channels,
            HOMOLUMO_TARGETS,
            PCE_TARGETS,
            args.dropout,
            args.heads
        ).to(device)
        
        try:
            # Load the fully pretrained model (MLM + Frozen HOMO + Frozen HOMO_EXP)
            pretrained_state_dict = torch.load(HOMOLUMO_MODEL_PATH, map_location=device)
            
            # Check if shapes match before loading
            # This is a simple check, might need more robust handling for grid search
            model_dict = model.donor_encoder.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            
            if len(pretrained_dict) != len(pretrained_state_dict):
                print("Warning: Some pretrained weights were skipped due to shape mismatch (likely due to architecture change in grid search).")
            
            model.donor_encoder.load_state_dict(pretrained_dict, strict=False)
            model.acceptor_encoder.load_state_dict(pretrained_dict, strict=False)
            print(f"Successfully loaded compatible pre-trained GNN weights from {HOMOLUMO_EXP_MODEL_PATH}.")
        except FileNotFoundError:
            print(f"Pre-trained GNN weights file {HOMOLUMO_EXP_MODEL_PATH} not found. Training from scratch.")
        except RuntimeError as e:
            print(f"Error loading pre-trained weights: {e}. Training from scratch.")

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        criterion = nn.MSELoss()
        
        # === Key modification: two-stage training strategy ===
        # Stage 1: Freeze GNN, only train regression head and fingerprint layers (Warmup)
        print("Phase 1: Freezing GNN to warmup regression head...")
        for param in model.donor_encoder.parameters():
            param.requires_grad = False
        for param in model.acceptor_encoder.parameters():
            param.requires_grad = False
            
        # Train for warmup_epochs epochs during the warmup phase
        warmup_epochs = args.warmup_epochs
        model = train_and_validate(
            model, train_loader, val_loader, criterion,
            optimizer, device, warmup_epochs, scheduler, fold, writer, y_mean, y_std, PCE_MODEL_PATH
        )
        
        # 阶段 2: 解冻 GNN，进行全局微调
        print("Phase 2: Unfreezing GNN for fine-tuning...")
        for param in model.donor_encoder.parameters():
            param.requires_grad = True
        for param in model.acceptor_encoder.parameters():
            param.requires_grad = True
            
        # Lower the learning rate for fine-tuning
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1
            
        # 继续训练剩余 Epoch
        remaining_epochs = args.epochs - warmup_epochs
        model = train_and_validate(
            model, train_loader, val_loader, criterion,
            optimizer, device, remaining_epochs, scheduler, fold, writer, y_mean, y_std, PCE_MODEL_PATH
        )
        
        # Generate predictions for this fold
        model.eval()
        fold_predictions = []
        fold_actuals = []
        fold_ids = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                donor_batch = Batch.from_data_list(batch.donor).to(device)
                acceptor_batch = Batch.from_data_list(batch.acceptor).to(device)
                
                # Get actual batch sizes
                expected_batch_size = len(batch.donor)  # This should be the actual batch size
                
                out = model(donor_batch, acceptor_batch)
                
                out_pce = out[:, 0]
                y_pce = batch.y.view(-1, PCE_TARGETS)[:, 0]
                
                # Inverse scale predictions
                out_pce_original = out_pce * y_std + y_mean
                
                fold_predictions.extend(out_pce_original.cpu().numpy())
                fold_actuals.extend(y_pce.cpu().numpy())
                fold_ids.extend(batch.mol_id.view(-1).cpu().numpy())

            fold_errors = np.abs(np.array(fold_predictions) - np.array(fold_actuals))
            error_df = pd.DataFrame({
                'Mol_ID': fold_ids,
                'Actual': fold_actuals,
                'Predicted': fold_predictions,
                'Absolute_Error': fold_errors
            })
            # Save full fold errors for downstream diagnostics in ckpt/p3 directory
            all_errors_path = os.path.join(PCE_MODEL_PATH, f'fold{fold+1}_all_errors.csv')
            top50_errors_path = os.path.join(PCE_MODEL_PATH, f'fold{fold+1}_top50_errors.csv')
            error_df.to_csv(all_errors_path, index=False)
            top_errors = error_df.sort_values('Absolute_Error', ascending=False).head(50)
            top_errors.to_csv(top50_errors_path, index=False)
            print(f"Top 50 absolute errors for fold {fold+1} (saved to {top50_errors_path}):")
            print(top_errors[['Mol_ID', 'Absolute_Error']].to_string(index=False))
        
        all_predictions.extend(fold_predictions)
        all_actuals.extend(fold_actuals)
        # Calculate R2 score for this fold

        r2_score_fold = r2_score(fold_actuals, fold_predictions)
        print(f"Fold {fold+1} R2 Score: {r2_score_fold}")
        writer.add_scalar('CrossValidation/Fold_R2', r2_score_fold, fold)
        fold_r2_scores.append(r2_score_fold)
    
    # Save cross-validation results in ckpt/p3 directory
    results_df = pd.DataFrame({
        'Actual': all_actuals,
        'Predicted': all_predictions
    })
    cv_results_path = os.path.join(PCE_MODEL_PATH, 'cross_validation_results.csv')
    results_df.to_csv(cv_results_path, index=False)
    print(f"\nCross-validation results saved to {cv_results_path}")
    
    # Calculate and save stats
    mean_r2 = np.mean(fold_r2_scores)
    std_r2 = np.std(fold_r2_scores)
    print(f"\nMean R2: {mean_r2:.4f} ± {std_r2:.4f}")
    
    stats = {
        'mean_r2': mean_r2,
        'std_r2': std_r2,
        'fold_scores': fold_r2_scores,
        'args': vars(args)
    }
    
    # Save stats to results directory
    output_path = os.path.join(results_dir, args.output_file)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Stats saved to {output_path}")
    
    writer.close()