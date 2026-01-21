import torch
from rdkit import Chem
from torch_geometric.data import Batch
import numpy as np

from config import (
    IN_CHANNELS, EDGE_DIM, HIDDEN_CHANNELS, OUT_CHANNELS, HOMOLUMO_TARGETS, PCE_TARGETS, HEADS, DROPOUT_RATE,
    PREDICT_DATA_PATH, PCE_MODEL_PATH
)
from dataset import mol_to_graph
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol_to_graph(mol)

def load_model(model_class, model_path, device):
    # Handle MOE2 and P3 with correct arguments
    if model_class.__name__ == 'MOE2':
        model = model_class(
            IN_CHANNELS, EDGE_DIM, HIDDEN_CHANNELS, OUT_CHANNELS,
            mlm_output_dim=9, regression_targets=HOMOLUMO_TARGETS, heads=HEADS, dropout_rate=DROPOUT_RATE
        )
    elif model_class.__name__ == 'P3':
        model = model_class(
            IN_CHANNELS, EDGE_DIM, HIDDEN_CHANNELS, OUT_CHANNELS,
            HOMOLUMO_TARGETS, PCE_TARGETS, dropout_rate=DROPOUT_RATE, num_heads=HEADS
        )
    else:
        model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_pce(donor_smiles, acceptor_smiles, model, device, y_mean=0.0, y_std=1.0):
    model.eval()
    results = []
    for d_smi, a_smi in zip(donor_smiles, acceptor_smiles):
        try:
            donor_graph = smiles_to_graph(d_smi)
            acceptor_graph = smiles_to_graph(a_smi)
        except Exception:
            results.append(-100.0)
            continue
        donor_batch = Batch.from_data_list([donor_graph]).to(device)
        acceptor_batch = Batch.from_data_list([acceptor_graph]).to(device)
        with torch.no_grad():
            out = model(donor_batch, acceptor_batch)
            out = out[:, 0] if out.dim() > 1 else out

            # Undo normalization
            val = out.cpu().numpy().item()
            val = val * y_std + y_mean
            results.append(val)
    return np.array(results)

def predict_homo_lumo(smiles_list, model, device):
    model.eval()
    graphs = []
    for smi in smiles_list:
        try:
            graph = smiles_to_graph(smi)
            graphs.append(graph)
        except Exception:
            continue  # skip invalid SMILES
    if not graphs:
        return  # do not return anything if no valid graphs
    batch = Batch.from_data_list(graphs).to(device)
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, task_type="homo_lumo")
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return out.cpu().numpy()

if __name__ == "__main__":
    import pandas as pd
    from model import P3NodeLevel
    from dataset import OPVDataset
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    import torch
    import os

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = OPVDataset(PREDICT_DATA_PATH)

    # Replicate Training Split to get statistics for Fold 2
    TARGET_FOLD_IDX = 1  # Fold 2 (0-based index 1)
    
    # Calculate global stats (using all data) for generalization
    all_y = dataset.data['PCE'].values
    y_mean = np.mean(all_y)
    y_std = np.std(all_y)
    print(f"Global Stats | Mean: {y_mean:.4f}, Std: {y_std:.4f}")

    donor_smiles = [dataset.data.iloc[i]['Donor SMILES'] for i in range(len(dataset))]
    acceptor_smiles = [dataset.data.iloc[i]['Acceptor SMILES'] for i in range(len(dataset))]
    true_pce = [dataset.data.iloc[i]['PCE'] for i in range(len(dataset))]

    # Load model
    checkpoint_path = os.path.join(PCE_MODEL_PATH, f"PCE_model_fold{TARGET_FOLD_IDX+1}.pth")
    model = load_model(P3NodeLevel, checkpoint_path, device)

    # Predict in batches with tqdm
    from tqdm import tqdm
    batch_size = 128
    pred_pce = []
    for i in tqdm(range(0, len(donor_smiles), batch_size), desc="Predicting PCE"):
        donor_batch = donor_smiles[i:i+batch_size]
        acceptor_batch = acceptor_smiles[i:i+batch_size]
        batch_pred = predict_pce(donor_batch, acceptor_batch, model, device, y_mean=y_mean, y_std=y_std)
        pred_pce.extend(batch_pred)
    pred_pce = np.array(pred_pce)

    # Compute R^2
    r2 = r2_score(true_pce, pred_pce)
    print(f"R^2 score on csv: {r2:.4f}") 