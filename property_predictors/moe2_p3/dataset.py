import torch 
from torch_geometric.data import Dataset, Data

import os
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
import random

def mol_to_graph(mol):
    from torch_geometric.data import Data  # Ensure Data is imported

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(full_atom_features(atom))
    x = torch.tensor(atom_features, dtype=torch.float)
    
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        bond_features = bond_full_features(bond)
        edge_attr += [bond_features, bond_features]
    
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, bond_feature_dim()), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def full_atom_features(atom):

    atom_type = atom.GetSymbol()
    possible_atoms = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
    atom_type_one_hot = [int(atom_type == a) for a in possible_atoms]
    
    formal_charge = [atom.GetFormalCharge()]
    
    hybridization = atom.GetHybridization()
    possible_hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    hybridization_one_hot = [int(atom.GetHybridization() == h) for h in possible_hybridizations]
    
    aromatic = [int(atom.GetIsAromatic())]

    degree = [atom.GetDegree()]

    num_hydrogens = [atom.GetTotalNumHs()]

    in_ring = [int(atom.IsInRing())]

    try:
        partial_charge = [float(atom.GetProp('_GasteigerCharge'))]
    except:
        partial_charge = [0.0]

    is_heteroatom = [int(atom_type not in ['C'])]

    is_terminal = [int(atom.GetDegree() == 1)]

    is_positive_charge = [int(atom.GetFormalCharge() > 0)]

    is_negative_charge = [int(atom.GetFormalCharge() < 0)]

    atomic_radius_map = {
        'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
        'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
    }
    atomic_radius = [atomic_radius_map.get(atom_type, 1.70)]  # Default: 1.70 Å

    lone_pairs = [atom.GetTotalValence() - atom.GetExplicitValence() - atom.GetTotalNumHs() - atom.GetFormalCharge()]

    radical_electrons = [atom.GetNumRadicalElectrons()]

    is_double_bond_carbon = [int(atom_type == 'C' and atom.GetTotalNumHs() < 4 and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2)]

    is_triple_bond_carbon = [int(atom_type == 'C' and atom.GetTotalNumHs() < 4 and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP)]

    is_aromatic_carbon = [int(atom_type == 'C' and atom.GetIsAromatic())]

    is_charged = [int(atom.GetFormalCharge() != 0)]
    
    # Total atom feature dimension:
    # atom_type (9) + formal_charge (1) + hybridization_one_hot (5) + aromatic (1) +
    # degree (1) + num_hydrogens (1) + in_ring (1) + partial_charge (1) +
    # is_heteroatom (1) + is_terminal (1) + is_positive_charge (1) + is_negative_charge (1) +
    # atomic_radius (1) + lone_pairs (1) + radical_electrons (1) +
    # is_double_bond_carbon (1) + is_triple_bond_carbon (1) + is_aromatic_carbon (1) +
    # is_charged (1) = 28
    return (
        atom_type_one_hot + 
        formal_charge + 
        hybridization_one_hot + 
        aromatic + 
        degree + 
        num_hydrogens + 
        in_ring + 
        partial_charge + 
        is_heteroatom + 
        is_terminal + 
        is_positive_charge + 
        is_negative_charge + 
        atomic_radius + 
        lone_pairs + 
        radical_electrons + 
        is_double_bond_carbon + 
        is_triple_bond_carbon + 
        is_aromatic_carbon + 
        is_charged
    )

def bond_full_features(bond):
    # Bond type one-hot
    bond_type = bond.GetBondType()
    possible_bond_types = [Chem.rdchem.BondType.SINGLE,
                           Chem.rdchem.BondType.DOUBLE,
                           Chem.rdchem.BondType.TRIPLE,
                           Chem.rdchem.BondType.AROMATIC]
    bond_type_one_hot = [int(bond_type == bt) for bt in possible_bond_types]
    
    # Bond stereo
    stereo = bond.GetStereo()
    possible_stereo = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE
    ]
    bond_stereo_one_hot = [int(stereo == st) for st in possible_stereo]
    
    # Is conjugated
    is_conjugated = [int(bond.GetIsConjugated())]
    
    # Is in a ring
    is_in_ring = [int(bond.IsInRing())]
    
    # Number of rings the bond is part of
    num_rings =   [bond.GetOwningMol().GetRingInfo().NumBondRings(bond.GetIdx())]
    
    # Bond direction (applicable for double bonds)
    bond_direction = [int(bond.GetBondDir() != Chem.rdchem.BondDir.NONE)]
    
    # Whether this is a bridge bond (bond in a bridged ring)
    is_bridge_bond = [int(bond.IsInRing() and bond.GetOwningMol().GetRingInfo().NumBondRings(bond.GetIdx()) > 1)]
    
    # Bond order
    bond_order_map = {
        Chem.rdchem.BondType.SINGLE: 1.0,
        Chem.rdchem.BondType.DOUBLE: 2.0,
        Chem.rdchem.BondType.TRIPLE: 3.0,
        Chem.rdchem.BondType.AROMATIC: 1.5
    }
    bond_order = [bond_order_map.get(bond_type, 1.0)]
    
    
    # Whether the bond participates in conjugation
    is_conjugated = [int(bond.GetIsConjugated())]
    
    # Total bond feature dimension:
    # bond_type_one_hot (4) + bond_stereo_one_hot (4) + is_conjugated (1) + 
    # is_in_ring (1) + num_rings (1) + bond_direction (1) + is_bridge_bond (1) + 
    # bond_order (1) + is_conjugated (1) = 15
    return (
        bond_type_one_hot + 
        bond_stereo_one_hot + 
        is_conjugated + 
        is_in_ring + 
        num_rings + 
        bond_direction + 
        is_bridge_bond + 
        bond_order + 
        is_conjugated
    )

def bond_feature_dim():
    return 15


def global_scalar_features(mol):
    """Return a small set of global scalar descriptors as a tensor."""
    feats = [
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumRotatableBonds(mol),
    ]
    return torch.tensor(feats, dtype=torch.float)

class HomoLumoDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        
        # Detect column format: check if it's OPV format (with Donor/Acceptor) or simple format
        if 'smiles' in self.data.columns:
            # Simple format: smiles, HOMO_calib, LUMO_calib
            self.format_type = 'simple'
        elif 'Donor SMILES' in self.data.columns and 'Acceptor SMILES' in self.data.columns:
            # OPV format: Donor SMILES, Acceptor SMILES, HOMO_D, LUMO_D, HOMO_A, LUMO_A
            self.format_type = 'opv'
            # Create expanded dataset with both donor and acceptor molecules
            self._expand_opv_data()
        else:
            raise ValueError(f"Unknown dataset format. Expected 'smiles' column or 'Donor SMILES'/'Acceptor SMILES' columns.")

        # Determine which HOMO/LUMO columns to use for simple format
        # Support both calibrated (`HOMO_calib`/`LUMO_calib`) and original (`HOMO`/`LUMO`) names.
        if self.format_type == 'simple':
            if 'HOMO_calib' in self.data.columns and 'LUMO_calib' in self.data.columns:
                self.homo_col = 'HOMO_calib'
                self.lumo_col = 'LUMO_calib'
            elif 'HOMO' in self.data.columns and 'LUMO' in self.data.columns:
                self.homo_col = 'HOMO'
                self.lumo_col = 'LUMO'
            else:
                raise ValueError(
                    "Simple HOMO/LUMO dataset must contain either "
                    "`HOMO_calib`/`LUMO_calib` or `HOMO`/`LUMO` columns."
                )

    def _expand_opv_data(self):
        """Expand OPV dataset to include both donor and acceptor molecules"""
        expanded_rows = []
        for idx, row in self.data.iterrows():
            # Add donor molecule
            if pd.notna(row['Donor SMILES']) and pd.notna(row['HOMO_D']) and pd.notna(row['LUMO_D']):
                expanded_rows.append({
                    'smiles': row['Donor SMILES'],
                    'HOMO': row['HOMO_D'],
                    'LUMO': row['LUMO_D']
                })
            # Add acceptor molecule
            if pd.notna(row['Acceptor SMILES']) and pd.notna(row['HOMO_A']) and pd.notna(row['LUMO_A']):
                expanded_rows.append({
                    'smiles': row['Acceptor SMILES'],
                    'HOMO': row['HOMO_A'],
                    'LUMO': row['LUMO_A']
                })
        self.data = pd.DataFrame(expanded_rows)
        self.format_type = 'simple'  # Now it's in simple format

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data.iloc[idx]['smiles']
        # Use the selected HOMO/LUMO column names (calibrated or original)
        homo = self.data.iloc[idx][self.homo_col]
        lumo = self.data.iloc[idx][self.lumo_col]

        # Use RDKit to generate the molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Handle invalid SMILES
            raise ValueError(f"Invalid SMILES: {smiles}")

        label = [homo] + [lumo] 
        label = torch.tensor(label, dtype=torch.float32)

        # Convert molecule to graph data
        graph = mol_to_graph(mol)
        graph.y = label  # Attach labels to graph data
        graph.task_type = 'homo_lumo'

        if self.transform:
            graph = self.transform(graph)

        return graph
    
class MLMDataset(Dataset):
    def __init__(self, csv_path, mask_ratio=0.15, transform=None):
        self.data = pd.read_csv(csv_path)
        self.mask_ratio = mask_ratio
        self.atom_type_vocab = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']
        self.transform = transform
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Original SMILES processing logic remains unchanged
        smiles = self.data.iloc[idx]['smiles']

        # Use RDKit to generate the molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Handle invalid SMILES
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Convert molecule to graph data
        graph = mol_to_graph(mol)
        # Apply masking logic
        graph = self.apply_mask(graph)
        graph.task_type = 'mlm'
        return graph

    def apply_mask(self, data):
        """Atom type masking strategy"""
        num_nodes = data.x.size(0)
        num_mask = max(1, int(num_nodes * self.mask_ratio))
        
        # Randomly select atoms to mask
        mask_indices = torch.randperm(num_nodes)[:num_mask]
        
        # Save original atom types as labels
        atom_type_labels = torch.full((num_nodes,), -100)  # -100 means not used in loss
        atom_type_labels[mask_indices] = data.x[mask_indices, :len(self.atom_type_vocab)].argmax(dim=1)
        
        # Mask features (zero out node features to prevent information leakage)
        masked_x = data.x.clone()
        masked_x[mask_indices] = 0  # Mask all features for the selected nodes
        
        # Add mask labels to data object
        data.x = masked_x
        data.mask_labels = atom_type_labels  # New label field
        return data
     
class OPVDataset(Dataset):
    def __init__(self, csv_path,scaler=None, transform=None, pre_transform=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[
            (self.data['Donor SMILES'].str.len() <= 600) & 
            (self.data['Acceptor SMILES'].str.len() <= 600)
        ]
        self.scaler = scaler
        super(OPVDataset, self).__init__('.', transform, pre_transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        donor_smiles = self.data.iloc[idx]['Donor SMILES']
        acceptor_smiles = self.data.iloc[idx]['Acceptor SMILES']
        mol_id = self.data.iloc[idx]['Mol_ID']  # Persist unique row id
        
        donor_mol = Chem.MolFromSmiles(donor_smiles)
        acceptor_mol = Chem.MolFromSmiles(acceptor_smiles)
        
        donor_graph = mol_to_graph(donor_mol)
        acceptor_graph = mol_to_graph(acceptor_mol)

        donor_global = global_scalar_features(donor_mol).unsqueeze(0)
        acceptor_global = global_scalar_features(acceptor_mol).unsqueeze(0)
        
        y = torch.tensor([
            self.data.iloc[idx]['PCE'],
            # self.data.iloc[idx]['Voc'],
            # self.data.iloc[idx]['Jsc'],
            # self.data.iloc[idx]['FF']
        ], dtype=torch.float)
        data = Data(
            donor=donor_graph,
            acceptor=acceptor_graph,
            y=y,
            donor_global=donor_global,
            acceptor_global=acceptor_global,
            mol_id=torch.tensor([mol_id], dtype=torch.long),
        )
        return data