#!/usr/bin/env python3
import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

# Setup paths to avoid namespace conflicts
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the MatGPT model directory to path to avoid 'model' namespace conflict
_MATGPT_MODEL_DIR = os.path.join(CODE_DIR, 'model')
sys.path.insert(0, _MATGPT_MODEL_DIR)
from MatGPT import MatGPT, MatGPTConfig
sys.path.pop(0)

from utils import sample_SMILES, set_seed
from vocabulary import read_vocabulary

def get_parser():
    parser = argparse.ArgumentParser(description='Generate SMILES using trained MatGPT model')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vocab_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--max_length', type=int, default=340)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--nucleus_p', type=float, default=None)
    parser.add_argument('--use_temperature_annealing', action='store_true')
    parser.add_argument('--min_temperature', type=float, default=0.5)
    parser.add_argument('--max_temperature', type=float, default=1.5)
    parser.add_argument('--use_diversity_sampling', action='store_true')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--save_invalid', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    return parser

def load_model(model_path, vocab, config_path=None, device='cuda:0'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Load config
    config_dict = {'vocab_size': len(vocab)}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict.update(json.load(f))
    
    model_config = MatGPTConfig(**config_dict)
    model = MatGPT(model_config)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    
    return model

def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    
    vocab = read_vocabulary(args.vocab_path)
    model = load_model(args.model_path, vocab, args.config_path, args.device)
    
    n_batches = (args.n_samples + args.batch_size - 1) // args.batch_size
    all_smiles = []

    for _ in tqdm(range(n_batches), desc="Sampling"):
        batch_size = min(args.batch_size, args.n_samples - len(all_smiles))
        smiles, _, _ = sample_SMILES(
            model=model, voc=vocab, n_mols=batch_size, 
            block_size=args.max_length, temperature=args.temperature, top_k=args.top_k,
            nucleus_p=args.nucleus_p, use_temperature_annealing=args.use_temperature_annealing,
            min_temp=args.min_temperature, max_temp=args.max_temperature,
            use_diversity_sampling=args.use_diversity_sampling
        )
        all_smiles.extend(smiles)

    # Filter
    valid_smiles = []
    invalid_smiles = []
    for s in all_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol: valid_smiles.append(s)
        else: invalid_smiles.append(s)

    print(f"Total: {len(all_smiles)}, Valid: {len(valid_smiles)} ({len(valid_smiles)/len(all_smiles):.2%})")

    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    output_list = all_smiles if args.save_invalid else valid_smiles
    pd.DataFrame(output_list, columns=['SMILES']).to_csv(args.output_path, index=False)
    print(f"Results saved to {args.output_path}")

if __name__ == '__main__':
    main()
