import torch
import numpy as np
from tqdm import tqdm

from utils import randomize_smiles

class Dataset(torch.utils.data.Dataset):
    # Custom PyTorch Dataset for SMILES

    def __init__(self, smiles_list, vocabulary, tokenizer, aug_prob=0, preprocess=False):
        self._vocabulary = vocabulary
        self._tokenizer = tokenizer
        self._smiles_list = list(smiles_list)
        self._aug_prob = aug_prob
        
        if preprocess: # preprocess - remove the smiles with unknown tokens
            remove_list = []
            for s in tqdm(self._smiles_list):
                tokens = self._tokenizer.tokenize(s)
                encoded = self._vocabulary.encode(tokens)
                if encoded[0] == -1:
                    remove_list.append(s)
            for s in tqdm(remove_list):
                self._smiles_list.remove(s)

    def __getitem__(self, i):
        smi = self._smiles_list[i]

        p = np.random.uniform() # data augmentation
        # Skip randomization for very long SMILES (>300 chars) as they can hang RDKit
        if p < self._aug_prob and len(smi) <= 300:
            try:
                smi = randomize_smiles(smi)
            except Exception as e:
                print(f"[ERROR] Randomization failed for idx {i} ({smi}): {e}", flush=True)
                # Fallback to original SMILES on error
                smi = self._smiles_list[i]

        try:
            tokens = self._tokenizer.tokenize(smi)
            encoded = self._vocabulary.encode(tokens)
        except Exception as e:
            print(f"[ERROR] Tokenization failed for idx {i} ({smi}): {e}", flush=True)
            raise e
            
        return encoded[:-1], encoded[1:]

    def __len__(self):
        return len(self._smiles_list)