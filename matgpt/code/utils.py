import random
import signal
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

from vocabulary import SMILESTokenizer, read_vocabulary

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _timeout_handler(signum, frame):
    raise TimeoutError("RDKit randomization timed out")

def randomize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    atom_indices = list(range(mol.GetNumAtoms()))
    if not atom_indices:
        return smiles
    np.random.shuffle(atom_indices)

    # Set timeout for RDKit operations
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(5) # 5 seconds timeout

    try:
        new_mol = Chem.RenumberAtoms(mol, atom_indices)
        if Chem.SanitizeMol(new_mol, catchErrors=True) != 0:
            signal.alarm(0)
            return smiles
        res = Chem.MolToSmiles(new_mol, canonical=False)
        signal.alarm(0)
        return res
    except TimeoutError:
        print(f"[Warning] Timeout randomizing SMILES: {smiles}")
        return smiles
    except Exception as e:
        signal.alarm(0)
        print(f"[Warning] RDKit error during SMILES randomization: {e}")
        return smiles
    finally:
        signal.alarm(0)

def likelihood(model, seqs):
    nll_loss = nn.NLLLoss(reduction="none")
    seqs = seqs.cuda()
    outputs = model(seqs[:, :-1])
    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs
    log_probs = logits.log_softmax(dim=2)
    return nll_loss(log_probs.transpose(1, 2), seqs[:, 1:]).sum(dim=1)

def nucleus_sampling(logits, p=0.9):
    """Nucleus sampling (top-p sampling) for better diversity."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('Inf')
    return logits

def temperature_annealing(temperature, step, total_steps, min_temp=0.5, max_temp=1.5):
    """Temperature annealing for controlled diversity during generation."""
    import math
    progress = step / total_steps
    annealed_temp = min_temp + 0.5 * (max_temp - min_temp) * (1 + math.cos(math.pi * progress))
    return annealed_temp

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample_SMILES(model, voc, n_mols=100, block_size=340, temperature=1.0, top_k=10, 
                  nucleus_p=None, use_temperature_annealing=False, min_temp=0.5, max_temp=1.5,
                  use_diversity_sampling=False):
    """
    SMILES sampling for MatGPT.
    """
    model.eval()
    device = next(model.parameters()).device
    eos_token = voc['$']
    pad_token = voc['<pad>']
    start_token = voc['^']
    tokenizer = SMILESTokenizer()

    # If the caller wrapped the model in DataParallel/DistributedDataParallel,
    # unwrap for sampling to avoid scatter/gather overhead on every token step.
    actual_model = getattr(model, "module", model)

    if hasattr(actual_model, 'generate_with_diversity') and use_diversity_sampling:
        codes = actual_model.generate_with_diversity(
            start_tokens=torch.full((n_mols, 1), start_token, dtype=torch.long, device=device),
            max_length=block_size,
            temperature=temperature,
            top_k=top_k,
            nucleus_p=nucleus_p,
            use_temperature_annealing=use_temperature_annealing,
            min_temp=min_temp,
            max_temp=max_temp
        )
    else:
        # Standard sampling loop
        codes = torch.full((n_mols, block_size), pad_token, dtype=torch.long, device=device)
        codes[:, 0] = start_token
        is_finished = torch.zeros(n_mols, dtype=torch.bool, device=device)
        
        for step in range(1, block_size):
            active = ~is_finished
            if not active.any(): break

            inputs = codes[active, :step]
            outputs = actual_model(inputs)
            
            # Extract logits
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            logits = logits[:, -1, :] / temperature

            if use_temperature_annealing:
                current_temp = temperature_annealing(temperature, step, block_size, min_temp, max_temp)
                logits = logits / current_temp

            if nucleus_p is not None:
                logits = nucleus_sampling(logits, nucleus_p)

            if top_k is not None:
                logits = top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(-1)

            codes[active, step] = sampled
            is_finished[active] = (sampled == eos_token)

    # Decode sequences
    decoded = []
    for i in range(n_mols):
        seq = codes[i]
        # Find EOS
        eos_indices = (seq == eos_token).nonzero(as_tuple=True)[0]
        if len(eos_indices) > 0:
            seq = seq[:eos_indices[0] + 1]
        
        token_ids = seq.cpu().numpy()
        try:
            tokens = [voc.idx2token[idx] for idx in token_ids]
            decoded.append(tokenizer.untokenize(tokens))
        except KeyError:
            decoded.append("")

    return decoded, codes, None # NLLs removed for simplicity if not used

def evaluate_smiles_validity(smiles_list):
    valid_smiles = []
    for smi in smiles_list:
        if is_valid_smiles(smi):
            valid_smiles.append(smi)
    valid_ratio = len(valid_smiles) / len(smiles_list) if smiles_list else 0
    return valid_ratio, valid_smiles

def model_validity(model, vocab_path=None, vocab=None, n_mols=100, block_size=340):
    """
    Calculate model validity.
    
    Args:
        model: The model to evaluate
        vocab_path: Path to vocabulary file (optional if vocab is provided)
        vocab: Vocabulary object (optional if vocab_path is provided)
        n_mols: Number of molecules to generate
        block_size: Block size for generation
    """
    if vocab is not None:
        voc = vocab
    elif vocab_path is not None:
        voc = read_vocabulary(vocab_path)
    else:
        raise ValueError("Either vocab_path or vocab must be provided")
    print("sample")
    smiles, _, _ = sample_SMILES(model, voc, n_mols=n_mols, block_size=block_size)
    print("evaluate")
    valid_ratio, _ = evaluate_smiles_validity(smiles)
    return valid_ratio

def calc_fingerprints(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    # Filter out None mols
    valid_indices = [i for i, m in enumerate(mols) if m is not None]
    valid_mols = [mols[i] for i in valid_indices]
    
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, radius=2, nBits=2048) for x in valid_mols]
    smiles_canonical = [Chem.MolToSmiles(x, isomericSmiles=False) for x in valid_mols]
    
    # Return same length as input if possible, or just the valid ones?
    # Original code returned based on valid mols.
    return fps, smiles_canonical

def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)

def int_div(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list if s]
    mols = [m for m in mols if m]
    if len(mols) < 2: return 0.0
    
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols]
    similarity_matrix = []
    for i in range(len(fps)):
        similarities = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        similarity_matrix.extend(similarities)
    
    return 1 - np.mean(similarity_matrix) if similarity_matrix else 0.0

def filter_valid_smiles(df, smiles_columns):
    valid_rows = df.copy()
    for col in smiles_columns:
        if col in valid_rows.columns:
            valid_rows = valid_rows[valid_rows[col].apply(is_valid_smiles)]
    return valid_rows

def is_valid_smiles(smiles: str) -> bool:
    if not smiles or not isinstance(smiles, str): return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and mol.GetNumAtoms() > 0

def freeze_parameters(model, config):
    """Freeze model parameters based on configuration."""
    for name, param in model.named_parameters():
        # Default: trainable
        param.requires_grad = True
        
        # Freezing logic
        parts = name.split('.')
        layer_num = None
        if 'blocks' in parts:
            idx = parts.index('blocks')
            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                layer_num = int(parts[idx + 1])

        # Freeze early layers, tok_emb, pos_emb
        if (layer_num is not None and layer_num < config.get('freeze_layers', 0)) \
           or 'tok_emb' in name \
           or 'pos_emb' in name:
            param.requires_grad = False

        # Force trainable if in keep_train_modules
        if any(m in name for m in config.get('keep_train_modules', [])):
            param.requires_grad = True

        # Force frozen if in freeze_modules
        if any(m in name for m in config.get('freeze_modules', [])):
            param.requires_grad = False
    
    print("Model parameters frozen according to config.")

def make_collate_fn(max_length):
    """Collate function that drops sequences longer than max_length."""
    def collate_fn(encoded_seqs):
        # Filter out sequences that exceed the allowed length
        filtered = [seq for seq in encoded_seqs if len(seq[0]) <= max_length]
        # If all sequences were filtered out, fall back to keeping the shortest one
        if not filtered:
            shortest = min(encoded_seqs, key=lambda s: len(s[0]))
            filtered = [shortest]
        max_len = max(len(seq[0]) for seq in filtered)
        collated_arr_x = torch.zeros(len(filtered), max_len, dtype=torch.long)
        collated_arr_y = torch.zeros(len(filtered), max_len, dtype=torch.long)
        for i, seq in enumerate(filtered):
            collated_arr_x[i, :len(seq[0])] = torch.as_tensor(seq[0], dtype=torch.long)
            collated_arr_y[i, :len(seq[1])] = torch.as_tensor(seq[1], dtype=torch.long)
        return collated_arr_x, collated_arr_y
    return collate_fn