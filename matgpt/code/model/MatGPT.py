# codes/MatGPT.py

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class MatGPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        # Ablation flags
        self.use_rotary = kwargs.get('use_rotary', False)
        self.use_rel_pos_bias = kwargs.get('use_rel_pos_bias', False)
        self.use_gated_mlp = kwargs.get('use_gated_mlp', False)
        
        # Diversity parameters
        self.use_diversity_loss = kwargs.get('use_diversity_loss', False)
        self.diversity_weight = kwargs.get('diversity_weight', 0.1)
        self.use_nucleus_sampling = kwargs.get('use_nucleus_sampling', False)
        self.nucleus_p = kwargs.get('nucleus_p', 0.9)
        self.use_temperature_annealing = kwargs.get('use_temperature_annealing', False)
        self.min_temperature = kwargs.get('min_temperature', 0.5)
        self.max_temperature = kwargs.get('max_temperature', 1.5)
        
        for k, v in kwargs.items():
            setattr(self, k, v)


class MatGPT1Config(MatGPTConfig):
    n_layer = 12
    n_head = 12
    n_embd = 768
    block_size = 1024


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding."""
    
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Generate rotation matrix - ensure we have exactly dim/2 frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim//2).float() / (dim//2)))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # Return cos and sin separately
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        return cos, sin


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Apply rotary positional embeddings to query and key."""
    # q, k: [batch, n_head, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim//2]
    # position_ids: [seq_len]
    
    # Get the correct cos and sin for each position
    cos = cos[position_ids]  # [seq_len, head_dim//2]
    sin = sin[position_ids]  # [seq_len, head_dim//2]
    
    # Expand to match q, k dimensions
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
    
    # Apply rotary embeddings only to the first half of the head dimension
    head_dim = q.size(-1)
    half_dim = head_dim // 2
    
    q_embed = q.clone()
    k_embed = k.clone()
    
    # Apply to first half
    q_embed[..., :half_dim] = (q[..., :half_dim] * cos) + (rotate_half(q[..., :half_dim]) * sin)
    k_embed[..., :half_dim] = (k[..., :half_dim] * cos) + (rotate_half(k[..., :half_dim]) * sin)
    
    return q_embed, k_embed


def nucleus_sampling(logits, p=0.9):
    """Nucleus sampling (top-p sampling)."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('Inf')
    
    return logits


def temperature_annealing(temperature, step, total_steps, min_temp=0.5, max_temp=1.5):
    """Temperature annealing during generation."""
    # Cosine annealing schedule
    progress = step / total_steps
    annealed_temp = min_temp + 0.5 * (max_temp - min_temp) * (1 + math.cos(math.pi * progress))
    return annealed_temp


class DiversityLoss(nn.Module):
    """Diversity loss."""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, logits, targets=None):
        """
        Compute diversity loss based on output distribution entropy.
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            targets: Target tokens (optional)
            
        Returns:
            diversity_loss: Scalar diversity loss
        """
        # Compute entropy of the output distribution
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch_size, seq_len]
        
        # Average entropy across sequence length
        avg_entropy = torch.mean(entropy)
        
        # Diversity loss: encourage higher entropy (more diverse outputs)
        # We want to maximize entropy, so we minimize negative entropy
        diversity_loss = -avg_entropy
        
        return diversity_loss


class CausalSelfAttention(nn.Module):
    """Causal self-attention with rotary positional embeddings and relative position bias."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        
        # Attention parameters
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.scale = 1.0 / math.sqrt(config.n_embd // config.n_head)
        
        # Rotary positional embedding
        self.use_rotary = getattr(config, 'use_rotary', True)
        if self.use_rotary:
            self.rotary_emb = RotaryPositionalEmbedding(config.n_embd // config.n_head, config.block_size)
        self.head_dim = config.n_embd // config.n_head
        
        # Relative position bias
        self.use_rel_pos_bias = getattr(config, 'use_rel_pos_bias', True)
        if self.use_rel_pos_bias:
            self.relative_position_bias = nn.Parameter(torch.randn(2 * config.block_size - 1, config.n_head))
            self.max_seq_len = config.block_size
        
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, layer_past=None, use_causal_mask=True):
        B, T, C = x.size()
        
        # Calculate query, key, values for all heads
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        
        # Apply rotary positional embeddings
        if self.use_rotary:
            cos, sin = self.rotary_emb(x, T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin, torch.arange(T, device=x.device))
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        if self.use_rel_pos_bias:
            # Create relative position indices
            coords_h = torch.arange(T, device=x.device)
            coords_w = torch.arange(T, device=x.device)
            coords = coords_h[:, None] - coords_w[None, :]
            coords = coords + self.max_seq_len - 1  # Shift to be non-negative
            
            # Get relative position bias
            relative_position_bias = self.relative_position_bias[coords]  # [seq_len, seq_len, n_head]
            relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # [1, n_head, seq_len, seq_len]
            att = att + relative_position_bias
        
        # Apply causal mask
        if use_causal_mask:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs
        
        # Output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save


class GatedMLP(nn.Module):
    """Gated MLP for better feature interaction."""
    
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.mlp_ratio = 4
        
        # Gated linear layers
        self.fc1 = nn.Linear(config.n_embd, self.mlp_ratio * config.n_embd)
        self.fc2 = nn.Linear(config.n_embd, self.mlp_ratio * config.n_embd)
        self.fc3 = nn.Linear(self.mlp_ratio * config.n_embd, config.n_embd)
        
        self.drop = nn.Dropout(config.resid_pdrop)
        self.act = nn.GELU()
        
    def forward(self, x):
        # Gated mechanism
        gate = self.act(self.fc1(x))
        value = self.act(self.fc2(x))
        gated_value = gate * value
        out = self.fc3(gated_value)
        return self.drop(out)


class StandardMLP(nn.Module):
    """Standard MLP."""
    
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        
    def forward(self, x):
        return self.mlp(x)


class Block(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.use_gated_mlp = getattr(config, 'use_gated_mlp', True)
        if self.use_gated_mlp:
            self.mlp = GatedMLP(config)
        else:
            self.mlp = StandardMLP(config)

    def forward(self, x, attn_output=False, use_causal_mask=True):
        # Pre-norm architecture
        y, attn = self.attn(self.ln1(x), use_causal_mask=use_causal_mask)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        
        if attn_output:
            return x, attn
        else:
            return x


class MatGPT(nn.Module):
    """Materials Generative Pretrained Transformer."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Position embedding (learnable, but can be overridden by rotary)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        
        # Dropout
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm and head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Diversity components
        if getattr(config, 'use_diversity_loss', True):
            self.diversity_loss = DiversityLoss()
        
        # Initialize weights
        self.block_size = config.block_size
        self.apply(self._init_weights)
        
        print("Number of parameters in MatGPT: %.2fM" % (self.get_num_params() / 1e6))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_emb.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """Configure optimizer with weight decay."""
        # Separate parameters for weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                
                if pn.endswith('bias') or ('bias' in pn):
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add('pos_emb')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        # Add relative_position_bias to no_decay if it exists
        if 'relative_position_bias' in param_dict:
            no_decay.add('relative_position_bias')
        
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(self, idx, targets=None, attn_output=False, use_causal_mask=True, compute_diversity_loss=False):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # Forward the MatGPT model
        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)

        if attn_output:
            attn_maps = []
            for layer in self.blocks:
                x, attn = layer(x, attn_output=True, use_causal_mask=use_causal_mask)
                attn_maps.append(attn)
        else:
            for layer in self.blocks:
                x = layer(x, use_causal_mask=use_causal_mask)
        
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        diversity_loss = None
        
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
            
            # Add diversity loss if enabled
            if compute_diversity_loss and hasattr(self, 'diversity_loss'):
                diversity_loss = self.diversity_loss(logits, targets)

        if attn_output:
            return logits, loss, attn_maps, diversity_loss
        else:
            return logits, loss, diversity_loss

    def generate_with_diversity(self, start_tokens, max_length, temperature=1.0, top_k=None, 
                               nucleus_p=0.9, use_temperature_annealing=True, min_temp=0.5, max_temp=1.5):
        """
        Generate sequences using nucleus sampling and temperature annealing.
        This version uses pre-allocation, active masking, and per-sequence early stopping for speed.
        """
        self.eval()
        device = next(self.parameters()).device
        batch_size, seq_len = start_tokens.shape
        eos_token = 0  # Assuming 0 is the end token
        
        # Pre-allocate output tensor
        codes = torch.full((batch_size, max_length), eos_token, dtype=torch.long, device=device)
        codes[:, :seq_len] = start_tokens
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        current_lengths = torch.ones(batch_size, dtype=torch.long, device=device) * seq_len
        
        for step in range(seq_len, max_length):
            active = ~is_finished
            if not active.any():
                break
            inputs = codes[active, :step]
            logits, _, _ = self(inputs)
            logits = logits[:, -1, :] / temperature
            
            # Apply temperature annealing if enabled
            if use_temperature_annealing:
                current_temp = temperature_annealing(temperature, step, max_length, min_temp, max_temp)
                logits = logits / current_temp
            
            # Apply nucleus sampling if enabled
            if nucleus_p is not None:
                logits = nucleus_sampling(logits, nucleus_p)
            
            # Apply top-k sampling if specified
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Update codes and finished mask
            codes[active, step] = next_token
            new_eos_mask = (next_token == eos_token)
            is_finished[active] = new_eos_mask | is_finished[active]
            current_lengths[active] = torch.where(new_eos_mask, current_lengths[active], step + 1)
        
        # Return the generated codes (optionally trim to actual lengths)
        return codes 