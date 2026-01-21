import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.utils import to_dense_batch

class MOE2(nn.Module):
    """Molecular Orbital Energy Estimator (MOE2)"""

    def __init__(self, in_channels, edge_dim, hidden_channels, out_channels,mlm_output_dim=9,regression_targets=2,heads=8, dropout_rate=0.2):
        super(MOE2, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=edge_dim, dropout=dropout_rate)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=edge_dim, dropout=dropout_rate)
        self.conv3 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, edge_dim=edge_dim, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.regression_head = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, regression_targets)
        )

        self.pool = AttentionalAggregation(nn.Sequential(
                nn.Linear(out_channels, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 1)
            ))
        self.mlm_head = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, mlm_output_dim)
        )

    def forward(self, x, edge_index, edge_attr, batch, task_type="mlm", batch_size=None):
        x = self.relu(self.conv1(x, edge_index, edge_attr))
        x = self.relu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)

        if task_type == "mlm":
            return self.mlm_head(x)  # [num_nodes, mlm_output_dim]
        elif task_type == "homo_lumo":
            graph_emb = self.pool(x, batch, dim_size=batch_size)
            return self.regression_head(graph_emb)  # [batch_size, 2]
        elif task_type == "embed":
            graph_emb = self.pool(x, batch, dim_size=batch_size)
            return graph_emb
        elif task_type == "embed_node":
            return x
        else:
            raise ValueError("Invalid task type")


class CrossGraphAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout) 
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, src, tgt, tgt_mask=None, src_mask=None):

        kpm = (~tgt_mask) if tgt_mask is not None else None
        m = src_mask.unsqueeze(-1) if src_mask is not None else None  # [B, Nq, 1]

        attn_output, _ = self.cross_attn(src, tgt, tgt, key_padding_mask=kpm)
        attn_output = self.dropout(attn_output)

        if m is not None:
            attn_output = attn_output * m
            src = src * m                      

        src = self.norm1(src + attn_output)

        ffn_output = self.ffn(src)
        ffn_output = self.dropout(ffn_output)
        if m is not None:
            ffn_output = ffn_output * m

        src = self.norm2(src + ffn_output)

        if m is not None:
            src = src * m                      
        return src



def dense_to_sparse(x, mask):
    B, N, D = x.size()
    x_out = x[mask]
    batch = torch.arange(B, device=x.device).unsqueeze(1).expand(B, N)[mask]
    return x_out, batch


class P3(nn.Module):
    """Photovoltaic Performance Predictor Node Level (P3NodeLevel)"""
    def __init__(self, in_channels, edge_dim, hidden_channels, out_channels,
                 homolumo_targets, pce_targets, dropout_rate=0.2, num_heads=8):
        super().__init__()
        self.donor_encoder = MOE2(
            in_channels, edge_dim, hidden_channels, out_channels,
            regression_targets=homolumo_targets
        )
        self.acceptor_encoder = MOE2(
            in_channels, edge_dim, hidden_channels, out_channels,
            regression_targets=homolumo_targets
        )

        self.cross1 = CrossGraphAttention(out_channels, num_heads, dropout_rate)
        self.cross2 = CrossGraphAttention(out_channels, num_heads, dropout_rate)

        self.pool = AttentionalAggregation(nn.Sequential(
            nn.Linear(out_channels, 128), nn.ReLU(), nn.Linear(128, 1)
        ))

        # Normalization layers for feature fusion
        self.norm_graph = nn.LayerNorm(out_channels)
        self.norm_attn = nn.LayerNorm(out_channels)
        
        # Project HOMO/LUMO to higher dimension and normalize
        self.homolumo_projector = nn.Sequential(
            nn.Linear(homolumo_targets, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )

        self.regression_head = nn.Sequential(
            # Input dim: 
            # Graph: out_channels * 2 (donor + acceptor)
            # Attn: out_channels * 2
            # HOMO/LUMO: 64 * 2 (projected)
            nn.Linear(out_channels * 4 + 64 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, pce_targets)
        )

    def forward(self, donor, acceptor):
        batch_size = donor.num_graphs

        # Donor
        d_node_emb = self.donor_encoder(donor.x, donor.edge_index, donor.edge_attr, donor.batch, task_type="embed_node")
        # Acceptor
        a_node_emb = self.acceptor_encoder(acceptor.x, acceptor.edge_index, acceptor.edge_attr, acceptor.batch, task_type="embed_node")
        
        # Graph Embedding (Global)
        d_graph_emb = self.donor_encoder.pool(d_node_emb, donor.batch, dim_size=batch_size)
        a_graph_emb = self.acceptor_encoder.pool(a_node_emb, acceptor.batch, dim_size=batch_size)
        
        # HOMO/LUMO (Physics)
        d_homolumo = self.donor_encoder.regression_head(d_graph_emb)
        a_homolumo = self.acceptor_encoder.regression_head(a_graph_emb)

        d_dense, d_mask = to_dense_batch(d_node_emb, donor.batch, batch_size=batch_size)
        a_dense, a_mask = to_dense_batch(a_node_emb, acceptor.batch, batch_size=batch_size)

        d_old, a_old = d_dense, a_dense
        d_dense = self.cross1(d_old, a_old, a_mask, d_mask)
        a_dense = self.cross2(a_old, d_old, d_mask, a_mask)
        
        # Dense → Sparse
        d_attn_out, d_batch = dense_to_sparse(d_dense, d_mask)
        a_attn_out, a_batch = dense_to_sparse(a_dense, a_mask)

        d_attn_graph = self.pool(d_attn_out, d_batch, dim_size=batch_size)
        a_attn_graph = self.pool(a_attn_out, a_batch, dim_size=batch_size)

        d_graph_emb = self.norm_graph(d_graph_emb)
        a_graph_emb = self.norm_graph(a_graph_emb)
        
        d_attn_graph = self.norm_attn(d_attn_graph)
        a_attn_graph = self.norm_attn(a_attn_graph)
        
        d_homolumo = self.homolumo_projector(d_homolumo)
        a_homolumo = self.homolumo_projector(a_homolumo)

        z = torch.cat([d_graph_emb, a_graph_emb, d_attn_graph, a_attn_graph, d_homolumo, a_homolumo], dim=-1)
        return self.regression_head(z)
