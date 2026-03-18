import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Tuple
from torch import Tensor

# @torch.jit.script
def softmax_dropout(input, dropout_prob: float, is_training: bool):
    return F.dropout(F.softmax(input, -1), dropout_prob, is_training)

class SelfMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=True, scaling_factor=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query: Tensor, attn_bias: Tensor = None) -> Tensor:
        n_node, n_graph, embed_dim = query.size()
        q, k, v = self.in_proj(query).chunk(3, dim=-1)

        _shape = (-1, n_graph * self.num_heads, self.head_dim)
        q = q.contiguous().view(_shape).transpose(0, 1) * self.scaling
        k = k.contiguous().view(_shape).transpose(0, 1)
        v = v.contiguous().view(_shape).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        if attn_bias is not None:
            attn_weights += attn_bias
        attn_probs = softmax_dropout(attn_weights, self.dropout, self.training)

        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(n_node, n_graph, embed_dim)
        return self.out_proj(attn)

class Graphormer3DEncoderLayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim, num_heads, dropout=0.1, attn_dropout=0.1, act_dropout=0.1):
        super().__init__()
        self.self_attn = SelfMultiheadAttention(embed_dim, num_heads, attn_dropout)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = dropout
        self.act_dropout = act_dropout

    def forward(self, x: Tensor, attn_bias: Tensor = None):
        residual = x
        x = self.attn_norm(x)
        x = self.self_attn(x, attn_bias)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        residual = x
        x = self.ffn_norm(x)
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=self.act_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return residual + x

# @torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2 * pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

class GaussianLayer(nn.Module):
    def __init__(self, K=128, edge_types=132 * 132):
        super().__init__()
        self.K = K
        self.edge_types = edge_types
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means.weight, 0, 5)
        nn.init.uniform_(self.stds.weight, 0, 5)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-5
        return gaussian(x.float(), mean, std).type_as(self.means.weight)

class NodeTaskHead(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.scaling = (embed_dim // num_heads) ** -0.5
        self.force_proj1 = nn.Linear(embed_dim, 1)
        self.force_proj2 = nn.Linear(embed_dim, 1)
        self.force_proj3 = nn.Linear(embed_dim, 1)

    def forward(self, query, attn_bias, delta_pos):
        bsz, n_node, _ = query.size()
        q = self.q_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2) * self.scaling
        k = self.k_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(query).view(bsz, n_node, self.num_heads, -1).transpose(1, 2)

        attn = q @ k.transpose(-1, -2)
        attn_probs = softmax_dropout(attn.view(-1, n_node, n_node) + attn_bias, 0.1, self.training)
        attn_probs = attn_probs.view(bsz, self.num_heads, n_node, n_node)

        rot_attn_probs = attn_probs.unsqueeze(-1) * delta_pos.unsqueeze(1).type_as(attn_probs)
        rot_attn_probs = rot_attn_probs.permute(0, 1, 4, 2, 3)
        x = rot_attn_probs @ v.unsqueeze(2)
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(bsz, n_node, 3, -1)

        f1 = self.force_proj1(x[:, :, 0, :]).squeeze(-1)
        f2 = self.force_proj2(x[:, :, 1, :]).squeeze(-1)
        f3 = self.force_proj3(x[:, :, 2, :]).squeeze(-1)
        return torch.stack([f1, f2, f3], dim=-1)

class Graphormer3D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.atom_types = 132
        self.edge_types = self.atom_types * self.atom_types
        self.tags_num = 2
        
        # Embeddings
        self.atom_encoder = nn.Embedding(self.atom_types, config.embed_dim, padding_idx=0)
        self.tag_encoder = nn.Embedding(self.tags_num, config.embed_dim)

        # Gaussian Features
        self.gbf = GaussianLayer(config.num_kernel, self.edge_types)
        self.bias_proj = nn.Linear(config.num_kernel, config.num_heads)
        self.edge_proj = nn.Linear(config.num_kernel, config.embed_dim)

        # Encoder Layers
        self.layers = nn.ModuleList([
            Graphormer3DEncoderLayer(
                embed_dim=config.embed_dim,
                ffn_dim=config.ffn_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
                act_dropout=config.act_dropout
            ) for _ in range(config.num_layers)
        ])

        # Output Heads
        self.final_ln = nn.LayerNorm(config.embed_dim)
        self.rmsd_proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, 1)
        )
        self.rmsd_weights = nn.Embedding(3, 1)
        nn.init.normal_(self.rmsd_weights.weight, 0, 0.01)
        self.node_head = NodeTaskHead(config.embed_dim, config.num_heads)

    def forward(self, atoms, tags, pos, real_mask):
        # Geometry Features
        n_graph, n_node = atoms.size()
        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1)
        dir_vec = delta_pos / (dist.unsqueeze(-1) + 1e-5)

        # Edge Features
        edge_type = atoms.view(n_graph, n_node, 1) * self.atom_types + atoms.view(
            n_graph, 1, n_node)

        gbf_feat = self.gbf(dist, edge_type)

        # Node Features
        node_feat = self.atom_encoder(atoms) + self.tag_encoder(tags)
        edge_feat = self.edge_proj(gbf_feat.sum(dim=2))
        node_feat += edge_feat

        # Attention Bias
        attn_bias = self.bias_proj(gbf_feat).permute(0, 3, 1, 2)
        attn_bias = attn_bias.reshape(-1, n_node, n_node)

        # Transformer Encoder
        x = node_feat.transpose(0, 1)
        for layer in self.layers:
            x = layer(x, attn_bias)
        x = self.final_ln(x).transpose(0, 1)

        # rmsd Prediction
        pred_rmsd = self.rmsd_proj(x) * self.rmsd_weights(tags)
        pred_rmsd = pred_rmsd.masked_fill(~real_mask.unsqueeze(-1), 0).sum(dim=1)

        # Node Displacements
        node_displacements = self.node_head(x, attn_bias, dir_vec)
        return pred_rmsd, node_displacements