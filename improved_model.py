import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool


class AdvancedBindingModel(nn.Module):
    """
    More sophisticated version with proper multi-head attention
    """
    def __init__(self, 
                 ligand_feat_dim=9,
                 protein_feat_dim=21,
                 gat_hidden=128,
                 gat_heads=4,
                 num_heads=8,
                 d_model=256,
                 dropout=0.2):
        super().__init__()
        
        # Ligand encoder
        self.gat1 = GATConv(ligand_feat_dim, gat_hidden, heads=gat_heads, dropout=dropout)
        self.gat2 = GATConv(gat_hidden * gat_heads, d_model, heads=1, dropout=dropout)
        
        # Protein encoder
        self.protein_encoder = nn.Sequential(
            nn.Conv1d(protein_feat_dim, 256, kernel_size=8, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Multi-head cross-attention
        self.cross_attn_lp = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_pl = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        '''# Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(2 * d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )'''
        self.fc = nn.Linear(2* d_model, 1)
    
    def forward(self, ligand_graph, protein_seq):
        # Encode ligand
        h_lig = F.relu(self.gat1(ligand_graph.x, ligand_graph.edge_index))
        h_lig = self.gat2(h_lig, ligand_graph.edge_index)
        h_lig_pooled = global_mean_pool(h_lig, ligand_graph.batch).unsqueeze(1)  # [B, 1, d_model]
        
        # Encode protein
        h_prot = self.protein_encoder(protein_seq)  # [B, d_model, L]
        h_prot = h_prot.transpose(1, 2)  # [B, L, d_model]
        
        # Cross-attention: ligand attends to protein
        h_lig_att, _ = self.cross_attn_lp(h_lig_pooled, h_prot, h_prot)
        h_lig_att = self.norm1(h_lig_att + h_lig_pooled).squeeze(1)  # [B, d_model]
        
        # Cross-attention: protein attends to ligand
        h_prot_pooled = h_prot.mean(dim=1, keepdim=True)  # [B, 1, d_model]
        h_prot_att, _ = self.cross_attn_pl(h_prot_pooled, h_lig_pooled, h_lig_pooled)
        h_prot_att = self.norm2(h_prot_att + h_prot_pooled).squeeze(1)  # [B, d_model]
        
        # Combine and predict
        h_combined = torch.cat([h_lig_att, h_prot_att], dim=1)
        affinity = self.fc(h_combined)
        
        return affinity