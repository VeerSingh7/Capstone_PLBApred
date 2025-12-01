import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops

# Amino acid encoding
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)} # 1-based indexing, 0 for padding

def protein_to_sequence(sequence, max_len=1000):
    """Convert protein sequence to integer indices"""
    seq_indices = [AA_TO_INT.get(aa, 0) for aa in sequence]
    if len(seq_indices) < max_len:
        seq_indices += [0] * (max_len - len(seq_indices))
    else:
        seq_indices = seq_indices[:max_len]
    
    # One-hot encode: (max_len, 21) -> Transpose to (21, max_len) for Conv1d
    # Actually, Conv1d expects (batch, channels, length). 
    # If we pass indices to an Embedding layer, it's different.
    # But BaselineGNN uses Conv1d directly on inputs? 
    # self.protein_cnn = nn.Conv1d(21, 256, kernel_size=8)
    # This implies input should be one-hot encoded with 21 channels.
    
    one_hot = torch.zeros(21, max_len)
    for i, idx in enumerate(seq_indices):
        one_hot[idx, i] = 1.0
    return one_hot

def smiles_to_graph(smiles):
    """Convert SMILES to PyG Data object with 9 atom features"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node features (Atom features) - 9 features
    # 1. Atomic number (one-hot for common atoms in drugs? Or just atomic num?)
    # BaselineGNN expects 9 channels. Let's define 9 features.
    # Example: C, N, O, F, P, S, Cl, Br, Other (One-hot 9 dims)
    # Or: AtomicNum, Degree, FormalCharge, Hybridization, Aromaticity, Mass, etc.
    # Let's use a simple one-hot encoding for common atoms + some properties to get 9.
    
    atom_types = [6, 7, 8, 9, 15, 16, 17, 35] # C, N, O, F, P, S, Cl, Br
    
    features = []
    for atom in mol.GetAtoms():
        # One-hot for atom type (8 dims)
        atom_num = atom.GetAtomicNum()
        one_hot = [1 if atom_num == x else 0 for x in atom_types]
        if sum(one_hot) == 0:
            one_hot = [0] * 8 # "Other" implicitly all zeros or we need a 9th bin?
            # Let's make the 9th feature "Is Aromatic"
        
        is_aromatic = 1 if atom.GetIsAromatic() else 0
        feats = one_hot + [is_aromatic] # 8 + 1 = 9 features
        features.append(feats)
        
    x = torch.tensor(features, dtype=torch.float)
    
    # Edges
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        
    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
    return Data(x=x, edge_index=edge_index)

class KIBADataset(Dataset):
    def __init__(self, csv_file, mode='train'):
        self.df = pd.read_csv(csv_file)
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        smiles = row['Drug']
        seq = row['Target']
        y = row['Y']
        
        graph = smiles_to_graph(smiles)
        if graph is None:
            # Handle invalid SMILES - return dummy or skip
            # For simplicity, return a dummy graph
            graph = Data(x=torch.zeros(1, 9), edge_index=torch.empty((2, 0), dtype=torch.long))
            
        prot_feat = protein_to_sequence(seq)
        
        return {
            'ligand': graph,
            'protein': prot_feat,
            'affinity': torch.tensor(y, dtype=torch.float).view(1)
        }

def collate_fn(batch):
    ligands = [item['ligand'] for item in batch]
    proteins = torch.stack([item['protein'] for item in batch])
    affinities = torch.stack([item['affinity'] for item in batch])
    
    ligand_batch = Batch.from_data_list(ligands)
    
    return Batch(
        ligand=ligand_batch,
        protein=proteins,
        affinity=affinities
    )
