"""
PHASE 1 - Single molecule prediction
Loads Zhangbin's trained GCN model and predicts Active/Inactive for one SMILES string.
"""

import sys
import torch
import torch.nn.functional as F

# ---- Import FPPool featurizer (from the cloned FPPool repo) ----
sys.path.insert(0, "./FPPool")
from fppcode.feature.featurizer import GenNodeEdgeFeatures39_WithFP

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from rdkit import Chem

# ---- Re-define the EXACT same model architecture used in training ----
# (copied from train_baseline_gnns.py - BaselineGNN class)
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GINEConv, global_mean_pool, global_add_pool, global_max_pool


def get_pooling(pooling):
    if pooling == "mean":
        return global_mean_pool
    elif pooling == "sum":
        return global_add_pool
    elif pooling == "max":
        return global_max_pool
    raise ValueError(f"Unsupported pooling: {pooling}")


class BaselineGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=128, num_layers=3,
                 dropout=0.2, gnn_type="GCN", pooling="mean", num_classes=2):
        super().__init__()
        self.gnn_type = gnn_type.upper()
        self.pool = get_pooling(pooling)
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        if self.gnn_type == "GCN":
            self.convs.append(GCNConv(node_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
        else:
            raise ValueError("This script currently only loads GCN (best_model.pt is GCN)")

        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_embedding = self.pool(x, batch)
        logits = self.classifier(graph_embedding)
        return logits


def smiles_to_fppool_graph(smiles, fp_types=["RdkitFP"]):
    """Convert a single SMILES string into the graph format FPPool/the model expects."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    data = Data()
    data.smiles = canonical
    data.y = torch.tensor([0], dtype=torch.long)  # dummy label, unused at inference

    featurizer = GenNodeEdgeFeatures39_WithFP(fp_types=fp_types)
    data = featurizer(data)

    data.x = data.x.float()
    data.edge_attr = data.edge_attr.float()
    data.fp = data.fp.bool()

    return data


def load_model(model_path="best_model.pt", device="cpu"):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = BaselineGNN(
        node_dim=checkpoint["node_dim"],
        edge_dim=checkpoint["edge_dim"],
        gnn_type="GCN",
        pooling="mean",
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def predict_smiles(smiles, model, device="cpu"):
    graph = smiles_to_fppool_graph(smiles)

    # DataLoader wraps a single graph into a "batch of 1" - the model expects batched input
    loader = DataLoader([graph], batch_size=1)
    batch = next(iter(loader)).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)[0]

    inactive_prob = probs[0].item()
    active_prob = probs[1].item()
    predicted_class = int(active_prob > inactive_prob)

    return {
        "smiles": smiles,
        "prediction": "Active" if predicted_class == 1 else "Inactive",
        "predicted_class": predicted_class,
        "inactive_probability": round(inactive_prob, 4),
        "active_probability": round(active_prob, 4),
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = load_model("best_model.pt", device=device)
    print("Model loaded!\n")

    # Test molecules - aspirin and ethanol, just to see the pipeline run end to end
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CCO",                     # Ethanol
    ]

    for smiles in test_smiles:
        result = predict_smiles(smiles, model, device=device)
        print(result)