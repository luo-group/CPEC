import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
import h5py


class GATModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.emb_file = h5py.File(config.emb_file)
        self.in_channels = config.in_channels
        self.hidden_channels = config.hidden_channels
        self.out_dim = config.out_dim
        self.edge_dim = config.edge_dim
        self.heads = config.heads
        self.dropout = config.dropout
        self.append_scalar_features = config.append_scalar_features
        self.num_layers = len(self.hidden_channels)
        
        self.conv1 = GATv2Conv(self.in_channels, self.hidden_channels[0], heads=self.heads[0], edge_dim=self.edge_dim, dropout=self.dropout)
        for i in range(1, self.num_layers):
            setattr(self, f'conv{i+1}', GATv2Conv(self.hidden_channels[i-1]*self.heads[i-1], self.hidden_channels[i], heads=self.heads[i], edge_dim=self.edge_dim, dropout=self.dropout))
        self.linear = nn.Linear(self.hidden_channels[-1] * self.heads[-1], self.out_dim)
        
        
    def single_pass(self, x, edge_index, edge_attr, name, batch=None):
        if batch is not None:
            embeddings = []
            for idx, i in enumerate(name):
                emb = torch.tensor(self.emb_file[i][()])
                embeddings.append(emb)
            embeddings = torch.cat(embeddings, dim=0).to(x.device).to(x.dtype)
        else:
            embeddings = torch.tensor(self.emb_file[name][()])
            if len(embeddings) < len(x):
                padding = torch.zeros((len(x) - len(embeddings), embeddings.shape[-1]))
                embeddings = torch.cat([embeddings, padding], dim=0)
            embeddings = embeddings.to(x.device).to(x.dtype)
        if self.append_scalar_features:
            x = torch.cat([x, embeddings], dim=1)
        else:
            x = embeddings

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x, inplace=True)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x, inplace=True)

        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        x = self.linear(x)

        return x
        
        
    def forward(self, X):
        anchor = self.single_pass(X.x_anchor, X.edge_index_anchor, X.edge_attr_anchor, X.name_anchor, X.x_anchor_batch if hasattr(X, 'x_anchor_batch') else None)
        pos = self.single_pass(X.x_pos, X.edge_index_pos, X.edge_attr_pos, X.name_pos, X.x_pos_batch if hasattr(X, 'x_pos_batch') else None)
        neg = self.single_pass(X.x_neg, X.edge_index_neg, X.edge_attr_neg, X.name_neg, X.x_neg_batch if hasattr(X, 'x_neg_batch') else None)

        return anchor, pos, neg


if __name__ == '__main__':
    import easydict
    
    config = easydict.EasyDict({'emb_file': '/work/jiaqi/PenLight2/data/ec-data/ec_esm1b.h5', 'in_channels': 1024, 'hidden_channels': [128, 512], 'out_dim': 128, 'edge_dim': 35, 'heads': [8,1], 'dropout': 0.5, 'append_scalar_features': False})
    
    model = GATModel(config)
    print(model)

