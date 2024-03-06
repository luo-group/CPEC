import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.input_dim = config.input_dim
        self.hidden_dims = config.hidden_dims
        self.out_dim = config.out_dim
        self.dropout = config.dropout
        self.num_layers = len(self.hidden_dims) + 1
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(self.input_dim, self.hidden_dims[0]), 
                                         nn.ReLU(), 
                                         nn.Dropout(self.dropout)))
        for i in range(1, len(self.hidden_dims)):
            self.layers.append(nn.Sequential(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]),
                                             nn.ReLU(),
                                             nn.Dropout(self.dropout)))
        self.layers.append(nn.Linear(self.hidden_dims[-1], self.out_dim))
        
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
        return x
    
if __name__ == '__main__':
    from easydict import EasyDict
    config = EasyDict({'input_dim': 1280, 'hidden_dims': [1000, 512], 'out_dim': 377, 'dropout': 0.5})
    
    model = MLPModel(config)
    model.to('cuda:0')
    print(model)
    x = torch.randn(8, 1280).to('cuda:0')
    y = model(x)
    print(y.shape)
    



