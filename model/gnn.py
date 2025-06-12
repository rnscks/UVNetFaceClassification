from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, NNConv, GATConv
import torch.nn.functional as F
from torch_geometric.data import Data   

class SAGE(nn.Module):
    def __init__(self, hidden_size=32, dropout_rate=0.0):
        super(SAGE, self).__init__()
        self.hidden_size = hidden_size
        self.sage1 = SAGEConv(64, hidden_size)
        self.sage2 = SAGEConv(hidden_size, 64)

        self.dropout_rate = dropout_rate

    def forward(self, data: Data):
        x = self.sage1(
            x=data.x, 
            edge_index=data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate)
        x = self.sage2(
            x=x, 
            edge_index=data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate)
        return x

class GCN(nn.Module):
    def __init__(self, hidden_size: int=64, dropout_rate: float=0.0):
        super(GCN, self).__init__()
        self.hidden_size = hidden_size
        self.gcn1 = GCNConv(64, hidden_size)
        self.gcn2 = GCNConv(hidden_size, 64)
        self.dropout_rate = dropout_rate


    def forward(self, data: Data):        
        x = self.gcn1(
            x=data.x, 
            edge_index=data.edge_index)
        x = F.dropout(F.relu(x), self.dropout_rate)
        
        x = self.gcn2(
            x=x, 
            edge_index=data.edge_index)
        x = F.dropout(F.relu(x), self.dropout_rate)
        return x

class GAT(nn.Module):   
    def __init__(self, hidden_size: int=64, dropout_rate: float=0.0):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.gat1 = GATConv(64, hidden_size, edge_dim=64, heads=16, concat=False)
        self.gat2 = GATConv(hidden_size, 64, edge_dim=64, heads=16, concat=False)
        self.dropout_rate = dropout_rate


    def forward(self, data: Data):
        x = self.gat1(
            x=data.x, 
            edge_index=data.edge_index,
            edge_attr=data.edge_attr)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate)
        x = self.gat2(
            x=x, 
            edge_index=data.edge_index,
            edge_attr=data.edge_attr)
        x = F.relu(x)
        x = F.dropout(x, self.dropout_rate)
        return x    