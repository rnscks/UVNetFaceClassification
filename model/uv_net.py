import torch
from torch import nn
import torch.nn.functional as F 
from model.gnn import GCN, SAGE, GAT

from model.uv_encoder import UVNetSurfaceEncoder, UVNetCurveEncoder
from torch_geometric.data import Data


class UVNet(nn.Module):
    def __init__(
        self,
        num_classes,
        dropout=0.0,
        gnn_module: nn.Module = GCN()):
        super().__init__()
        self.crv_encoder = UVNetCurveEncoder(
            dropout_rate=dropout)
        self.srf_encoder = UVNetSurfaceEncoder(
            dropout_rate=dropout)
        
        self.gnn = gnn_module
        self.clf = nn.Linear(64, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        crv_feat = self.crv_encoder(data.edge_attr)
        srf_feat = self.srf_encoder(data.x)
        data.x = srf_feat
        data.edge_attr = crv_feat 
        
        x = self.gnn(data)  
        return self.clf(x)  


if __name__ == "__main__":  
    uvnet = UVNet(num_classes=10, dropout=0.3) 
    data = torch.load("data/mfacd/labeled_graph/0-0-0-0-0-23.pt")
    data = uvnet(data)