from typing import List, Tuple, Dict, Set, Any
import torch
from torch import nn
import torch.nn.functional as F

from torch.nn import Sequential as Linear, BatchNorm1d
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.pool import max_pool_neighbor_x
from torch.nn import Module, Dropout
from torch_geometric.nn import NNConv

def conv1d(
    in_channels: int, 
    out_channels: int, 
    kernel_size: int, 
    padding: int=0, 
    bias: bool=False):
    return nn.Sequential(
        nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            bias=bias),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2))

def conv2d(
    in_channels: int, 
    out_channels: int, 
    kernel_size: int, 
    padding: int=0, 
    bias: bool=False):
    
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            bias=bias
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=2))

class UVNetSurfaceEncoder(nn.Module):
    def __init__(self, dropout_rate:float=0.3):
        super(UVNetSurfaceEncoder, self).__init__()
        self.out_channels = 64
        in_channels = 7
        self.conv1 = conv2d(in_channels, 16, kernel_size = 3, padding=0, bias=True)
        self.conv2 = conv2d(16, 32, kernel_size = 3, padding=1, bias=True)
        self.conv3 = conv2d(32, self.out_channels, kernel_size = 3, padding=1, bias=True)
        self.dropout_rate = dropout_rate

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m:nn.Module):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x:torch.Tensor):
        x=self.conv1(x)
        x=F.relu(x)
        x=F.dropout(x, self.dropout_rate)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.dropout(x, self.dropout_rate)
        x=self.conv3(x)
        x=F.relu(x)
        x=F.dropout(x, self.dropout_rate)
        x = x.view(-1, self.out_channels)
        return x

class UVNetCurveEncoder(nn.Module):
    def __init__(self, dropout_rate:float=0.3):
        super(UVNetCurveEncoder, self).__init__()
        self.dropout_rate = dropout_rate
        in_channels = 6
        self.out_channels = 64
        self.conv1 = conv1d(in_channels, 16, kernel_size=3, padding=0, bias=True)
        self.conv2 = conv1d(16, 32, kernel_size=3, padding=1, bias=True)
        self.conv3 = conv1d(32, self.out_channels, kernel_size=3, padding=1, bias=True)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        x=self.conv1(x)
        x=F.relu(x)
        x=F.dropout(x, self.dropout_rate)
        x=self.conv2(x)
        x=F.relu(x)
        x=F.dropout(x, self.dropout_rate)
        x=self.conv3(x)
        x=F.relu(x)
        x=F.dropout(x, self.dropout_rate)
        x = x.view(-1, self.out_channels)
        return x