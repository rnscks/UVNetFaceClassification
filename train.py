import torch
import torch_geometric
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import pandas as pd
import torch.nn.functional as F
import torch_geometric.loader
from dataset.dataset import GraphDataset
from model.uv_net import UVNet
from model.gnn import GCN, SAGE, GAT
from torch_geometric.data import Data   

from dataset.preprocess import PreprocessUVGraph 

def train(
    model:nn.Module, 
    loader:torch_geometric.loader.DataLoader, 
    optimizer:torch.optim.Optimizer, 
    criterion:torch.nn.CrossEntropyLoss, 
    device: torch.device):
    model.train()

    for data in loader:
        data: Data = data.to(device)    
        out:torch.Tensor = model(data)  # [num_node, num_classes]
        
        label:torch.Tensor = data.y.long()
        loss:torch.Tensor = criterion(out, label)
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    return

def evaluate(model:nn.Module, 
             loader:torch_geometric.loader.DataLoader, 
             criterion:torch.nn.CrossEntropyLoss, 
             device:torch.device):
    model.eval()
    correct:int = 0
    total = 0
    total_loss:float = 0.0    
    for data in loader:
        data:Data = data.to(device)
        with torch.no_grad():
            out = model(data)
            preds= F.softmax(out, dim=-1)
            _, max_preds = torch.max(preds, 1)
            loss:torch.Tensor = criterion(out, data.y.long())
        correct += (max_preds == data.y).sum().item()
        total += data.y.size(0)
        total_loss += loss.item()
    acc = correct / total * 100
    total_loss /= len(loader)
    return acc, total_loss 

def train_uvnet(
    train_dataset:GraphDataset,
    val_dataset:GraphDataset ,
    result_path: str, 
    model_path: str = None, 
    gnn: nn.Module = None):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 1. 모델 정의
    # MFACD++ 라벨 수 25
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UVNet(num_classes=25, dropout=0.0, gnn_module=gnn).to(device)

    # 2. 손실 함수 및 옵티마이저 정의
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)


    # 3. 에폭 단위 반복
    acc_history = []
    loss_history = []   
    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, optimizer, criterion, device)
        acc, loss = evaluate(model, val_loader, criterion, device) 
        scheduler.step(loss)  # Update learning rate based on loss
        acc_history.append(acc) 
        loss_history.append(loss)   
        print(f"[Epoch {epoch}] Loss: {loss:.4f}, Accuracy: {acc:.4f} %")
        
    torch.save(model.state_dict(), model_path)   
    df = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        'loss': loss_history,
        'accuracy': acc_history})  
    df.to_excel(result_path, index=False)   


if __name__ == "__main__":
    
    stp_file_path = 'data/step'
    save_file_path = 'data/labeled_graph'
    
    PreprocessUVGraph.preprocss(
        stp_dir_path=stp_file_path,
        save_dir_path=save_file_path
    )
    dataset = GraphDataset('data/labeled_graph')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    print(f"Train size: {train_size}, Validation size: {val_size}")
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_uvnet(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        result_path='GCN.xlsx',
        model_path='uvnet_gcn.pth',
        gnn=GCN())
    train_uvnet(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        result_path='SAGE.xlsx',
        model_path='uvnet_sage.pth',
        gnn=SAGE()) 
    train_uvnet(    
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        result_path='GAT.xlsx',
        model_path='uvnet_gat.pth',
        gnn=GAT())