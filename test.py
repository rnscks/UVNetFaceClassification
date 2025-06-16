import torch
import torch.nn.functional as F

from model.uv_net import UVNet
from model.gnn import GCN, SAGE, GAT

from dataset.load_stp_file import STEPFileLoader
from dataset.preprocess import PreprocessUVGraph
from display.display_stp_model import display_labeled_face


def test(stp_model_path: str = '', 
         model_path: str = 'data/trained_model/uvnet_sage.pth',
         gnn_module:torch.nn.Module=GAT()):
    stp_model = STEPFileLoader(stp_model_path).load_topods_shape()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    uv_net = UVNet(
        num_classes=25,
        dropout=0.0,
        gnn_module=gnn_module).to(device)
    uv_net.load_state_dict(
        torch.load(model_path, map_location=device))

    uv_net.eval()
    uv_graph, labeled_face = PreprocessUVGraph.generate_uv_graph(
        shape=stp_model)
    out = uv_net(uv_graph)  
    
    preds = F.softmax(out, dim=-1)  
    _, max_preds = torch.max(preds, 1)  # [num_faces, num_classes]
    for idx, face in enumerate(labeled_face.keys()):
        labeled_face[face] = max_preds[idx].item()   
    print(f"Number of labeled faces: {len(labeled_face)}")
    display_labeled_face(
        labeled_faces=labeled_face,
        props_file_path='data/LABEL_PROPS.json',
        with_label_name=False)
    
def test_labeled(stp_model_path: str = ''):
    labeled_face, shape = STEPFileLoader(stp_model_path).load_labeled_topods_face()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    uv_net = UVNet(
        num_classes=25,
        dropout=0.0,
        gnn_module=SAGE()).to(device)
    uv_net.load_state_dict(
        torch.load("data/trained_model/uvnet_sage.pth", map_location=device))

    uv_net.eval()
    uv_graph, labeled_face = PreprocessUVGraph.generate_uv_graph(
        shape=shape,
        labeled_face=labeled_face)
    out = uv_net(uv_graph)  
    preds = torch.argmax(out, dim=1)
    
    for idx, face in enumerate(labeled_face.keys()):
        labeled_face[face] = preds[idx].item()   
        
    acc = torch.sum(
        torch.tensor(
            [labeled_face[face] == uv_graph.y[idx].item() for idx, face in enumerate(labeled_face.keys())])
    ) / len(labeled_face)
    print(f"Accuracy: {acc * 100:.2f}%")    
    
    display_labeled_face(
        labeled_faces=labeled_face,
        props_file_path='data/LABEL_PROPS.json',
        with_label_name=True)
    
if __name__ == "__main__":
    test(
        stp_model_path='data/demo/original/현대위아/SADDLE.stp',
        model_path='data/trained_model/uvnet_gat.pth',
        gnn_module=GAT())