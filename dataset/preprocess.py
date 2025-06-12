# 라벨과 shape(occwl Solid)를 활용해서 그래프를 생성하고
# 학습 가능하도록 .pt로 저장하는 코드
import os
from typing import List, Dict, Tuple  
import numpy as np  
from tqdm import tqdm

import torch
from torch_geometric.data import Data

from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face
from occwl.graph import face_adjacency
from occwl.solid import Solid
from occwl.face import Face
from occwl.edge import Edge  

from dataset.load_stp_file import STEPFileLoader
from dataset.extract_uv_feat import FreatureExtract


class PreprocessUVGraph:
    def __init__(self) -> None: 
        pass
    
    @staticmethod
    def preprocss(
        stp_dir_path: str,
        save_dir_path: str) -> None:
        stp_file_paths = [f for f in os.listdir(stp_dir_path) if f.endswith('.step') or f.endswith('.stp')]
        
        for stp_file_name in tqdm(stp_file_paths):
            stp_file_path: str = os.path.join(stp_dir_path, stp_file_name)
            
            file_name: str = os.path.basename(stp_file_name).split('.')[0]  
            save_path: str = os.path.join(save_dir_path, file_name + ".pt")
            PreprocessUVGraph.save_stp_to_graph(stp_file_path, save_path)     
        return   
    
    @staticmethod
    def save_stp_to_graph(
        stp_file_path: str,
        save_path: str) -> None:
        stp_loader = STEPFileLoader(stp_file_path)
        # 라벨된 페이스는 여기서 로드하는데?
        # 라벨된 페이스의 순서와 UV 그래프에서의 페이스의 순서가 다른데, 굳이 다르게 할 필요 없이 처음부터 순서를 통일하면 안되나?
        labeled_faces, shape = stp_loader.load_labeled_topods_face()
        try:
            # 그래프를 만드는 코드인데, 라벨된 페이스는 왜 같이 출력함?
            uv_graph, _ = PreprocessUVGraph.generate_uv_graph(shape, labeled_faces)
        except Exception as e:
            raise ValueError(f"Failed to generate UV graph from {stp_file_path}: {e}")
        torch.save(uv_graph, save_path)
        return
    
    @staticmethod
    def generate_uv_graph(
        shape: TopoDS_Shape,
        labeled_face: Dict[TopoDS_Face, int]=None) -> Tuple[Data, Dict[TopoDS_Face, int]]:
        shape = Solid(shape)
        adjacency_info = face_adjacency(shape)
        faces: List[Face] = [list(adjacency_info.nodes.data())[i][1]['face'] for i in range(len(adjacency_info.nodes))]
        edges: List[Edge] = [list(adjacency_info.edges.data())[i][2]['edge'] for i in range(len(adjacency_info.edges))]
        
        node_feat: np.ndarray = FreatureExtract.extract_uv_features_from(faces)
        edge_attr: np.ndarray = FreatureExtract.extract_u_features_from(edges)
        node_feat = np.transpose(node_feat, (0, 3, 1, 2))
        edge_attr = np.transpose(edge_attr, (0, 2, 1))  
        
        if labeled_face != None:    
            y_labels: List[int] = [0] * len(faces)

            for face in labeled_face.keys():
                face_idx: int = faces.index(Face(face))
                y_labels[face_idx] = labeled_face[face]
                
            node_feat = torch.as_tensor(node_feat, dtype=torch.float32)
            edge_index = torch.tensor(list(adjacency_info.edges), dtype=torch.long).t().contiguous()
            edge_attr = torch.as_tensor(edge_attr, dtype=torch.float32) 
            y_labels = torch.as_tensor(y_labels, dtype=torch.long) 
            data = Data(
                x=node_feat,
                edge_index=edge_index, 
                edge_attr=edge_attr,  
                y=y_labels)
            return data, labeled_face
        else:
            node_feat = torch.as_tensor(node_feat, dtype=torch.float32)
            edge_index = torch.tensor(list(adjacency_info.edges), dtype=torch.long).t().contiguous()
            edge_attr = torch.as_tensor(edge_attr, dtype=torch.float32) 
            data = Data(
                x=node_feat,
                edge_index=edge_index, 
                edge_attr=edge_attr)
            labeled_face = {}
            for face in faces:
                labeled_face[face.topods_shape()] = 0
            return data, labeled_face


if __name__ == "__main__":
    stp_dir_path = "data/step/"
    save_dir_path = "data/labeled_graph/" 
    
    PreprocessUVGraph.preprocss(
        stp_dir_path,
        save_dir_path)