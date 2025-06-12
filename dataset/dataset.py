import os
from typing import List
import tqdm

import torch
from torch_geometric.data import Data, Batch, Dataset
    
class GraphDataset(Dataset):
    def __init__(self, dataset_path: str, transform=None):
        self.graph_list: List[Data] = []
        dataset_path: str = dataset_path
        pt_files: List[str] = [f for f in os.listdir(dataset_path) if f.endswith('.pt')]
        
        for pt_file in tqdm.tqdm(pt_files):
            file_name = pt_file.split(".")[0]
            pt_file = os.path.join(dataset_path, pt_file)
            data: Data = torch.load(pt_file, weights_only=False)
            data.filename = file_name
            if data.edge_index.size()[0]!=0:
                self.graph_list.append(data)

    def len(self):
        return len(self.graph_list)
    
    def get(self, idx):
        sample = self.graph_list[idx]
        return sample
    
    def __len__(self):
        return len(self.graph_list)
    
    def __getitem__(self, idx):
        sample = self.graph_list[idx]
        return sample

    def _collate(self, batch):
        batched_graph = Batch([graph for graph in batch])
        return batched_graph