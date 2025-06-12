import os
from typing import List   
import numpy as np

from occwl.face import Face
from occwl.edge import Edge
from occwl.uvgrid import ugrid, uvgrid
from occwl.io import load_step

from display.display_uv_feat import display_uv_features, display_u_features


class FreatureExtract:
    @staticmethod
    def extract_u_features_from(edges: List[Edge], num_uv: int=16) -> List[np.ndarray]:
        features: List[np.ndarray] = []
        
        for edge in edges:
            points: List[float] = ugrid(edge, num_uv, method='point')
            tangents: List[float] = ugrid(edge, num_uv, method='tangent')
            
            if points is None or tangents is None:
                edge_feature = np.zeros([num_uv, 6], dtype=np.float32)
                return edge_feature
                
            edge_feature: np.ndarray = np.concatenate((points, tangents), axis=-1)
            features.append(edge_feature)  
        
        np_features: np.ndarray = np.asarray(features)
        return np_features
    
    @staticmethod
    def extract_uv_features_from(faces: List[Face], num_uv: int = 16) -> List[np.ndarray]:
        features: List[np.ndarray] = [] 
        
        for face in faces:  
            points: List[float] = uvgrid(face, num_uv, num_uv, method='point')    
            normals: List[float] = uvgrid(face, num_uv, num_uv, method='normal')  
            visibility_status: List[float] = uvgrid(face, num_uv, num_uv, method='visibility_status')    
            
            # 0: TopAbs_IN, 1: TopAbs_OUT, 2: TopAbs_ON, 3: TopAbs_UNKNOWN
            # Face의 내부(0)에 있거나 경계(테두리, 2)에 있을 경우 True로 설정함 
            mask = np.logical_or(visibility_status == 0, visibility_status == 2)
            
            face_feature: np.ndarray = np.concatenate((points, normals, mask), axis=-1)  
            features.append(face_feature)   
        features: np.ndarray = np.asarray(features)
        return features


if __name__ == "__main__":
    shape = load_step('data/step/9996.step')[0]
    edges = shape.edges()
    faces = shape.faces()

    uv_feats: np.ndarray = FreatureExtract.extract_uv_features_from(faces=faces)
    u_feats: np.ndarray = FreatureExtract.extract_u_features_from(edges=edges)
    
    display_uv_features(uv_feats)
    display_u_features(u_feats)