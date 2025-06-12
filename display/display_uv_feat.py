import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def display_uv_features(
    uv_feats: np.ndarray) -> None:
    
    num_uv: int = uv_feats.shape[1]
    for idx, uv_feat in enumerate(uv_feats):
        points = uv_feat[:, :, :3]  # Extract 3D points
        normals = uv_feat[:, :, 3:6]  # Extract normals
        mask = uv_feat[:, :, 6]  # Extract mask
        
        normalized_x = (points[:, :, 0] - np.min(points[:, :, 0])) / (np.max(points[:, :, 0]) - np.min(points[:, :, 0]))     
        normalized_y = (points[:, :, 1] - np.min(points[:, :, 1])) / (np.max(points[:, :, 1]) - np.min(points[:, :, 1]))
        normalized_z = (points[:, :, 2] - np.min(points[:, :, 2])) / (np.max(points[:, :, 2]) - np.min(points[:, :, 2]))
        normal_point = np.stack((normalized_x, normalized_y, normalized_z), axis=-1)  # Normalize points
        # nan 값은 0으로 
        normal_point = np.nan_to_num(normal_point, nan=0.0)
        

        fig = plt.figure(figsize=(15, 5))
        # Plot 3D points (정규화된 포인트 사용)
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(normal_point[:, :, 0].flatten(), 
                   normal_point[:, :, 1].flatten(), 
                   normal_point[:, :, 2].flatten(), 
                   s=10, alpha=0.6, c='blue')
        ax1.plot_wireframe(normal_point[:, :, 0], 
                          normal_point[:, :, 1], 
                          normal_point[:, :, 2], 
                          color='blue', alpha=0.3, linewidth=0.7)
        ax1.grid(True)
        ax1.set_title(f'Face {idx} - Points (Normalized)')
        ax1.set_xlabel('X (Normalized)')
        ax1.set_ylabel('Y (Normalized)')
        ax1.set_zlabel('Z (Normalized)')
        
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_zlim([0, 1])    
        
        # Plot normals
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.quiver(normal_point[:, :, 0].flatten(), 
                   normal_point[:, :, 1].flatten(), 
                   normal_point[:, :, 2].flatten(), 
                   normals[:, :, 0].flatten(), 
                   normals[:, :, 1].flatten(), 
                   normals[:, :, 2].flatten(), 
                   length=0.1, normalize=True, color='red', alpha=0.6)
        ax2.set_title(f'Face {idx} - Normals')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.set_zlim([0, 1])
        
        ax3 = fig.add_subplot(133)
        ax3.imshow(mask, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        ax3.set_title(f'Face {idx} - Mask')
        ax3.set_xlabel('U')
        ax3.set_ylabel('V')

        # 모든 격자점에 대해 눈금 표시
        ax3.set_xticks(np.arange(0, num_uv, 1))
        ax3.set_yticks(np.arange(0, num_uv, 1))

        # 주요 눈금에만 레이블 표시 (가독성을 위해)
        ax3.set_xticks(np.arange(0, num_uv, step=num_uv//4), minor=False)
        ax3.set_yticks(np.arange(0, num_uv, step=num_uv//4), minor=False)
        ax3.set_xticklabels(np.round(np.linspace(0, 1, num_uv//4), 2))
        ax3.set_yticklabels(np.round(np.linspace(0, 1, num_uv//4), 2))

        ax3.set_xlim([0, num_uv])
        ax3.set_ylim([0, num_uv])
                
        plt.tight_layout()
        plt.show()


def display_u_features(
    u_feats: np.ndarray) -> None:
    # shape: (27, 16, 6)
    num_uv: int = u_feats.shape[1]
    for idx, u_feat in enumerate(u_feats):
        points: np.ndarray = u_feat[:, :3]
        tangents: np.ndarray = u_feat[:, 3:]
        
        
        # Normalize points for visualization
        normalized_x = (points[:, 0] - np.min(points[:, 0])) / (np.max(points[:, 0]) - np.min(points[:, 0]))     
        normalized_y = (points[:, 1] - np.min(points[:, 1])) / (np.max(points[:, 1]) - np.min(points[:, 1]))
        normalized_z = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2]) - np.min(points[:, 2]))
        normal_point = np.stack((normalized_x, normalized_y, normalized_z), axis=-1)
        normal_point = np.nan_to_num(normal_point, nan=0.0)

        fig = plt.figure(figsize=(10, 5))
        
        # Plot 3D points
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(normal_point[:, 0].flatten(), 
               normal_point[:, 1].flatten(), 
               normal_point[:, 2].flatten(), 
               s=10, alpha=0.6, c='blue')
        ax1.set_title(f'Curve {idx} - Points (Normalized)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_zlim([0, 1])

        # Plot tangents
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.quiver(normal_point[:, 0].flatten(), 
              normal_point[:, 1].flatten(), 
              normal_point[:, 2].flatten(), 
              tangents[:, 0].flatten(), 
              tangents[:, 1].flatten(), 
              tangents[:, 2].flatten(), 
              length=0.1, normalize=True, color='red', alpha=0.6)
        ax2.set_title(f'Curve {idx} - Tangents')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.set_zlim([0, 1])
        plt.tight_layout()
        plt.show()

def display_uv_graph(
    uv_graph: Data) -> None:
    G = nx.Graph()
    edge_index = uv_graph.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0,i], edge_index[1,i])

    plt.figure(figsize=(10,10))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=500)
    plt.show()
    return