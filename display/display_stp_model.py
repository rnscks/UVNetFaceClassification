import json
from typing import List, Dict, Tuple, Any

from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Display.SimpleGui import init_display
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.gp import gp_Pnt


def load_label_props(file_path: str) -> Tuple[List[str], Dict[str, Tuple]]:
    with open(file_path, 'r') as f:
        feat_props = json.load(f)
        label_names: List[str] = feat_props['LABEL_NAMES']
        color_table: Dict[str, Tuple] = feat_props['LABEL_COLORS']
    return label_names, color_table

def get_face_center(face):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    center = props.CentreOfMass()   
    return gp_Pnt(center.X(), center.Y(), center.Z())

def display_labeled_face(
    labeled_faces: Dict[TopoDS_Face, int],
    props_file_path: str,
    with_label_name: bool = False):
    display, start_display, _, _ = init_display()
    display.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
    label_names, color_table = load_label_props(props_file_path)
    
    for idx, face in enumerate(labeled_faces.keys()):
        label_name = label_names[labeled_faces[face]]
        color = color_table[label_name]
        
        display.DisplayShape(
            face, 
            update=True, 
            color=Quantity_Color(color[0]/255, color[1]/255, color[2]/255, Quantity_TOC_RGB),
            transparency=0.0)
        if with_label_name:
            display.DisplayMessage(get_face_center(face), f"{idx}, {label_name}", height=20, message_color=(0, 0, 0))
    start_display()
    
def display_shape(
    shape: TopoDS_Shape) -> None:
    display, start_display, _, _ = init_display()
    display.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
    display.DisplayShape(shape, update=True)
    start_display()
    return

# def display_predicted_face(
#     preds: List[int],
#     faces: List[TopoDS_Face],
#     props_file_path: str,
#     with_labels: bool = True,
#     capture_path: str = None):
#     display, start_display, _, _ = init_display()
#     display.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
#     label_names, color_table = load_label_props(props_file_path)
    
#     face_idx: int = 0
#     for face, pred in zip(faces, preds):  
#         label_name = label_names[pred]
#         color = color_table[label_name]
        
#         display.DisplayShape(
#             face, 
#             update=True, 
#             color=Quantity_Color(color[0]/255, color[1]/255, color[2]/255, Quantity_TOC_RGB),
#             transparency=0.0)
#         if with_labels: 
#             display.DisplayMessage(get_face_center(face), f"{face_idx}, {label_name}", height=25, message_color=(0, 0, 0))
#         face_idx += 1   
        
#     if capture_path != None:
#         display.View.Dump(capture_path)
#         print(f"Captured image saved to {capture_path}")    
#         display.EraseAll()
#         return
#     start_display()
