from typing import List, Tuple, Dict, Set

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods, TopoDS_Face, TopoDS_Shape
from OCC.Core.StepRepr import StepRepr_RepresentationItem


class STEPFileLoader:
    def __init__(self, file_path: str) -> None:
        self.reader: STEPControl_Reader = STEPControl_Reader()  
        self.file_path = file_path
        if not self.file_path or not isinstance(self.file_path, str):
            raise ValueError(f"Invalid file path: {self.file_path}. It must be a non-empty string.")    
        if not self.file_path.endswith('.stp') and not self.file_path.endswith('.step'):    
            raise ValueError(f"Invalid file format: {self.file_path}. Only .stp and .step files are supported.")    
        
        status = self.reader.ReadFile(self.file_path)
        self.reader.TransferRoots()
        return
        
        
    def load_labeled_topods_face(self) -> Tuple[Dict[TopoDS_Face, int], TopoDS_Shape]:
        shape: TopoDS_Shape = self.load_topods_shape()
        treader = self.reader.WS().TransferReader()

        face_id_table: Dict[TopoDS_Face, int] = {}
        face_list: List[TopoDS_Face] = self._get_face_list_from(shape)
        
        for face in face_list:
            item = treader.EntityFromShapeResult(face, 1)
            if item is None:
                raise ValueError(f"Failed to get item from face in {self.file_path}")
            
            item = StepRepr_RepresentationItem.DownCast(item)
            label: str = item.Name().ToCString()
            if label == '' or int(label) > 25 or int(label) < 0:
                raise ValueError(f"Invalid face label: {label} in {self.file_path}")
            face_id_table[face] = int(label)
        return face_id_table, shape

    def load_topods_shape(self) -> TopoDS_Shape:
        shape: TopoDS_Shape = self.reader.OneShape()
        if shape == None:
            raise ValueError(f"Failed to transfer shape from {self.file_path}")
        if shape.IsNull():
            raise ValueError(f"Shape is null for {self.file_path}")
        return shape

    def _get_face_list_from(self, shape: TopoDS_Shape) -> List[TopoDS_Face]:
        fset: Set[TopoDS_Face] = set()
        exp = TopExp_Explorer(shape,TopAbs_FACE)
        while exp.More():
            shape: TopoDS_Face = exp.Current()
            exp.Next()
            face: TopoDS_Face = topods.Face(shape)
            fset.add(face)
        return list(fset)



if __name__ == "__main__":
    from OCC.Display.SimpleGui import init_display
    from display.display_stp_model import display_labeled_face
    
    file_path = 'data/step/9996.step'
    step_loader = STEPFileLoader(file_path)    
    labeled_faces, _ = step_loader.load_labeled_topods_face()
    shape: TopoDS_Shape = step_loader.load_topods_shape()
    
    # shape 가시화
    # display, start_display, add_menu, add_function_to_menu = init_display() 
    # display.DisplayShape(shape, update=True)
    # start_display()
    
    # labeled_faces 가시화
    display_labeled_face(
        labeled_faces=labeled_faces, 
        props_file_path="data/LABEL_PROPS.json",
        with_label_name=True)