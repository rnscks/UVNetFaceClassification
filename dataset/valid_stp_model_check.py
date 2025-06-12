from typing import List, Set
import logging

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Face, TopoDS_Solid, topods
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer

from occwl.graph import face_adjacency
from occwl.solid import Solid

# 로깅 설정
logging.basicConfig(
    filename='data/stp_model_validation.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_face_list(shape: TopoDS_Shape) -> List[TopoDS_Shape]:
    fset: Set[TopoDS_Shape] = set()
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face: TopoDS_Face = topods.Face(exp.Current())
        exp.Next()
        fset.add(face)
    return list(fset)   

def is_valid_solid(stp_file_path: str) -> bool:
    reader = STEPControl_Reader()
    reader.ReadFile(stp_file_path)
    reader.TransferRoots()

    # STEP 모델의 기본 요소 확인
    shape: TopoDS_Shape = reader.OneShape() 
    if shape == None:
        logging.error(f"{stp_file_path}: Shape is None.")
        return False
    
    treader  = reader.WS().TransferReader() 
    face_list: List[TopoDS_Face] = get_face_list(shape)
    if len(face_list) == 0:
        logging.error(f"{stp_file_path}: No faces found in the shape.")
        return False
    
    # STEP 파일에 정의된 라벨 확인
    for face in face_list:
        item = treader.EntityFromShapeResult(face, 1)
        if item == None:
            logging.error(f"{stp_file_path}: Item is None.")
            return False
        
        item = StepRepr_RepresentationItem.DownCast(item)
        label: str = item.Name().ToCString()
        try:
            if label == "":
                logging.error(f"{stp_file_path}: Label is empty for face.")
                return False
            label = int(label)
            if 0 > label or 25 < label:
                logging.error(f"{stp_file_path}: Label {label} is out of range (0-25).")
                return False 
        except Exception as e:
            logging.error(f"{stp_file_path}: Invalid label format for face({e}).")
            return False
    try:
        # UV 그래프 생성에 필수 로직
        solid = Solid(shape)
        adjacency = face_adjacency(solid)
    except Exception as e:
        logging.error(f"{stp_file_path}: Error generating face adjacency({e}).")
        return False
    return True


if __name__ == "__main__":
    import os
    from tqdm import tqdm
    
    dir_path = ''
    # stp_files = [f for f in os.listdir(dir_path) if f.endswith('.stp') or f.endswith('.step')]  
    
    # for stp_file in tqdm(stp_files):
    #     if not is_valid_solid(os.path.join(dir_path, stp_file)):
    #         os.remove(os.path.join(dir_path, stp_file))
    file_path = "data/stp2/23052.step"
    if not is_valid_solid(file_path):
        print(f"Invalid STEP file: {file_path}")    
    else:
        print(f"Valid STEP file: {file_path}")
