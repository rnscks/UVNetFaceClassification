from typing import List

from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse   
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Edge
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Circle
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut


class TurningMachiningModelGenerator:
    def __init__(self, 
                 stock_height: float=10.0, 
                 stock_radius: float=2.0):
        self.origin: gp_Pnt = gp_Pnt(0, 0, 0)
        self.direction: gp_Dir = gp_Dir(0, 0, 1)
        self.height: float = stock_height
        self.radius: float = stock_radius
        self.stock_model: TopoDS_Shape = self._create_stock_model()
        self.is_top_turning: bool = False
        self.is_bottom_turning: bool = False


    def _create_stock_model(self) -> TopoDS_Shape:
        axis = gp_Ax2(self.origin, self.direction)  
        cylinder = BRepPrimAPI_MakeCylinder(axis, self.radius, self.height).Shape()
        return cylinder

    def _search__edge_on(self, opt: str ='top') -> List[TopoDS_Edge]:
        def is_on_top_circle_edge(edge: TopoDS_Edge) -> bool:
            edge_curve = BRepAdaptor_Curve(edge)
            if edge_curve.GetType() == GeomAbs_Circle:
                circle = edge_curve.Circle()
                circle_center = circle.Location()
                circle_height = circle_center.Z()
                return abs(circle_height - self.height) <= 1e-6
            return False
        
        def is_on_bottom_circle_edge(edge: TopoDS_Edge) -> bool:
            edge_curve = BRepAdaptor_Curve(edge)
            if edge_curve.GetType() == GeomAbs_Circle:
                circle = edge_curve.Circle()
                circle_center = circle.Location()
                circle_height = circle_center.Z()
                return abs(circle_height) <= 1e-6
            return False    
        
        edge_list: List[TopoDS_Edge] = []
        edge_exp = TopExp_Explorer(self.stock_model, TopAbs_EDGE)   
        while edge_exp.More():
            edge = edge_exp.Current()
            if edge.IsNull():    
                edge_exp.Next() 
                continue
            try:
                if opt == 'top' and is_on_top_circle_edge(edge):
                    edge_list.append(edge)  
                elif opt == 'bottom' and is_on_bottom_circle_edge(edge):
                    edge_list.append(edge)
            except Exception as e:
                raise ValueError(f"Error processing edge: {e}")
            edge_exp.Next()
        return edge_list

    def set_stock_properties(self, height: float, radius: float) -> None:
        self.height = height
        self.radius = radius
        self.stock_model = self._create_stock_model()
        return
    
    def set_straight_on(self, opt: str = 'top', height1: float=1.0, height2: float=2.0) -> None:
        return NotImplementedError("Straight functionality is not yet implemented.")
    
    def set_rough_on(self, opt: str = 'top', height: float=5.0) -> None:
        return NotImplementedError("Rough functionality is not yet implemented.")
    
    def set_turing_chamfer_on(self, opt: str = 'top', height: float=5.0) -> None:
        return NotImplementedError("Turning chamfer functionality is not yet implemented.")
    
    def set_tapper_on(self, opt: str='top', height: float=1.0) -> None:  
        if opt == 'top':    
            center = gp_Pnt(
                self.origin.X() + self.direction.X() * (self.height - height),
                self.origin.Y() + self.direction.Y() * (self.height - height), 
                self.origin.Z() + self.direction.Z() * (self.height - height))
            direction = gp_Dir(
                self.direction.X(),
                self.direction.Y(),
                self.direction.Z()) 
            
            box = BRepPrimAPI_MakeBox(
                gp_Pnt(-self.radius, -self.radius, self.height - height),
                2*self.radius,
                2*self.radius,
                height).Shape() 
            self.stock_model = BRepAlgoAPI_Cut(self.stock_model, box).Shape()   
            self.is_top_turning = True
        elif opt == 'bottom':
            center = gp_Pnt(
                self.origin.X() + self.direction.X() * height,
                self.origin.Y() + self.direction.Y() * height,
                self.origin.Z() + self.direction.Z() * height)
            direction = gp_Dir( 
                -self.direction.X(),
                -self.direction.Y(),
                -self.direction.Z())
            
            box = BRepPrimAPI_MakeBox(
                gp_Pnt(-self.radius, -self.radius, 0),
                2*self.radius,
                2*self.radius,
                height).Shape() 
            self.stock_model = BRepAlgoAPI_Cut(self.stock_model, box).Shape()
            self.is_bottom_turning = True
        else:
            raise ValueError("Invalid option for cone placement. Use 'top' or 'bottom'.")   
        
        
        cone_axis = gp_Ax2(center, direction)
        cone = BRepPrimAPI_MakeCone(cone_axis, self.radius, 0, height).Shape()
        fused_shape = BRepAlgoAPI_Fuse(self.stock_model, cone).Shape()  
        self.stock_model = fused_shape  
        return
    
    def set_turing_fillet_on(self, opt: str='top', radius: float=0.1) -> None:  
        fillet = BRepFilletAPI_MakeFillet(self.stock_model)
        if opt == 'top':
            top_edges: List[TopoDS_Edge] = self._search__edge_on(opt='top')
            for edge in top_edges:
                fillet.Add(radius, edge)
            self.is_top_turning = True
        elif opt == 'bottom':   
            bottom_edges: List[TopoDS_Edge] = self._search__edge_on(opt='bottom')
            for edge in bottom_edges:
                fillet.Add(radius, edge)
            self.is_bottom_turning = True       
        self.stock_model = fillet.Shape()
        return
    
    def set_grooving_S(self, 
                       height: float=5.0, 
                       depth: float=0.5, 
                       width: float=1.0) -> None:        
        groove_center = gp_Pnt(
            self.origin.X(),
            self.origin.Y(), 
            self.origin.Z() + height)
        
        groove_outer_radius = self.radius + 1.0 
        groove_axis = gp_Ax2(groove_center, self.direction)
        groove_outer = BRepPrimAPI_MakeCylinder(
            groove_axis, groove_outer_radius, width).Shape()
        
        groove_inner_radius = self.radius - depth
        groove_inner = BRepPrimAPI_MakeCylinder(
            groove_axis, groove_inner_radius, width).Shape()
        
        groove_volume = BRepAlgoAPI_Cut(groove_outer, groove_inner).Shape()
        result = BRepAlgoAPI_Cut(self.stock_model, groove_volume).Shape()
        self.stock_model = result
        return
    
    def set_grooving_T(self, 
                    height: float=5.0,
                    depth: float=0.5,
                    width: float=1.0) -> None:
        return NotImplementedError("Grooving T functionality is not yet implemented.")
    
    def set_step_S_on(self,
                   height: float=5.0,
                   width: float=1.0) -> None:
        return NotImplementedError("Step functionality is not yet implemented.")
    
    def set_step_T_on(self,
                     height: float=5.0,
                     width: float=1.0) -> None:
        return NotImplementedError("Step functionality is not yet implemented.")  
    
    def get_turning_machining_model(self) -> TopoDS_Shape:
        return self.stock_model


if __name__ == "__main__":
    from OCC.Display.SimpleGui import init_display
    from OCC.Core.gp import gp_Pnt, gp_Dir
    
    stock_height = 15.0
    stock_radius = 5.0
    
    turning_model_generator = TurningMachiningModelGenerator(
        stock_height=stock_height,
        stock_radius=stock_radius)
    turning_model_generator.set_tapper_on('top', height=2.0)
    turning_model_generator.set_turing_fillet_on('bottom', radius=0.5)
    turning_model_generator.set_grooving_S(height=2.0, depth=0.5, width=2.0)
    turning_model_generator.set_grooving_S(height=10.0, depth=1.5, width=1.0)

    
    stp_model = turning_model_generator.get_turning_machining_model()     

    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(stp_model, update=True)
    
    start_display()