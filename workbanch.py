import random
from typing import List

from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeBox
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeChamfer
from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Edge
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.GeomAbs import GeomAbs_Circle


class TurningMainShapeGenerator:
    def __init__(self, radius: float, height: float):
        self.radius: float = radius
        self.height: float = height
        self.stock_shape: TopoDS_Shape = self._generate_cylinder_shape(radius=radius, height=height)
    
    
    def tapper(self, opt: str='top', height: float=1.0) -> None:  
        if opt == 'top':    
            center = gp_Pnt(0.0, 0.0, self.height - height)
            direction = gp_Dir(0.0, 0.0, 1.0) 
            
            box = BRepPrimAPI_MakeBox(
                gp_Pnt(-self.radius, -self.radius, self.height - height),
                2*self.radius,
                2*self.radius,
                height).Shape() 
            self.stock_shape = BRepAlgoAPI_Cut(self.stock_shape, box).Shape()   
            self.is_top_turning = True
        elif opt == 'bottom':
            center = gp_Pnt(0.0, 0.0, height)
            direction = gp_Dir(0.0, 0.0, -1.0)
            
            box = BRepPrimAPI_MakeBox(  
                gp_Pnt(-self.radius, -self.radius, 0),
                2*self.radius,
                2*self.radius,
                height).Shape() 
            self.stock_shape = BRepAlgoAPI_Cut(self.stock_shape, box).Shape()
            self.is_bottom_turning = True
        else:
            raise ValueError("Invalid option for cone placement. Use 'top' or 'bottom'.")   
        
        
        cone_axis = gp_Ax2(center, direction)
        cone = BRepPrimAPI_MakeCone(cone_axis, self.radius, 0, height).Shape()
        fused_shape = BRepAlgoAPI_Fuse(self.stock_shape, cone).Shape()  
        self.stock_shape = fused_shape  
        return self.stock_shape 
    
    def grooving(
        self,
        height: float = 1.0,
        depth: float = 1.0,
        width: float = 1.0) -> TopoDS_Shape:
        if height + width > self.height:
            raise ValueError("Height and width exceed the stock shape height.") 
        if depth >= self.radius:
            raise ValueError("Depth must be less than the radius of the stock shape.")  
        
        groove_shape: TopoDS_Shape = self._generate_tube_shape(
            outer_radius=self.radius, 
            inner_radius=self.radius - depth, 
            height=height, 
            width=width) 
        self.stock_shape = self._extrude_cut(
            stock_shape=self.stock_shape,   
            extrusion_shape=groove_shape)   
        return self.stock_shape
    
    def step(
        self,
        height: float = 1.0,
        depth: float = 1.0,
        width: float =1.0) -> TopoDS_Shape:
        if height + width > self.height:
            raise ValueError("Height and width exceed the stock shape height.") 
        step_shape: TopoDS_Shape = self._generate_tube_shape(
            outer_radius=self.radius + depth, 
            inner_radius=self.radius, 
            height=height, 
            width=width) 
        self.stock_shape = self._fuse_shapes(
            stock_shape=self.stock_shape,   
            fuse_shape=step_shape)   
        return self.stock_shape
    
    def _generate_tube_shape(self, 
                             outer_radius: float, 
                             inner_radius: float, 
                             height: float,
                             width: float) -> TopoDS_Shape:
        axis = gp_Ax2(gp_Pnt(0, 0, height), gp_Dir(0, 0, 1))
        outer_cylinder = BRepPrimAPI_MakeCylinder(axis, outer_radius, width).Shape()
        inner_cylinder = BRepPrimAPI_MakeCylinder(axis, inner_radius, width).Shape()
        return self._extrude_cut(stock_shape=outer_cylinder, extrusion_shape=inner_cylinder)
    
    def _generate_cylinder_shape(self, radius: float, height: float) -> TopoDS_Shape:   
        axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        return BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()   
    
    def _extrude_cut(self, 
                     stock_shape: TopoDS_Shape, 
                     extrusion_shape: TopoDS_Shape):
        cut_operation = BRepAlgoAPI_Cut(stock_shape, extrusion_shape)
        cut_operation.Build()
        
        if cut_operation.IsDone():
            return cut_operation.Shape()
        else:
            raise RuntimeError("Failed to perform extrude cut operation")   
    
    def _fuse_shapes(self, 
                     stock_shape: TopoDS_Shape, 
                     fuse_shape: TopoDS_Shape) -> TopoDS_Shape:
        fuse_operation = BRepAlgoAPI_Fuse(stock_shape, fuse_shape)
        fuse_operation.Build()
        
        if fuse_operation.IsDone():
            return fuse_operation.Shape()
        else:
            raise RuntimeError("Failed to perform fuse operation")  

class ChamferingRoundingGenerator:
    def __init__(self,
                 stock_shape: TopoDS_Shape):
        self.stock_shape: TopoDS_Shape = stock_shape
        self.circular_edges: List[tuple[TopoDS_Edge, float, float]] = self._find_circular_edges()
    
    def _find_circular_edges(self) -> List[tuple[TopoDS_Edge, float, float]]:
        circular_edges = []
        edge_explorer = TopExp_Explorer(self.stock_shape, TopAbs_EDGE)
        
        while edge_explorer.More():
            edge = edge_explorer.Current()
            edge_explorer.Next()
            
            curve_adaptor = BRepAdaptor_Curve(edge)
            
            if curve_adaptor.GetType() == GeomAbs_Circle:
                circle = curve_adaptor.Circle()
                center = circle.Location()
                radius = circle.Radius()
                circular_edges.append((edge, center.Z(), radius))
        
        circular_edges.sort(key=lambda x: (x[1], x[2]))
        print (f' {len(circular_edges)}')
        return circular_edges
    
    def _apply_random_operation(self, radius: float, operation_type: str) -> TopoDS_Shape:
        if not self.circular_edges:
            return self.stock_shape
        
        attempted_indices = []
        while len(attempted_indices) < len(self.circular_edges):
            available_indices = [i for i in range(len(self.circular_edges)) if i not in attempted_indices]
            index = random.choice(available_indices)
            attempted_indices.append(index)
            target_edge = self.circular_edges[index][0]
            
            try:
                if operation_type == "chamfer":
                    maker = BRepFilletAPI_MakeChamfer(self.stock_shape)
                    maker.Add(radius, target_edge)
                elif operation_type == "round":
                    maker = BRepFilletAPI_MakeFillet(self.stock_shape)
                    maker.Add(radius, target_edge)
                else:
                    raise ValueError(f"Invalid operation type: {operation_type}")
                
                maker.Build()
                if maker.IsDone():
                    self.stock_shape = maker.Shape()
                    self.circular_edges = self._find_circular_edges()
                    return self.stock_shape
            except:
                continue
        
        return self.stock_shape
    
    def _apply_operation_by_index(self, radius: float, operation_type: str, index: int) -> TopoDS_Shape:
        if index < 0 or index >= len(self.circular_edges):
            raise ValueError(f"Invalid index {index}. Available indices: 0 to {len(self.circular_edges)-1}")
        
        target_edge = self.circular_edges[index][0]
        
        try:
            if operation_type == "chamfer":
                maker = BRepFilletAPI_MakeChamfer(self.stock_shape)
                maker.Add(radius, target_edge)
            elif operation_type == "round":
                maker = BRepFilletAPI_MakeFillet(self.stock_shape)
                maker.Add(radius, target_edge)
            else:
                raise ValueError("Invalid operation type")
            
            maker.Build()
            
            if maker.IsDone():
                self.stock_shape = maker.Shape()
                self.circular_edges = self._find_circular_edges()
                print(f"{operation_type.capitalize()} applied to edge at index {index}")
                return self.stock_shape
            else:
                print(f"Warning: Edge at index {index} is not suitable for {operation_type}")
                return self.stock_shape
                
        except:
            print(f"Warning: Edge at index {index} is not suitable for {operation_type}")
            return self.stock_shape
        
    def chamfer(self, radius: float, index: int = None) -> TopoDS_Shape:
        if index is not None:
            return self._apply_operation_by_index(radius, "chamfer", index)
        else:
            return self._apply_random_operation(radius, "chamfer")
        
    def round(self, radius: float, index: int = None) -> TopoDS_Shape:
        if index is not None:
            return self._apply_operation_by_index(radius, "round", index)
        else:
            return self._apply_random_operation(radius, "round")

if __name__ == "__main__":
    from OCC.Display.SimpleGui import init_display
    
    origin = gp_Pnt(0, 0, 0)    
    direction = gp_Dir(0, 0, 1) 
    height = 15.0
    radius = 4.0
    
    turning_shape_generator = TurningMainShapeGenerator(
        radius=radius,
        height=height)
    turning_shape_generator.tapper('top', height=2.0)
    turning_shape_generator.tapper('bottom', height=1.0)
    turning_shape_generator.grooving(3.0, 0.5, 2.0)
    turning_shape_generator.step(9.0, 1.0, 3.0)
    ret = turning_shape_generator.step(8.0, 2.0, 3.0)

    chamfer_round_generator = ChamferingRoundingGenerator(ret)
    chamfer_round_generator.chamfer(0.2, 28)
    chamfer_round_generator.round(0.4, 14)
    ret= chamfer_round_generator.stock_shape

    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(ret, update=True)
    display.FitAll()
    start_display()