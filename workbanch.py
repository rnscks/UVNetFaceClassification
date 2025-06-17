from typing import List

from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeBox
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


    def chamfer(self, radius: float, index: int) -> TopoDS_Shape:
        return NotImplementedError()
    
    def round(self, radius: float, index: int) -> TopoDS_Shape:
        return NotImplementedError()

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
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(ret, update=True)
    display.FitAll()
    start_display()