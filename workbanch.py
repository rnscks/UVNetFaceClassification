from OCC.Core.gp import gp_Pnt, gp_Dir
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Display.SimpleGui import init_display

from dataset.generation.turning import TurningMainShapeGenerator
from dataset.generation.turning import ChamferingRoundingGenerator

origin = gp_Pnt(0, 0, 0)    
direction = gp_Dir(0, 0, 1) 
height = 15.0
radius = 4.0

turning_shape_generator = TurningMainShapeGenerator(
    radius=radius,
    height=height)
turning_shape_generator.tapper('top', height=2.0)
turning_shape_generator.grooving(3.0, 0.5, 2.0)
turning_shape_generator.step(9.0, 1.0, 3.0)
ret = turning_shape_generator.step(8.0, 2.0, 3.0)

chamfer_round_generator = ChamferingRoundingGenerator(ret)
chamfer_round_generator.chamfer(0.5, 5)
ret = chamfer_round_generator.round(0.4, 1)

display, start_display, add_menu, add_function_to_menu = init_display()
print(f"Number of circular edges: {len(chamfer_round_generator.circular_edges)}")   
colors = ['blue', 'green', 'yellow', 'red']

edge_props_tuple_set = set()
for i, edge_tuple in enumerate(chamfer_round_generator.circular_edges):
    edge, r, z = edge_tuple  
    if (r, z) in edge_props_tuple_set:  
        print(f"Skipping duplicate edge {i}: {edge_tuple}")
        continue    
    edge_props_tuple_set.add((r, z))  
    print(f"Edge {i}: r={r}, z={z}")
    display.DisplayShape(TopoDS_Shape(edge), update=True, color=colors[i % len(colors)])  
display.DisplayShape(ret, update=True, transparency=0.9)
display.FitAll()
start_display()