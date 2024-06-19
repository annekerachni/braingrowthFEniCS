import gmsh

# sphere mesh
#############
gmsh.initialize()
gmsh.model.add("ellipsoid")
gmsh.option.setNumber("Mesh.Algorithm3D", 1) # 3D mesh algorithm (1=Delaunay, 2=New Delaunay, 4=Frontal, 5=Frontal Delaunay, 6=Frontal Hex, 7=MMG3D, 9=R-tree)
gmsh.option.setNumber("Mesh.MeshSizeMax", 0.05) # max size of tetrahedrons
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1) # "Optimize the mesh using Netgen to improve the quality of tetrahedral elements"
gmsh.option.setNumber("Mesh.Smoothing", 1)

radius = 0.5 # SI (m)
center_coords = [radius, radius, radius]
mesh = gmsh.model.occ.addSphere(center_coords[0], center_coords[1], center_coords[2], radius)
# dilate_coef = [0.9, 1.0, 0.7] 
# gmsh.model.occ.dilate([(3, sphere_mesh)], center_coords[0], center_coords[1], dilate_coef[0], dilate_coef[1], dilate_coef[2])
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)
gmsh.write("./contact_mechanics/one_sphere/sphere_mesh.msh") 
# if visualisation == True:
# gmsh.fltk.run() # plot ellipsoid
gmsh.finalize()
#sphere_mesh = meshio.read("./contact_mechanics/one_sphere/results_sphere_contact.msh")

# convert .msh to .xdmf
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0]))) # braingrowthFEniCS
from utils.converters import convert_meshformats
convert_meshformats.msh_to_xml_xdmf("./contact_mechanics/one_sphere/sphere_mesh.msh", "./contact_mechanics/one_sphere/sphere_mesh.xml") 
