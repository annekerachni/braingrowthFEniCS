import fenics
import vedo.dolfin
import pygalmesh

"""reference: https://pypi.org/project/pygalmesh/"""

# sphere mesh
# -----------
"""
s = pygalmesh.Ball([0, 0, 0], 1.0)

tets_circumradius = 0.03 # 0.05 for a sphere with 143808 tets; 0.04 for a sphere with 280647 tets; 0.03 for a sphere with 663855 tets
mesh = pygalmesh.generate_mesh(s, max_cell_circumradius=tets_circumradius)
"""

# ellipsoid mesh
# --------------
coef_x, coef_y, coef_z = 0.9, 1.0, 0.7 # ellipsoid dilatation coefficients
s = pygalmesh.Stretch(pygalmesh.Ball([0, 0, 0], 1.0), [coef_x, coef_y, coef_z])

tets_circumradius = 0.03 # 0.05 for an ellipsoid with 217424 tets; 0.04 for an ellipsoid with 424549 tets; 0.03 for an ellipsoid with 1006635 tets
mesh = pygalmesh.generate_mesh(s, max_cell_circumradius=tets_circumradius) 

# mesh.points, mesh.cells

# generate meshes
# ---------------
mesh.write("./data/pygalmesh/mesh_PYGALMESH_vol_tets{}.vtk".format( str(tets_circumradius).replace(".", "") )) # to visualize in Paraview
mesh.write("./data/pygalmesh/mesh_PYGALMESH_vol_tets{}.xml".format( str(tets_circumradius).replace(".", "") )) # to build FEniCS mesh + visualize in vedo.dolfin
mesh.write("./data/pygalmesh/mesh_PYGALMESH_surf_tets{}.stl".format( str(tets_circumradius).replace(".", "") )) # to manage Delaunay triangulation

# FEniCS input mesh + display mesh with vedo.dolfin
# -------------------------------------------------
mesh_fenics = fenics.Mesh("./data/pygalmesh/mesh_PYGALMESH_vol_tets{}.xml".format(str(tets_circumradius).replace(".", "")))
vedo.dolfin.plot(mesh_fenics, wireframe=False, text='pygalmesh mesh', style='paraview', axes=4).close()