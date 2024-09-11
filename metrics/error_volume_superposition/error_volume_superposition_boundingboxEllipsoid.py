# metrics from and co-developped with Stéphane Urcun (Department of Engineering, Faculty of Science, Technology and Medicine, University of Luxembourg, Esch-sur-Alzette, Luxembourg.)

import fenics
import numpy as np
#import vedo.dolfin
import meshio
import gmsh
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
#from braingrowth_3D.biophysical_quasistatic_3D.FEM_biomechanical_model_quasistatic import projection

# brain data to compare 
#######################
# N.B. Need to be registered 
brain_dHCP_28GW_path = "./data/dHCP_28GW/transformed_dhcp28GW_isotropic_Taubin5_in_meters_reoriented_V2.xdmf"
brain_numerical_28GW_path = "./results/simulation_28GW/simulation_alphaTANFAbased_H0segmentation_28GW_volume.xdmf"

# Load brain_numerical (volume) at 28GW [m]
###########################################
mesh_brain_numerical_28 = fenics.Mesh()
with fenics.XDMFFile(brain_numerical_28GW_path) as infile: 
	infile.read(mesh_brain_numerical_28)

ZbrainNum = fenics.FunctionSpace(mesh_brain_numerical_28, "DG", 0)
markers_brain_numerical_28_ZbrainNum = fenics.Function(ZbrainNum)
markers_brain_numerical_28_ZbrainNum.vector()[:] = 1 # mark all initial numerical mesh tetraedrons with 1	

dx_brain_numerical_28 = fenics.Measure('dx', domain=mesh_brain_numerical_28)
volume_brain_numerical_28 = fenics.assemble(fenics.Constant(1) * dx_brain_numerical_28) # [m3]
print("\nvolume_brain_numerical_28GW = {} m3".format(volume_brain_numerical_28))

#hmin_brain_numerical_28 = mesh_brain_numerical_28.hmin() # min cell edge size [m]
hmin_brain_numerical_28 = fenics.CellDiameter(mesh_brain_numerical_28) # symbolic expression providing cell diameter [m] for each cell of the mesh 
h_proj_brain_numerical_28 = fenics.project(hmin_brain_numerical_28, ZbrainNum) 
min_cell_size_numerical_28 = np.nanmin(h_proj_brain_numerical_28.vector()[:])
average_cell_size_numerical_28 = np.nanmean(h_proj_brain_numerical_28.vector()[:])
max_cell_size_numerical_28 = np.nanmax(h_proj_brain_numerical_28.vector()[:]) 
print("max cell size in brain_numerical_28GW = {} m".format(max_cell_size_numerical_28))

# Load brain_dHCP (volume) at 28GW (ground truth) [m]
#####################################################
mesh_brain_dHCP_28 = fenics.Mesh()
with fenics.XDMFFile(brain_dHCP_28GW_path) as infile: 
	infile.read(mesh_brain_dHCP_28)
 
ZbrainDHCP = fenics.FunctionSpace(mesh_brain_dHCP_28, "DG", 0)
markers_brain_dHCP_28_ZbrainDHCP = fenics.Function(ZbrainDHCP)
markers_brain_dHCP_28_ZbrainDHCP.vector()[:] = 1 # mark all initial dHCP mesh tetraedrons with 1
 
dx_brain_dHCP_28 = fenics.Measure('dx', domain=mesh_brain_dHCP_28)
volume_brain_dHCP_28 = fenics.assemble(fenics.Constant(1) * dx_brain_dHCP_28) # [m3]
print("\nvolume_brain_dHCP_28GW = {} m3".format(volume_brain_dHCP_28))

#hmin_brain_dHCP_28 = mesh_brain_dHCP_28.hmin() # min edge [m]
hmin_brain_dHCP_28 = fenics.CellDiameter(mesh_brain_dHCP_28)
h_proj_brain_dHCP_28 = fenics.project(hmin_brain_dHCP_28, ZbrainDHCP)
min_cell_size_dHCP_28 = np.nanmin(h_proj_brain_dHCP_28.vector()[:])
average_cell_size_dHCP_28 = np.nanmean(h_proj_brain_dHCP_28.vector()[:])
max_cell_size_dHCP_28 = np.nanmax(h_proj_brain_dHCP_28.vector()[:])
print("max cell size in brain_dHCP_28GW = {} m\n".format(max_cell_size_dHCP_28))

# Define bounding box around the two brain meshes
#################################################

brain_dHCP_maxX, brain_dHCP_minX = np.max(mesh_brain_dHCP_28.coordinates()[:,0]), np.min(mesh_brain_dHCP_28.coordinates()[:,0])
brain_dHCP_maxY, brain_dHCP_minY = np.max(mesh_brain_dHCP_28.coordinates()[:,1]), np.min(mesh_brain_dHCP_28.coordinates()[:,1])
brain_dHCP_maxZ, brain_dHCP_minZ =  np.max(mesh_brain_dHCP_28.coordinates()[:,2]), np.min(mesh_brain_dHCP_28.coordinates()[:,2])

brain_numerical_maxX, brain_numerical_minX = np.max(mesh_brain_numerical_28.coordinates()[:,0]), np.min(mesh_brain_numerical_28.coordinates()[:,0])
brain_numerical_maxY, brain_numerical_minY = np.max(mesh_brain_numerical_28.coordinates()[:,1]), np.min(mesh_brain_numerical_28.coordinates()[:,1])
brain_numerical_maxZ, brain_numerical_minZ =  np.max(mesh_brain_numerical_28.coordinates()[:,2]), np.min(mesh_brain_numerical_28.coordinates()[:,2])

brain_maxX, brain_minX = max(brain_dHCP_maxX, brain_numerical_maxX), min(brain_dHCP_minX, brain_numerical_minX)
brain_maxY, brain_minY = max(brain_dHCP_maxY, brain_numerical_maxY), min(brain_dHCP_minY, brain_numerical_minY)
brain_maxZ, brain_minZ = max(brain_dHCP_maxZ, brain_numerical_maxZ), min(brain_dHCP_minZ, brain_numerical_minZ)

dX = brain_maxX - brain_minX
dY = brain_maxY - brain_minY
dZ = brain_maxZ - brain_minZ
print("dX = {}mm, dY = {}mm, dZ = {}mm".format(dX*1000, dY*1000, dZ*1000))

hmin = min(min_cell_size_numerical_28, min_cell_size_dHCP_28) 
hmean = 0.5*(average_cell_size_numerical_28 + average_cell_size_dHCP_28)
hmax = max(max_cell_size_numerical_28, max_cell_size_dHCP_28)
"""
nX, nY, nZ = round(dX/h), round(dY/h), round(dZ/h) 
print("nX, nY, nZ = {}, {}, {}\n".format(nX, nY, nZ)) 

bounding_box_mesh = fenics.BoxMesh(fenics.Point(brain_minX, brain_minY, brain_minZ), fenics.Point(brain_maxX, brain_maxY, brain_maxZ), nX, nY, nZ)
"""

# parameters
############
mesh_bb_MSH_path = "./metrics/RMSE/RMSE_bounding_ellipsoid_hmean.msh"
visualisation = False

COG_bb = 0.5*(brain_minX + brain_maxX), 0.5*(brain_minY + brain_maxY), 0.5*(brain_minZ + brain_maxZ)

# create and write (.msh) tetraedral box
########################################

from utils.mesh_creator import create_box_with_tetraedric_elements, create_MSH_ellipsoid


mesh = create_MSH_ellipsoid.create_MSH_ellipsoid_mesh(mesh_bb_MSH_path, 
                                                      COG_bb[0], COG_bb[1], COG_bb[2],
                                                      1, 
                                                      (dX + 4e-3)*0.75, (dY + 4e-3)*0.75, (dZ + 4e-3)*0.75, 
                                                      hmean, hmean, 
                                                      1,
                                                      visualisation)



# convert mesh into .xdmf format
################################
mesh_bb_XDMF_path = "./metrics/RMSE/RMSE_bounding_ellipsoid_hmean.xdmf"
meshio.write(mesh_bb_XDMF_path, meshio.Mesh(points=mesh.points, cells={'tetra': mesh.cells_dict['tetra']}))

bounding_box_mesh = fenics.Mesh()
with fenics.XDMFFile(mesh_bb_XDMF_path) as infile: 
	infile.read(bounding_box_mesh) 

Z_bb = fenics.FunctionSpace(bounding_box_mesh, "DG", 0) 

# Project function
##################
#markers_bb = fenics.MeshFunction('size_t', bounding_box_mesh, bounding_box_mesh.topology().dim(), 0) 
markers_bb_num = fenics.MeshFunction("size_t", bounding_box_mesh, bounding_box_mesh.topology().dim() - 1) 
markers_bb_dHCP = fenics.MeshFunction("size_t", bounding_box_mesh, bounding_box_mesh.topology().dim() - 1) 

# Mark mesh_brain_numerical_28 within the bounding_box_mesh
###########################################################
class meshNumerical(fenics.SubDomain): # defined on domain dx_bb
    
    def __init__(self, mesh_brain_numerical_28):
            fenics.SubDomain.__init__(self)
            self.mesh_brain_numerical_28 = mesh_brain_numerical_28
            #self.bounding_box_mesh = bounding_box_mesh
            self.tree = fenics.BoundingBoxTree()
            self.tree.build(self.mesh_brain_numerical_28) 
            
    def inside(self, x, on_boundary):
        #if fenics.Point(*x)[:] in self.mesh_brain_numerical_28.coordinates():
        #tree = fenics.BoundingBoxTree(self.mesh_brain_numerical_28 , self.mesh_brain_numerical_28 .geometry().dim())
        
        cell_id = self.tree.compute_first_entity_collision(fenics.Point(*x))
        return cell_id < self.mesh_brain_numerical_28.num_cells()
            
region_numerical = meshNumerical(mesh_brain_numerical_28)
region_numerical.mark(markers_bb_num, 1)

dx_bb_num = fenics.Measure('dx', domain=bounding_box_mesh, subdomain_data=markers_bb_num)

# Mark mesh_brain_dHCP_28 within the bounding_box_mesh
###########################################################
class meshDHCP(fenics.SubDomain): # defined on domain dx_bb
    
    def __init__(self, mesh_brain_dHCP_28):
            fenics.SubDomain.__init__(self)
            self.mesh_brain_dHCP_28 = mesh_brain_dHCP_28
            #self.bounding_box_mesh = bounding_box_mesh
            self.tree = fenics.BoundingBoxTree()
            self.tree.build(self.mesh_brain_dHCP_28) 
            
    def inside(self, x, on_boundary):
        #if fenics.Point(*x)[:] in self.mesh_brain_dHCP_28.coordinates():
        #tree = fenics.BoundingBoxTree(self.mesh_brain_dHCP_28 , self.mesh_brain_dHCP_28 .geometry().dim())
        
        cell_id = self.tree.compute_first_entity_collision(fenics.Point(*x))
        return cell_id < self.mesh_brain_dHCP_28.num_cells()
            
region_dHCP = meshDHCP(mesh_brain_dHCP_28)
region_dHCP.mark(markers_bb_dHCP, 2)

dx_bb_dHCP = fenics.Measure('dx', domain=bounding_box_mesh, subdomain_data=markers_bb_dHCP)

# check markers
###############
from utils.export_functions import export_XML_PVD_XDMF

export_XML_PVD_XDMF.export_PVDfile("./metrics/RMSE/", 'markers_numerical_within_bb', markers_bb_num)
export_XML_PVD_XDMF.export_PVDfile("./metrics/RMSE/", 'markers_dHCP_within_bb', markers_bb_dHCP)

# Project function
##################
def local_project(v_value_to_project, V, u_FEM, dx): 
    """source: https://fenicsproject.org/olddocs/dolfin/2019.1.0/python/demos/elastodynamics/demo_elastodynamics.py.html"""

    dv = fenics.TrialFunction(V) 
    v_ = fenics.TestFunction(V)
    a_proj = fenics.inner(dv, v_) * dx 
    b_proj = fenics.inner(v_value_to_project, v_) * dx
    solver = fenics.LocalSolver(a_proj, b_proj)
    solver.factorize()
    solver.solve_local_rhs(u_FEM)

    return
        
# Projeter mesh functions on the respectively marked regions only
#################################################################
markers_brain_numerical_28_Zbb = fenics.Function(Z_bb)
# markers_brain_numerical_28_ZbrainNum.set_allow_extrapolation(True) # enable to project markers_brain_numerical_28_ZbrainNum onto Z_bb

markers_brain_dHCP_28_Zbb = fenics.Function(Z_bb)
#markers_brain_dHCP_28_ZbrainDHCP.set_allow_extrapolation(True) # enable to project markers_brain_dHCP_28_ZbrainDHCP onto Z_bb

#v = fenics.TestFunction(Z_bb)

#a_num = fenics.inner(markers_brain_numerical_28_ZbrainNum, v) * dx_bb(1)
#fenics.solve(fenics.lhs(a_num) == fenics.rhs(a_num), markers_brain_numerical_28_Zbb)
markers_brain_numerical_28_ZbrainNum.set_allow_extrapolation(True)
local_project(markers_brain_numerical_28_ZbrainNum, Z_bb, markers_brain_numerical_28_Zbb, dx_bb_num(1)) 

#a_dHCP = fenics.inner(markers_brain_dHCP_28_ZbrainDHCP, v) * dx_bb(2)
#fenics.solve(fenics.lhs(a_dHCP) == fenics.rhs(a_dHCP), markers_brain_dHCP_28_Zbb)
markers_brain_dHCP_28_ZbrainDHCP.set_allow_extrapolation(True)
local_project(markers_brain_dHCP_28_ZbrainDHCP, Z_bb, markers_brain_dHCP_28_Zbb, dx_bb_dHCP(2))

dx_bb = fenics.Measure('dx', domain=bounding_box_mesh)
non_superposed_volume = fenics.assemble( abs(markers_brain_dHCP_28_Zbb - markers_brain_numerical_28_Zbb) * dx_bb ) # keep tetraedrons that are non common 
#print('Surf non superpose (m^2) : ',float(Non_Superposition))

error_superposition = (volume_brain_dHCP_28 - non_superposed_volume) / volume_brain_dHCP_28
print('Error superposition {}%: '.format(float(1 - error_superposition) * 100))

exit()

# import vedo.dolfin
# vedo.dolfin.plot(markers_brain_dHCP_28_ZbrainDHCP)
# vedo.dolfin.plot(markers_brain_dHCP_28_Zbb)

















