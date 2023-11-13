import fenics
from scipy.spatial import cKDTree
from numba import prange
import numpy as np 

# Mappings
##########
def vertex_to_dof_scalars(S):
    vertex2dofs_S = fenics.vertex_to_dof_map(S)
    return vertex2dofs_S

def vertex_to_dof_vectors(V, gdim):

    vertex2dofs = fenics.vertex_to_dof_map(V)
    vertex2dofs = vertex2dofs.reshape((-1, gdim)) 

    return vertex2dofs

def surface_to_mesh(gdim, V, V_cortexsurface, vertex2dofs_B):

    # From the surface mesh (cortex envelop) to the whole mesh (B_2_V_dofmap; vertexB_2_dofsV_mapping --> used to compute Mesh_Nt)
    # --------------------------------------------------------
    DOF_coords_V = V.tabulate_dof_coordinates()
    DOF_coords_B = V_cortexsurface.tabulate_dof_coordinates()

    # find (redundant) triplet-DOFsV corresponding to each single DOFB
    Vtree = cKDTree(DOF_coords_V)
    _, singleDOFB_2_coupledDOFsV_dofmap = Vtree.query(DOF_coords_B, k=gdim) 

    # supress triplet-DOFsV redundancies
    coupledDOFsV_dofmap = []
    for dofB in prange(len(singleDOFB_2_coupledDOFsV_dofmap)):
        if dofB % gdim == 0:
            coupledDOFsV_dofmap.append(singleDOFB_2_coupledDOFsV_dofmap[dofB])
    coupledDOFsV_dofmap = np.array(coupledDOFsV_dofmap)

    # sort each triplet-DOFsV with first DOF the smallest value
    for DOFscouple_idx in prange(len(coupledDOFsV_dofmap)):
        coupledDOFsV_dofmap[DOFscouple_idx].sort()

    # triplet-DOFsB
    DOFsB = vertex2dofs_B 
    DOFsB[DOFsB[:, 0].argsort()] 

    # Build dofmap
    coupledDOFsV_dofmap = coupledDOFsV_dofmap.flatten() 
    DOFsB = DOFsB.flatten() 
    B_2_V_dofmap = []
    for dofB in prange(len(DOFsB)):
        B_2_V_dofmap.append(coupledDOFsV_dofmap[dofB])

    B_2_V_dofmap = np.array(B_2_V_dofmap)

    # vertex to DOF_Bref --> vertex to DOF_Vref
    vertexB_2_dofsV_mapping = []
    for vertex, dofB in enumerate(vertex2dofs_B):
        dof_B_1 = dofB[0]
        dof_B_2 = dofB[1]
        dof_B_3 = dofB[2]

        dof_V_1 = B_2_V_dofmap[dof_B_1]
        dof_V_2 = B_2_V_dofmap[dof_B_2]
        dof_V_3 = B_2_V_dofmap[dof_B_3]

        vertexB_2_dofsV_mapping.append([dof_V_1, dof_V_2, dof_V_3])

    vertexB_2_dofsV_mapping = np.asarray(vertexB_2_dofsV_mapping)

    return B_2_V_dofmap, vertexB_2_dofsV_mapping