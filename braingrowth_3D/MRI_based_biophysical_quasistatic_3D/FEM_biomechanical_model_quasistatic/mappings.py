import fenics
from scipy.spatial import cKDTree
from numba import prange
import numpy as np 


def vertex_to_dof_ScalarFunctionSpace(S):
    """Mapping between the index of a Point and the associated DOF (dim 1) in the Scalar Function Space"""
    vertex2dof_S = fenics.vertex_to_dof_map(S)
    return vertex2dof_S


def vertex_to_dofs_VectorFunctionSpace(V, gdim):
    """Mapping between the index of a Point and the associated DOFs (dim 3) in the 3D Vector Function Space"""
    vertex2dofs_V = fenics.vertex_to_dof_map(V)
    vertex2dofs_V = vertex2dofs_V.reshape((-1, gdim)) 
    return vertex2dofs_V

###

def surface_to_mesh_V(gdim, V, V_cortexsurface, vertex2dofs_Vboundary):
    """
    Build mappings between the boundary mesh and the whole mesh in the 3D Vector Function Space (V):
    - DOFs (dim 3) in the vector function space of the boundary mesh --> DOFs (dim 3) in the vector function space of the whole mesh
    - vertex index of a Point in the boundary mesh --> DOFs (dim 3) in the vector function space of the whole mesh
    """

    # From the surface mesh (cortex envelop) to the whole mesh (B_2_V_dofmap; vertexB_2_dofsV_mapping --> used to compute Mesh_Nt)
    # --------------------------------------------------------
    DOF_coords_V = V.tabulate_dof_coordinates()
    DOF_coords_V_boundary = V_cortexsurface.tabulate_dof_coordinates()

    # find (redundant) triplet-DOFsV corresponding to each single DOFB
    Vtree = cKDTree(DOF_coords_V)
    _, singleDOFVboundary_2_coupledDOFsV_dofmap = Vtree.query(DOF_coords_V_boundary, k=gdim) 

    # supress triplet-DOFsV redundancies
    coupledDOFsV_dofmap = []
    for dofVboundary in prange(len(singleDOFVboundary_2_coupledDOFsV_dofmap)):
        if dofVboundary % gdim == 0:
            coupledDOFsV_dofmap.append(singleDOFVboundary_2_coupledDOFsV_dofmap[dofVboundary])
    coupledDOFsV_dofmap = np.array(coupledDOFsV_dofmap)

    # sort each triplet-DOFsV with first DOF the smallest value
    for DOFscouple_idx in prange(len(coupledDOFsV_dofmap)):
        coupledDOFsV_dofmap[DOFscouple_idx].sort()

    # triplet-DOFs_Vboundary
    DOFsB = vertex2dofs_Vboundary 
    DOFsB[DOFsB[:, 0].argsort()] 

    # Build dofmap
    coupledDOFsV_dofmap = coupledDOFsV_dofmap.flatten() 
    DOFsB = DOFsB.flatten() 
    Vboundary_2_V_dofmap = []
    for dofB in prange(len(DOFsB)):
        Vboundary_2_V_dofmap.append(coupledDOFsV_dofmap[dofB])

    Vboundary_2_V_dofmap = np.array(Vboundary_2_V_dofmap)

    # vertex to DOF_V_boundary_ref --> vertex to DOF_V_ref
    vertexBoundaryMesh_2_DOFsVectorFunctionSpaceWholeMesh_mapping = [] 
    for vertex, dofB in enumerate(vertex2dofs_Vboundary):
        dof_B_1 = dofB[0]
        dof_B_2 = dofB[1]
        dof_B_3 = dofB[2]

        dof_V_1 = Vboundary_2_V_dofmap[dof_B_1]
        dof_V_2 = Vboundary_2_V_dofmap[dof_B_2]
        dof_V_3 = Vboundary_2_V_dofmap[dof_B_3]

        vertexBoundaryMesh_2_DOFsVectorFunctionSpaceWholeMesh_mapping.append([dof_V_1, dof_V_2, dof_V_3])

    vertexBoundaryMesh_2_DOFsVectorFunctionSpaceWholeMesh_mapping = np.asarray(vertexBoundaryMesh_2_DOFsVectorFunctionSpaceWholeMesh_mapping)

    return Vboundary_2_V_dofmap, vertexBoundaryMesh_2_DOFsVectorFunctionSpaceWholeMesh_mapping


def surface_to_mesh_S(S, S_cortexsurface):
    """
    Build mappings between the boundary mesh and the whole mesh in the Scalar Function Space (S):
    - DOF (dim 1) in the scalar function space of the boundary mesh --> DOF (dim 1) in the scalar function space of the whole mesh
    - vertex index of a Point in the boundary mesh --> DOF (dim 1) in the scalar function space of the whole mesh
    """

    # From the surface mesh (cortex envelop) to the whole mesh (B_2_V_dofmap; vertexB_2_dofsV_mapping --> used to compute Mesh_Nt)
    # --------------------------------------------------------
    DOF_coords_S = S.tabulate_dof_coordinates()
    DOF_coords_S_boundary = S_cortexsurface.tabulate_dof_coordinates()
    
    vertex2dof_S = fenics.vertex_to_dof_map(S)
    vertex2dof_S_boundary = fenics.vertex_to_dof_map(S_cortexsurface)
    
    # find DOF_S corresponding to each single DOF_S_boundary (single DOF in the Scalar Space)
    tree_S = cKDTree(DOF_coords_S) # the whole mesh
    _, DOF_S_boundary_2_DOF_S_mesh_dofmap = tree_S.query(DOF_coords_S_boundary) # gdim=1 since scalar dimension

    
    # Build dofmap: S boundary (DOF) --> S whole mesh (DOF)    
    Sboundary_2_S_dofmap = [] 
    for dof_S_boundary in prange(len(vertex2dof_S_boundary)):
        Sboundary_2_S_dofmap.append(DOF_S_boundary_2_DOF_S_mesh_dofmap[dof_S_boundary])

    Sboundary_2_S_dofmap = np.array(Sboundary_2_S_dofmap)
    

    # Build mapping: vertex indexed in the mesh boundary --> DOF indexed in the Scalar (S) Function Space in the whole mesh
    vertexBoundaryMesh_2_dofScalarFunctionSpaceWholeMesh_mapping = []
    for vertex, dof_S_boundary in enumerate(vertex2dof_S_boundary):       
        dof_S = Sboundary_2_S_dofmap[dof_S_boundary] # dof_S --> dof_S_WholeMesh
        vertexBoundaryMesh_2_dofScalarFunctionSpaceWholeMesh_mapping.append(dof_S)

    vertexBoundaryMesh_2_dofScalarFunctionSpaceWholeMesh_mapping = np.asarray(vertexBoundaryMesh_2_dofScalarFunctionSpaceWholeMesh_mapping)

    return Sboundary_2_S_dofmap, vertexBoundaryMesh_2_dofScalarFunctionSpaceWholeMesh_mapping 

###

def mesh_to_surface_V(mesh_vertex_coords, bmesh_vertex_coords):
    """For a given mesh vertex index (V ref), get the closest vertex index (B ref) on boundary"""

    vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping = []
    
    # Project B normals to all mesh Points/Vertices
    for vertex_index, coords in enumerate(mesh_vertex_coords): #N vertex indices in the whole Mesh ref
        
        # Compute vertex index (int) in B (Function Space) reference (i.e. less indices) of the closest Point at Top Boundary to given Mesh Point ('vertex_index')   
        closestBPointIndex_inBref = np.argmin(np.linalg.norm(bmesh_vertex_coords - mesh_vertex_coords[vertex_index], axis=1)) # B index (in the B ref) for 2D closest Point/Vertex
        vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping.append(closestBPointIndex_inBref)  # build mapping: vertex IDX i in whole mesh --> projected (onto boundary mesh) vertex IDX (B ref)  
        
    vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping = np.array(vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping)
    
    return vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping 