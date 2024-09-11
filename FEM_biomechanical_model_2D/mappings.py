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

###########
###########

def cortexsurface_to_mesh_V(gdim, V, V_cortexsurface, vertex2dofs_Vcortexsurface):
    """
    Build mappings between the boundary mesh and the whole mesh in the 3D Vector Function Space (V):
    - DOFs (dim 3) in the vector function space of the boundary mesh --> DOFs (dim 3) in the vector function space of the whole mesh
    - vertex index of a Point in the boundary mesh --> DOFs (dim 3) in the vector function space of the whole mesh
    """

    # From the surface mesh (cortex envelop) to the whole mesh (B_2_V_dofmap; vertexB_2_dofsV_mapping --> used to compute Mesh_Nt)
    # --------------------------------------------------------
    DOF_coords_V = V.tabulate_dof_coordinates()
    DOF_coords_Vcortexsurface = V_cortexsurface.tabulate_dof_coordinates()

    # find (redundant) triplet-DOFsV corresponding to each single DOF_Vcortexsurface
    tree_V = cKDTree(DOF_coords_V)
    _, singleDOFVboundary_2_coupledDOFsV_dofmap = tree_V.query(DOF_coords_Vcortexsurface, k=gdim) 

    # supress couple or triplet-DOFsV redundancies
    coupledDOFsV_dofmap = []
    for dofVboundary in prange(len(singleDOFVboundary_2_coupledDOFsV_dofmap)):
        if dofVboundary % gdim == 0:
            coupledDOFsV_dofmap.append(singleDOFVboundary_2_coupledDOFsV_dofmap[dofVboundary])
    coupledDOFsV_dofmap = np.array(coupledDOFsV_dofmap)

    # sort each triplet-DOFsV with first DOF the smallest value
    for DOFscouple_idx in prange(len(coupledDOFsV_dofmap)):
        coupledDOFsV_dofmap[DOFscouple_idx].sort()

    # couple or triplet-DOFs_Vcortexsurface
    DOFs_Vcortexsurface = vertex2dofs_Vcortexsurface 
    DOFs_Vcortexsurface[DOFs_Vcortexsurface[:, 0].argsort()] 

    # Build dofmap
    coupledDOFsV_dofmap = coupledDOFsV_dofmap.flatten() 
    DOFs_Vcortexsurface = DOFs_Vcortexsurface.flatten() 
    Vcortexsurface_2_V_dofmap = []
    for dofs_Vcortexsurface in prange(len(DOFs_Vcortexsurface)):
        Vcortexsurface_2_V_dofmap.append(coupledDOFsV_dofmap[dofs_Vcortexsurface])

    Vcortexsurface_2_V_dofmap = np.array(Vcortexsurface_2_V_dofmap)

    # vertex to DOF_V_cortexsurface_ref --> vertex to DOF_V_ref
    vertexCortexSurfaceMesh_2_DOFsVWholeMesh_mapping = [] 
    
    if gdim == 2:
        for vertex, dofs_Vcortexsurface in enumerate(vertex2dofs_Vcortexsurface):
            dof_Vcortexsurface_1 = dofs_Vcortexsurface[0]
            dof_Vcortexsurface_2 = dofs_Vcortexsurface[1]

            dof_V_1 = Vcortexsurface_2_V_dofmap[dof_Vcortexsurface_1]
            dof_V_2 = Vcortexsurface_2_V_dofmap[dof_Vcortexsurface_2]

            vertexCortexSurfaceMesh_2_DOFsVWholeMesh_mapping.append([dof_V_1, dof_V_2])
        
    elif gdim == 3:
        for vertex, dofs_Vcortexsurface in enumerate(vertex2dofs_Vcortexsurface):
            dof_Vcortexsurface_1 = dofs_Vcortexsurface[0]
            dof_Vcortexsurface_2 = dofs_Vcortexsurface[1]
            dof_dof_Vcortexsurface_3 = dofs_Vcortexsurface[2]

            dof_V_1 = Vcortexsurface_2_V_dofmap[dof_Vcortexsurface_1]
            dof_V_2 = Vcortexsurface_2_V_dofmap[dof_Vcortexsurface_2]
            dof_V_3 = Vcortexsurface_2_V_dofmap[dof_dof_Vcortexsurface_3]

            vertexCortexSurfaceMesh_2_DOFsVWholeMesh_mapping.append([dof_V_1, dof_V_2, dof_V_3])

    vertexCortexSurfaceMesh_2_DOFsVWholeMesh_mapping = np.asarray(vertexCortexSurfaceMesh_2_DOFsVWholeMesh_mapping)

    return Vcortexsurface_2_V_dofmap, vertexCortexSurfaceMesh_2_DOFsVWholeMesh_mapping


def cortexsurface_to_mesh_S(S, S_cortexsurface):
    """
    Build mappings between the boundary mesh and the whole mesh in the Scalar Function Space (S):
    - DOF (dim 1) in the scalar function space of the boundary mesh --> DOF (dim 1) in the scalar function space of the whole mesh
    - vertex index of a Point in the boundary mesh --> DOF (dim 1) in the scalar function space of the whole mesh
    """

    # From the surface mesh (cortex envelop) to the whole mesh (Vcortexsurface_2_V_dofmap; vertexcortexsurface_2_dofsV_mapping --> used to compute Mesh_Nt)
    # --------------------------------------------------------
    DOF_coords_S = S.tabulate_dof_coordinates()
    DOF_coords_Scortexsurface = S_cortexsurface.tabulate_dof_coordinates()
    
    vertex2dof_S = fenics.vertex_to_dof_map(S)
    vertex2dof_Scortexsurface = fenics.vertex_to_dof_map(S_cortexsurface)
    
    # find DOF_S corresponding to each single DOF_Scortexsurface (single DOF in the Scalar Space)
    tree_S = cKDTree(DOF_coords_S) # the whole mesh
    _, Scortexsurface_2_S_dofmap = tree_S.query(DOF_coords_Scortexsurface) # gdim=1 since scalar dimension

    
    # Build dofmap: S cortex surface (DOF) --> S whole mesh (DOF) 
    """   
    Scortexsurface_2_S_dofmap = [] 
    for dof_Scortexsurface in prange(len(vertex2dof_Scortexsurface)):
        Scortexsurface_2_S_dofmap.append(Scortexsurface_2_S_dofmap[dof_Scortexsurface])

    Scortexsurface_2_S_dofmap = np.array(Scortexsurface_2_S_dofmap)   
    """ 

    # Build mapping: vertex indexed in the cortex surface --> DOF indexed in the Scalar (S) Function Space in the whole mesh
    vertexBoundaryMesh_2_dofScalarFunctionSpaceWholeMesh_mapping = []
    for vertex, dof_Scortexsurface in enumerate(vertex2dof_Scortexsurface):       
        dof_S = Scortexsurface_2_S_dofmap[dof_Scortexsurface] # dof_S --> dof_S_WholeMesh
        vertexBoundaryMesh_2_dofScalarFunctionSpaceWholeMesh_mapping.append(dof_S)

    vertexBoundaryMesh_2_dofScalarFunctionSpaceWholeMesh_mapping = np.asarray(vertexBoundaryMesh_2_dofScalarFunctionSpaceWholeMesh_mapping)

    return Scortexsurface_2_S_dofmap, vertexBoundaryMesh_2_dofScalarFunctionSpaceWholeMesh_mapping 

###########
###########

def wholemesh_to_cortexsurface_vertexmap(mesh_vertex_coords, bmesh_vertex_coords):
    """For a given mesh vertex index (V ref), get the closest vertex index (B ref) on boundary"""

    vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping = []
    
    # Project B normals to all mesh Points/Vertices
    for vertex_index, coords in enumerate(mesh_vertex_coords): #N vertex indices in the whole Mesh ref
        
        # Compute vertex index (int) in B (Function Space) reference (i.e. less indices) of the closest Point at Top Boundary to given Mesh Point ('vertex_index')   
        closestBPointIndex_inBref = np.argmin(np.linalg.norm(bmesh_vertex_coords - mesh_vertex_coords[vertex_index], axis=1)) # B index (in the B ref) for 2D closest Point/Vertex
        vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping.append(closestBPointIndex_inBref)  # build mapping: vertex IDX i in whole mesh --> projected (onto boundary mesh) vertex IDX (B ref)  
        
    vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping = np.array(vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping)
    
    return vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping 