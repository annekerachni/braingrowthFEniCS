import fenics
import numpy as np

# Growth tensor definition
# ------------------------
def compute_growth_ponderation(): # gr
    return

def compute_topboundary_normals(mesh, ds, V): 

    u_trial = fenics.TrialFunction(V)
    v_test = fenics.TestFunction(V)
    Boundary_Nt = fenics.Function(V)
    
    n = fenics.FacetNormal(mesh)
    a = fenics.inner(u_trial, v_test) * ds # ds(101)  
    l = fenics.inner(n, v_test) * ds # ds(101)     
    A = fenics.assemble(a, keep_diagonal=True)
    L = fenics.assemble(l)
    A.ident_zeros()
    fenics.solve(A, Boundary_Nt.vector(), L)

    return Boundary_Nt 

def compute_mesh_projected_normals(V, mesh_vertex_coords, bmesh_vertex_coords, vertexB_2_dofsV_mapping, vertex2dofs_V, Boundary_Nt):
    
    Mesh_Nt = fenics.Function(V)

    # Project B normals to all mesh Points/Vertices
    for vertex_index, coords in enumerate(mesh_vertex_coords): #N vertex indices in the whole Mesh ref
        
        # Compute vertex index (int) in B (Function Space) reference (i.e. less indices) of the closest Point at Top Boundary to given Mesh Point ('vertex_index')   
        closestBPointIndex_inBref = np.argmin(np.linalg.norm(bmesh_vertex_coords - mesh_vertex_coords[vertex_index], axis=1)) # B index (in the B ref) for 2D closest Point/Vertex
        closestBPointDOFs_inVref = vertexB_2_dofsV_mapping[closestBPointIndex_inBref]

        # Allocate to given mesh Point ('vertex_index') the normal vector of the closest Point at Top Boundary (=> get projected normal vector for all mesh Points) 
        Mesh_Nt.vector()[vertex2dofs_V[vertex_index]] = Boundary_Nt.vector()[closestBPointDOFs_inVref] 
        
    return Mesh_Nt 

def compute_growth_tensor(Mesh_Nt, dg_TAN, dg_RAD, gdim): 
    """growth tensor"""
    Id = fenics.Identity(gdim)

    Nt_Nt_proj = fenics.outer(Mesh_Nt, Mesh_Nt) 

    Fg_TAN = (1 + dg_TAN) * ( Id - Nt_Nt_proj ) 
    Fg_RAD = (1 + dg_RAD) * Nt_Nt_proj 
    Fg = Fg_TAN + Fg_RAD 

    return Fg