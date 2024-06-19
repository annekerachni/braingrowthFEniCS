import fenics

def compute_distance_to_cortexsurface(vertex2dofs_S, d2s_S, mesh, bmesh_cortexsurface_bbtree):
    
    for idx, x in enumerate(mesh.coordinates()): # point: idx (x[0]), coords (x[1])
        _, distance_to_cortexsurface_x = bmesh_cortexsurface_bbtree.compute_closest_entity(fenics.Point(*x)) 
        d2s_S.vector()[vertex2dofs_S[idx]] = distance_to_cortexsurface_x
        
    return d2s_S

def compute_distance_to_cortexsurface_2(vertex2dofs_S, d2s_S, mesh, bmesh_cortexsurface_tree):
    
    for idx, x in enumerate(mesh.coordinates()): # point: idx (x[0]), coords (x[1])
        distance_to_cortexsurface_x, idex_of_node_in_mesh = bmesh_cortexsurface_tree.query(x)  
        d2s_S.vector()[vertex2dofs_S[idx]] = distance_to_cortexsurface_x
        
    return d2s_S


def compute_differential_term_DOF(S, d2s_S, H_S, gm_S): # gm

    for dof in S.dofmap().dofs():
        try:
            exp_term = fenics.exp(10 * (d2s_S.vector()[dof]/H_S.vector()[dof] - 1)) 
        except OverflowError:
            exp_term = float('inf') # infini

        gm_S.vector()[dof] = 1 / ( 1 + exp_term )

    return gm_S


def compute_shear_and_bulk_stiffnesses(gm, muCore, muCortex): # for each node
    return muCore * (1.0 - gm) + muCortex * gm 