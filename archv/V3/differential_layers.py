import fenics

def compute_distance_to_cortexsurface(vertex2dofs_S, d2s_S, mesh, bmesh_cortexsurface_bbtree):
    
    for idx, x in enumerate(mesh.coordinates()): # point: idx (x[0]), coords (x[1])
        _, distance_to_cortexsurface_x = bmesh_cortexsurface_bbtree.compute_closest_entity(fenics.Point(*x)) # TODO: need to recompute distance_Point_2_closestNodecortexsurfacebBdry?
        d2s_S.vector()[vertex2dofs_S[idx]] = distance_to_cortexsurface_x
    
    return d2s_S

"""
def compute_differential_term(d2s, H): # gm

    try:
        exp_term = fenics.exp(10 * (d2s/H - 1)) 
    except OverflowError:
        exp_term = float('inf') # infini

    gm = 1 / ( 1 + exp_term )

    return gm
"""


def compute_differential_term_DOF(S, d2s_S, H_S, gm_S): # gm

    for dof in S.dofmap().dofs():
        try:
            exp_term = fenics.exp(10 * (d2s_S.vector()[dof]/H_S.vector()[dof] - 1)) 
        except OverflowError:
            exp_term = float('inf') # infini

        gm_S.vector()[dof] = 1 / ( 1 + exp_term )

    return gm_S


def compute_stiffness(gm, muCore, muCortex): # for each node
    return muCore * (1.0 - gm) + muCortex * gm 

"""
def compute_stiffness_DOF(gm_S, muCore, muCortex): # for each node
    mu = muCore * (1.0 - gm_S) + muCortex * gm_S 
    mu_S = fenics.project(mu, S)
    return mu_S
"""