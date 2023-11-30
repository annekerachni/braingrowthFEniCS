import fenics

# Get growth tensor Function
############################
def local_project(v, V, u): # e.g. local_project(Fg_TAN + Fg_RAD, Vtensor, Fg) --> project the "value" Fg_TAN + Fg_RAD sur l'espace des tenseurs Vtensor. Le résultat de la projection est : Fg
    """Element-wise projection using LocalSolver"""

    dv = fenics.TrialFunction(V) 
    v_ = fenics.TestFunction(V)
    a_proj = fenics.inner(dv, v_) * fenics.Measure("dx") 
    b_proj = fenics.inner(v, v_) * fenics.Measure("dx") 
    solver = fenics.LocalSolver(a_proj, b_proj)
    solver.factorize()
    solver.solve_local_rhs(u)

    return