import fenics

# Get growth tensor Function
############################
def local_project(u, FunctionSpace, u_FEM): # e.g. local_project(Fg_TAN + Fg_RAD, Vtensor, Fg) --> project the "value" Fg_TAN + Fg_RAD sur l'espace des tenseurs Vtensor. Le résultat de la projection est : Fg
    """source: https://fenicsproject.org/olddocs/dolfin/2019.1.0/python/demos/elastodynamics/demo_elastodynamics.py.html"""

    dv = fenics.TrialFunction(FunctionSpace) 
    v_ = fenics.TestFunction(FunctionSpace)
    a_proj = fenics.inner(dv, v_) * fenics.Measure("dx") 
    b_proj = fenics.inner(u, v_) * fenics.Measure("dx") 
    solver = fenics.LocalSolver(a_proj, b_proj)
    solver.factorize()
    solver.solve_local_rhs(u_FEM)

    return