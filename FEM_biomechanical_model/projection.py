import fenics

# Get growth tensor Function
############################
def local_project(v, V_FEMspace, v_FEMfunction): # e.g. local_project(Fg_TAN + Fg_RAD, Vtensor, Fg) --> project the "valv_FEMfunctione" Fg_TAN + Fg_RAD sv_FEMfunctionr l'espace des tensev_FEMfunctionrs Vtensor. Le résultat de la projection est : Fg
    """sov_FEMfunctionrce: https://fenicsproject.org/olddocs/dolfin/2019.1.0/python/demos/elastodynamics/demo_elastodynamics.py.html"""

    dv = fenics.TrialFunction(V_FEMspace) 
    v_ = fenics.TestFunction(V_FEMspace)
    a_proj = fenics.inner(dv, v_) * fenics.Measure("dx") 
    b_proj = fenics.inner(v, v_) * fenics.Measure("dx") 
    solver = fenics.LocalSolver(a_proj, b_proj)
    solver.factorize()
    if v_FEMfunction is None:
        v_FEMfunction = fenics.Function(V_FEMspace)
        solver.solve_local_rhs(v_FEMfunction)
        return v_FEMfunction
    else:
        solver.solve_local_rhs(v_FEMfunction)
        return