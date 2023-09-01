import fenics

# File-writing functions
########################
def export_XMLfile(output_folderpath, name, mesh):
    fenics.File( str(output_folderpath + name + '.xml') ) << mesh
    print('tetra_{}.xml was written'.format(name))
    return 

def export_PVDfile(output_folderpath, name, geometry_entity):
    fenics.File( str(output_folderpath + name + '.pvd') ).write(geometry_entity)
    print('{}.pvd was written'.format(name))
    return 

def export_XDMFfile(output_folderpath, value):
    # Set name and options for elastodynamics solution XDMF file export
    results_file_path = str(output_folderpath) + str(value) + ".xdmf"
    xdmf_file = fenics.XDMFFile(results_file_path)
    xdmf_file.parameters["flush_output"] = True
    xdmf_file.parameters["functions_share_mesh"] = True
    xdmf_file.parameters["rewrite_function_mesh"] = True
    return xdmf_file

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
    