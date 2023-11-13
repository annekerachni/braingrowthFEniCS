# elastodynamic brain growth model (dynamic structural mechanics)

""" python3 -i simulation_solverFg0.py -i './data/dhcp/refined/dhcp21GW_17K_refined10.xml' -n True -p '{"H0": 0.04, "K": 100.0, "muCortex": 20.0, "muCore": 1.0, "rho": 0.01, "damping_coef": 10.0, "alphaTAN": 5.0, "alphaRAD": 0.0, "grTAN": 1.0, "grRAD": 1.0, "alphaM": 0.2, "alphaF": 0.4, "T0": 0.0, "Tmax": 1.0, "Nsteps": 50, "linearization_method":"newton", "linear_solver":"gmres", "preconditioner":"sor"}' -o './simulations/dhcp/12nov/' """

import fenics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import vedo.dolfin
import time
import argparse
import json
import time

import preprocessing, numerical_scheme_temporal, numerical_scheme_spatial, mappings, contact, differential_layers, growth, projection
from utils.export_functions import export_simulation_outputmesh_data, export_simulation_end_time_and_iterations, export_XML_PVD_XDMF
from utils.converters import convert_meshformats_to_vtk


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='braingrowthFEniCS: brain growth elastodynamics 3D model')

    parser.add_argument('-i', '--input', help='Input mesh path (xml)', type=str, required=False, 
                        default='./data/dhcp_atlas/dhcp21GW_17K_refined10.xml') # './data/dhcp_atlas/dhcp21GW_29296faces_107908tets_refined10.xml' 
                        
    parser.add_argument('-n', '--normalization', help='Is normalization of the input mesh required? (required by braingrowthFEniCS)', type=bool, required=False, default=True)
    
    parser.add_argument('-p', '--parameters', help='Simulation input parameters', type=json.loads, required=False, 
                        default={"H0": 0.03, 
                                 "K": 100.0, 
                                 "muCortex": 20.0, "muCore": 1.0, 
                                 "rho": 0.01, 
                                 "damping_coef": 10.0,
                                 "alphaTAN": 4.0, "alphaRAD": 0.0, "grTAN": 1.0, "grRAD": 1.0, 
                                 "alphaM": 0.2, "alphaF": 0.4, 
                                 "T0": 0.0, "Tmax": 0.5, "Nsteps": 100,
                                 "linearization_method":"newton", "linear_solver":"gmres", "preconditioner":"sor"}) # try with "preconditioner":"jacobi"; "ilu" https://fr.wikipedia.org/wiki/Pr%C3%A9conditionneur
    
    parser.add_argument('-o', '--output', help='Output folder path', type=str, required=False, 
                        default='./simulations/dhcp/')
                                    
    #parser.add_argument('-v', '--visualization', help='Visualization during simulation', type=bool, required=False, default=False)
    parser.add_argument('-v', '--visualization', help='Visualization during simulation', action='store_true')
    
    args = parser.parse_args() 

    # Form compiler options
    #######################
    # See https://fenicsproject.org/olddocs/dolfin/1.6.0/python/demo/documented/hyperelasticity/python/documentation.html
    
    fenics.parameters["form_compiler"]["optimize"] = True
    fenics.parameters["form_compiler"]["cpp_optimize"] = True # The form compiler to use C++ compiler optimizations when compiling the generated code.
    fenics.parameters["form_compiler"]["representation"] = "uflacs"
    fenics.parameters["form_compiler"]["quadrature_degree"] = 7
    fenics.parameters["allow_extrapolation"] = True
    fenics.parameters["std_out_all_processes"] = False

    #fenics.parameters["form_compiler"]["eliminate_zeros"] = True
    #fenics.parameters["form_compiler"]["precompute_basis_const"] = True
    #fenics.parameters["form_compiler"]["precompute_ip_const"] = True


    # Input mesh
    ############

    # Mesh
    # ----
    # mesh & boundary mesh
    print("\nimporting mesh...")
    inputmesh_path = args.input
    mesh = fenics.Mesh(inputmesh_path)
    bmesh = fenics.BoundaryMesh(mesh, "exterior") # bmesh at t=0.0 (cortex envelop)

    if args.visualization == True:
        fenics.plot(mesh) 
        plt.title("input mesh")
        plt.show() 

    # mesh characteristics
    characteristics0 = preprocessing.compute_geometrical_characteristics(mesh, bmesh)
    center_of_gravity0 = preprocessing.compute_center_of_gravity(characteristics0) 
    min_mesh_spacing0, average_mesh_spacing0, max_mesh_spacing0 = preprocessing.compute_mesh_spacing(mesh)
    print('input mesh characteristics: {}'.format(characteristics0))
    print('input mesh COG = [xG0:{}, yG0:{}, zG0:{}]'.format(center_of_gravity0[0], center_of_gravity0[1], center_of_gravity0[2]))
    print("input mesh min mesh spacing: {:.3f} mm".format(min_mesh_spacing0))
    print("input mesh mean mesh spacing: {:.3f} mm".format(average_mesh_spacing0))
    print("input mesh max mesh spacing: {:.3f} mm".format(max_mesh_spacing0))

    # normalization and boundary mesh update, normalized mesh characteristics
    if args.normalization == True:
        print("\nnormalizing mesh...")
        mesh = preprocessing.normalize_mesh(mesh, characteristics0, center_of_gravity0)
        bmesh = fenics.BoundaryMesh(mesh, "exterior") # update bmesh

        if args.visualization == True:
            fenics.plot(mesh) 
            plt.title("normalized mesh")
            plt.show()  
            #vedo.dolfin.plot(mesh, wireframe=False, text='normalized mesh', style='paraview', axes=4).close()
            
        characteristics = preprocessing.compute_geometrical_characteristics(mesh, bmesh) # n_nodes, coordinates, n_tets, n_faces_Surface, n_faces_Volume, minx, maxx, miny, maxy, minz, maxz 
        center_of_gravity = preprocessing.compute_center_of_gravity(characteristics) 
        min_mesh_spacing, average_mesh_spacing, max_mesh_spacing = preprocessing.compute_mesh_spacing(mesh)
        print('normalized mesh characteristics: {}'.format(characteristics))
        print('normalized mesh COG = [xG:{}, yG:{}, zG:{}]'.format(center_of_gravity[0], center_of_gravity[1], center_of_gravity[2]))
        print("normalized min_mesh_spacing: {:.3f} mm".format(min_mesh_spacing))
        print("normalized mesh mean mesh spacing: {:.3f} mm".format(average_mesh_spacing))
        print("normalized mesh max mesh spacing: {:.3f} mm".format(max_mesh_spacing))

    # Export the characteristics of mesh_TO 
    # -------------------------------------
    fenics.File(args.output + "mesh_T0.xml") << mesh
    convert_meshformats_to_vtk.xml_to_vtk(args.output + "mesh_T0.xml", args.output + "mesh_T0.vtk")
    export_simulation_outputmesh_data.export_resultmesh_data(args.output + "analytics/",
                                                             args.output + "mesh_T0.vtk",
                                                             args.parameters["T0"],
                                                             0,
                                                             0.0,
                                                             "mesh_T0.txt")

    # Boundaries
    # ----------
    print("\ncomputing and marking boundaries...")
    bmesh_cortexsurface_bbtree = fenics.BoundingBoxTree()
    bmesh_cortexsurface_bbtree.build(bmesh) 

    # initialize boundaries
    boundaries = fenics.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)  
    boundaries.set_all(100)

    # mark surface
    class CortexSurface(fenics.SubDomain): 

        def __init__(self, bmesh_cortexsurface_bbtree):
            fenics.SubDomain.__init__(self)
            self.bmesh_cortexsurface_bbtree = bmesh_cortexsurface_bbtree

        def inside(self, x, on_boundary): 
            _, distance = self.bmesh_cortexsurface_bbtree.compute_closest_entity(fenics.Point(*x)) # compute_closest_point() https://fenicsproject.org/olddocs/dolfin/1.5.0/python/programmers-reference/cpp/mesh/GenericBoundingBoxTree.html
            return fenics.near(distance, fenics.DOLFIN_EPS) # returns Points

    cortexsurface = CortexSurface(bmesh_cortexsurface_bbtree)
    cortexsurface.mark(boundaries, 101, check_midpoint=False) # https://fenicsproject.discourse.group/t/how-to-compute-boundary-mesh-and-submesh-from-an-halfdisk-mesh/9812/2

    # export marked boundaries
    export_XML_PVD_XDMF.export_PVDfile(args.output, 'boundaries_t0', boundaries)

    # Subdomains
    # ----------
    subdomains = fenics.MeshFunction("size_t", mesh, mesh.topology().dim()) 
    subdomains.set_all(0)


    # Input parameters
    ##################

    # Geometry 
    h = fenics.Expression('H0 + 0.01*t', H0=args.parameters["H0"], t=0.0, degree=0)
    gdim=3

    # Elastic parameters
    K  = fenics.Constant(args.parameters["K"])
    muCortex = fenics.Constant(args.parameters["muCortex"])
    muCore = fenics.Constant(args.parameters["muCore"])

    # Mass density
    rho = fenics.Constant(args.parameters["rho"])

    # Damping coefficients
    damping_coef = fenics.Constant(args.parameters["damping_coef"])

    # Growth parameters
    alphaTAN = fenics.Constant(args.parameters["alphaTAN"])
    grTAN = fenics.Constant(args.parameters["grTAN"])

    alphaRAD = fenics.Constant(args.parameters["alphaRAD"])
    grRAD = fenics.Constant(args.parameters["grRAD"])


    # Time integration
    ##################
    # Generalized-alpha method parameters
    alphaM = fenics.Constant(args.parameters["alphaM"])
    alphaF = fenics.Constant(args.parameters["alphaF"])
    gamma  = fenics.Constant( 0.5 + alphaF - alphaM )
    beta   = fenics.Constant( 0.25 * (gamma + 0.5)**2 )

    # Time-stepping parameters
    T0       = args.parameters["T0"]
    Tmax       = args.parameters["Tmax"]
    Nsteps  = args.parameters["Nsteps"]
    dt = fenics.Constant((Tmax-T0)/Nsteps)
    print('\ntime step: ~{:5} s'.format( float(dt) )) # in original BrainGrowth: dt = 0,000022361 ~ 2.10⁻⁵


    # FEM Function Spaces 
    #####################
    print("\ncreating Lagrange FEM function spaces and functions...")

    # Scalar Function Spaces
    S = fenics.FunctionSpace(mesh, "CG", 1) 
    S_cortexsurface = fenics.FunctionSpace(bmesh, "CG", 1) 

    # Vector Function Spaces
    V = fenics.VectorFunctionSpace(mesh, "CG", 1)
    V_cortexsurface = fenics.VectorFunctionSpace(bmesh, "CG", 1) 

    # Tensor Function Spaces
    #Vtensor = fenics.TensorFunctionSpace(mesh, "DG", 0)
    Vtensor = fenics.TensorFunctionSpace(mesh,'CG', 1, shape=(3,3)) # https://fenicsproject.discourse.group/t/outer-product-evaluation/2159; https://fenicsproject.discourse.group/t/how-to-choose-projection-space-for-stress-tensor-post-processing/5568/4

    # FEM Functions
    ###############

    # Scalar functions of V
    H = fenics.Function(S, name="H") 
    d2s = fenics.Function(S, name="d2s")
    gr = fenics.Function(S, name="gr") 
    gm = fenics.Function(S, name="gm") 
    mu = fenics.Function(S, name="mu") 

    dg_TAN = fenics.Function(S, name="dgTAN")
    dg_RAD = fenics.Function(S, name="dgRAD") 

    # Vector functions of V
    u = fenics.Function(V, name="Displacement") # Trial function. Current (unknown) displacement
    v_test = fenics.TestFunction(V) # Test function

    u_old = fenics.Function(V) # Fields from previous time step (displacement, velocity, acceleration)
    v_old = fenics.Function(V)
    a_old = fenics.Function(V)

    BoundaryMesh_Nt = fenics.Function(V, name="BoundaryMesh_Nt")
    Mesh_Nt = fenics.Function(V, name="Mesh_Nt")

    # Vector functions of Vtensor
    Fg_T = fenics.Function(Vtensor, name="Fg")


    # Mappings
    ##########
    print("\ncomputing mappings...")
    # From vertex to DOF in the whole mesh --> used to compute Mesh_Nt
    # ----------------------------------------------------------------
    vertex2dofs_V = mappings.vertex_to_dofs_VectorFunctionSpace(V, gdim)
    vertex2dofs_B = mappings.vertex_to_dofs_VectorFunctionSpace(V_cortexsurface, gdim)

    # From the surface mesh (cortex envelop) to the whole mesh (B_2_V_dofmap; vertexB_2_dofsV_mapping --> used to compute Mesh_Nt)
    # --------------------------------------------------------
    B_2_V_dofmap, vertexB_2_dofsV_mapping = mappings.surface_to_mesh_V(gdim, V, V_cortexsurface, vertex2dofs_B)
    Sboundary_2_S_dofmap, vertexBoundaryMesh_2_dofScalarFunctionSpaceWholeMesh_mapping = mappings.surface_to_mesh_S(S, S_cortexsurface)
    
    # From the whole mesh to the surface mesh (to be use for projections onto surface in contact process)
    # ---------------------------------------
    vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping_t0 = mappings.mesh_to_surface_V(mesh.coordinates(), bmesh.coordinates()) # at t=0. (keep reference projection before contact for proximity node to deep tetrahedra vertices)

    # Residual Form
    ###############

    # Measurement entities 
    # --------------------
    ds = fenics.Measure("ds", domain=mesh, subdomain_data=boundaries) 

    # a_new, v_new
    # ------------
    print("\nexpressing a_new and v_new thanks to β-Newmark time integration approximation...")
    a_new = numerical_scheme_temporal.update_acceleration(u, u_old, v_old, a_old, beta, dt, ufl=True)
    v_new = numerical_scheme_temporal.update_velocity(a_new, u_old, v_old, a_old, gamma, dt, ufl=True)

    # prerequisites before computing Fg and mu (H, d2s and gm=f(d2s, H) required)
    # ----------------------------------------
    print("\ninitializing distances to surface...")
    vertex2dofs_S = mappings.vertex_to_dof_ScalarFunctionSpace(S)
    d2s = differential_layers.compute_distance_to_cortexsurface(vertex2dofs_S, d2s, mesh, bmesh_cortexsurface_bbtree) # init at t=0.0

    print("\ninitializing differential term function...")
    H.assign( fenics.project(h, S) )  # helpers.local_project(h, S, H)
    gm = differential_layers.compute_differential_term_DOF(S, d2s, H, gm) # init 'gm' at t=0.0
    """for dof in S.dofmap().dofs():
        d2s_dof = d2s.vector()[dof]
        gm.vector()[dof] = compute_differential_term(d2s_dof, H.vector()[dof]) """

    # brain regions growth mask
    # -------------------------  
    print("mark no-growth brain regions (e.g. 'longitudinal fissure' - 'ventricules' - 'mammilary bodies')...") # code source: T.Tallinen et al. 2016. See detail in https://github.com/rousseau/BrainGrowth/blob/master/geometry.py   
    for vertex, scalarDOF in enumerate(vertex2dofs_S):
                                            
        rqp = np.linalg.norm( np.array([    0.8*characteristics["maxx"]/0.8251 * mesh.coordinates()[vertex, 0],
                                            0.63 * mesh.coordinates()[vertex, 1] + 0.075, 
                                            0.8*characteristics["maxz"]/0.637 * mesh.coordinates()[vertex, 2] - 0.25   ])) 
        
        if rqp < 0.6:
            gr.vector()[scalarDOF] = max(1.0 - 10.0*(0.6 - rqp), 0.0)
        else:
            gr.vector()[scalarDOF] = 1.0
    
    # Fg
    # --
    print("\ninitializing growth coefficients: dgTAN & dgRAD...")
    
    dg_TAN.assign( fenics.project(grTAN * alphaTAN * gm * gr * ((1 - alphaF) * dt), S) )
    dg_RAD.assign( fenics.project(grRAD * alphaRAD *  ((1 - alphaF) * dt), S) ) # = 0.0

    """
    helpers.local_project(grTAN * alphaTAN * gm * ((1 - alphaF) * dt), S, dg_TAN)
    helpers.local_project(grRAD * alphaRAD *  ((1 - alphaF) * dt), S, dg_RAD)"""

    """for dof in S.dofmap().dofs(): # init at t=0.0
        dg_TAN.vector()[dof] = grTAN * alphaTAN * gm.vector()[dof] * ((1 - alphaF) * dt)
        dg_RAD.vector()[dof] = grRAD * alphaRAD *  ((1 - alphaF) * dt)"""
    
    # demo: growth tensor variation needs to be considered at dt'= t_{n+1-αF} - t_{n} and not dt.
                # So: t_{n+1-αF} - t_{n} = (1-αF) * t_{n+1} + αF * t_{n} - t_{n}
                #                        = (1-αF) * t_{n+1} - (1-αF) * t_{n} 
                #                        = (1-αF) * (t_{n+1} - t_{n}) = (1-αF) * dt

    print("\ninitializing normals to boundary...")
    BoundaryMesh_Nt = fenics.Function(V, name="BoundaryMesh_Nt")
    BoundaryMesh_Nt.assign( growth.compute_topboundary_normals(mesh, ds, V) )

    print("\ninitializing projected normals of nodes of the whole mesh...")
    Mesh_Nt = fenics.Function(V, name="Mesh_Nt")
    Mesh_Nt.assign( growth.compute_mesh_projected_normals(V, mesh.coordinates(), bmesh.coordinates(), vertexB_2_dofsV_mapping, vertex2dofs_V, BoundaryMesh_Nt) )


    print("\ninitializing growth tensor...")
    #helpers.local_project( compute_growth_tensor(Mesh_Nt, dg_TAN, dg_RAD), Vtensor, Fg) # init at t=0.0 (local_project equivalent to .assign())"""
    Fg = growth.compute_growth_tensor(Mesh_Nt, dg_TAN, dg_RAD, gdim)
    projection.local_project(Fg, Vtensor, Fg_T) # projection of Fg onto Vtensor Function Space

    # mucontact
    # --
    print("\ninitializing local stiffness...")
    mu_ = differential_layers.compute_stiffness(gm, muCore, muCortex)
    mu_ = fenics.project(mu_, S)
    mu.assign( mu_ )

    # external forces
    # ---------------
    body_forces_V = fenics.Constant([0.0, 0.0, 0.0])
    tract_V = fenics.Constant([0.0, 0.0, 0.0])

    # penalty forces
    # --------------
    print("\ninitializing contact forces...")
    fcontact_global_V = fenics.Function(V, name="GlobalContactForces") 

    """
    # NNLt parameters
    n_surface_nodes = bmesh.num_vertices()
    coordinates_old = np.zeros((n_surface_nodes,3), dtype = np.float64)  #Stores positions when proximity list is updated
    NNLt = [[] for _ in range (n_surface_nodes)] #Triangle-proximity lists for surface nodes

    Ft, NNLt = contact.contact_mechanics_Tallinen(NNLt, 
                                               V, vertexB_2_dofsV_mapping, 
                                               S, vertexBoundaryMesh_2_dofScalarFunctionSpaceWholeMesh_mapping,
                                               K, average_mesh_spacing, gr, 
                                               mesh, bmesh, coordinates_old)
    """
    
    Ft = contact.contact_mechanics_V2(mesh, bmesh, V, 
                                   vertex2dofs_V, vertexB_2_dofsV_mapping, vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping_t0, 
                                   K, average_mesh_spacing)

    fcontact_global_V.assign(Ft)
    #fcontact_global_V.assign( contact.contact_CORRECTED(mesh, V, K, vertex2dofs_V, vertexB_2_dofsV_mapping, vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping_t0, average_mesh_spacing) )
    
    # Residual Form (ufl)
    # -------------
    print("\ngenerating Residual Form to minimize...")
    res = ( numerical_scheme_spatial.m(rho, numerical_scheme_temporal.avg(a_old, a_new, alphaM), v_test) 
          + numerical_scheme_spatial.c(damping_coef, numerical_scheme_temporal.avg(v_old, v_new, alphaF), v_test) 
          + numerical_scheme_spatial.k(numerical_scheme_temporal.avg(u_old, u, alphaF), v_test, Fg_T, mu, K, gdim) # Fg
          - numerical_scheme_spatial.contact(fcontact_global_V, ds, v_test) 
          - numerical_scheme_spatial.Wext(body_forces_V, v_test) 
          - numerical_scheme_spatial.traction(tract_V, ds, v_test) )

    # Non Linear Problem to solve
    #############################
    print("\nexpressing the non linear variational problem to solve...")
    jacobian = fenics.derivative(res, u) # we want to find u that minimize F(u) = 0 (F(u): total potential energy of the system), where F is the residual form of the PDE => dF(u)/du 

    bcs = []
    nonlinearvariationalproblem = fenics.NonlinearVariationalProblem(res, u, bcs, jacobian)   

    # Solver
    ########

    # Parameters
    # ----------
    nonlinearvariationalsolver = fenics.NonlinearVariationalSolver(nonlinearvariationalproblem) 
    # info(nonlinearvariationalsolver.parameters, True) # display the list of available parameters and default values
    # https://home.simula.no/~hpl/homepage/fenics-tutorial/release-1.0-nonabla/fenics_tutorial_1.0.pdf
    #https://link.springer.com/content/pdf/10.1007/978-3-319-52462-7_5.pdf
    # https://fenicsproject.org/qa/5894/nonlinearvariationalsolver-tolerance-what-solver-options/ (first used)

    # SOLVER PARAMETERS FOR NON-LINEAR PROBLEM 
    nonlinearvariationalsolver.parameters["nonlinear_solver"] = args.parameters["linearization_method"] # newton
    #nonlinearvariationalsolver.parameters['newton_solver']['convergence_criterion'] = "incremental" 
    nonlinearvariationalsolver.parameters['newton_solver']['absolute_tolerance'] = 1E-8 # 1E-8
    nonlinearvariationalsolver.parameters['newton_solver']['relative_tolerance'] = 1E-7 # 1E-7
    nonlinearvariationalsolver.parameters['newton_solver']['maximum_iterations'] = 25 # 50 (25)
    nonlinearvariationalsolver.parameters['newton_solver']['relaxation_parameter'] = 1.0 # means "full" Newton-Raphson iteration expression: u_k+1 = u_k - res(u_k)/res'(u_k) => u_k+1 = u_k - res(u_k)/jacobian(u_k)

    # CHOOSE AND PARAMETRIZE THE LINEAR SOLVER IN EACH NEWTON ITERATION (LINEARIZED PROBLEM) 
    nonlinearvariationalsolver.parameters['newton_solver']['linear_solver'] = args.parameters["linear_solver"] # linearized problem: AU=B --> Choose between direct method U=A⁻¹B O(N³) (e.g. 'mumps') or iterative/Krylov subspaces method U=A⁻¹B~(b + Ab + A²b + ...) O(num_iter * N²) (e.g. 'gmres' for non-symmetric problem , 'cg') to compute A⁻¹. 
    nonlinearvariationalsolver.parameters['newton_solver']['preconditioner'] = args.parameters["preconditioner"]

    nonlinearvariationalsolver.parameters['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-9 #1E-9
    nonlinearvariationalsolver.parameters['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-7 #1E-7
    nonlinearvariationalsolver.parameters['newton_solver']['krylov_solver']['maximum_iterations'] = 1000 # number of iterations with Krylov subspace method
    
    # Reusing previous unknown u_n as the initial guess to solve the next iteration n+1 
    """ nonlinearvariationalsolver.parameters['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True """ # https://link.springer.com/content/pdf/10.1007/978-3-319-52462-7_5.pdf --> "Using a nonzero initial guess can be particularly important for timedependent problems or when solving a linear system as part of a nonlinear iteration, since then the previous solution vector U will often be a good initial guess for the solution in the next time step or iteration."
    # parameters['krylov_solver']['monitor_convergence'] = True # https://fenicsproject.org/qa/1124/is-there-a-way-to-set-the-inital-guess-in-the-krylov-solver/
    
    # Resolution
    # ----------
    times = np.linspace(T0, Tmax, Nsteps+1) 

    #result_file = fenics.XDMFFile(args.output + "displacement.xdmf")
    result_file = fenics.XDMFFile(args.output + "growth_simulation.xdmf")
    result_file.parameters["flush_output"] = True
    result_file.parameters["functions_share_mesh"] = True
    result_file.parameters["rewrite_function_mesh"] = True

    # Saving functions fixed with time
    result_file.write(d2s, T0)
    result_file.write(H, T0)
    result_file.write(gm, T0)
    result_file.write(gr, T0)

    result_file.write(BoundaryMesh_Nt, T0)
    result_file.write(Mesh_Nt, T0)

    result_file.write(dg_TAN, T0)
    result_file.write(dg_RAD, T0)
    result_file.write(Fg_T, T0)

    result_file.write(mu, T0)

    start_time = time.time ()
    for i, dt in enumerate( tqdm( np.diff(times), desc='brain is growing...', leave=True) ):

    # collisions (fcontact_global_V) have to be detected at each step

        fenics.set_log_level(fenics.LogLevel.ERROR) # in order not to print solver info logs 

        t = times[i+1]
        t_i_plus_1_minus_alphaF = t - float(alphaF * dt) # Solver should be applied at t_{n+1-alpha_f}=t_{n+1}-alpha_f*dt (alpha-generalized method)

        
        # Detect and compute penalty forces to include collision correction into the residual form
        ##########################################################################################
        min_mesh_spacing, average_mesh_spacing, max_mesh_spacing = preprocessing.compute_mesh_spacing(mesh) # Update average mesh spacing
        n_surface_nodes = bmesh.num_vertices()  # NNLt parameters
        bmesh = fenics.BoundaryMesh(mesh, "exterior") # cortex envelop #print("\nupdating boundarymesh...")
        
        """
        Ft, NNLt = contact.contact_mechanics_Tallinen(NNLt, 
                                                   V, vertexB_2_dofsV_mapping, 
                                                   S, vertexBoundaryMesh_2_dofScalarFunctionSpaceWholeMesh_mapping,
                                                   K, average_mesh_spacing, gr, 
                                                   mesh, bmesh, coordinates_old) """
        
        Ft = contact.contact_mechanics_V2(mesh, bmesh, V, 
                                       vertex2dofs_V, vertexB_2_dofsV_mapping, vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping_t0, 
                                       K, average_mesh_spacing)
        
        fcontact_global_V.assign(Ft)
        
        """
        print("\nmin contact force: {}".format( np.min(fcontact_global_V.vector()[:]) ) )
        print("max contact force: {}".format( np.max(fcontact_global_V.vector()[:]) ) )
        
        if len( np.where(fcontact_global_V.vector()[:] != 0.)[0] ) != 0 : # if interlobes corrective forces detected 
                    
                    #print( "Contact forces will be applied on {} volume DOFs to avoid interlobes collison...".format(len(np.where(fcontact_global_V.vector()[:] != 0.)[0]) ) )

                    if args.visualization == True:

                        vedo.dolfin.plot(mesh,
                                         fcontact_global_V, 
                                         mode='mesh arrows', 
                                         wireframe=True, 
                                         style='meshlab', # meshlab; bw; matplotlib
                                         scale=0.1).clear() 

                        time.sleep(4.)
        """

        # Solve
        #######       
        nonlinearvariationalsolver.solve() 

        # Export displacement
        #####################
        result_file.write(u, t)
        result_file.write(fcontact_global_V, t)

        """
        if visualization == True:
            vedo.dolfin.plot(u, 
                             mode='displace', 
                             text="Step {} / {}:\nMesh at time {} / tmax={}\nDisplacement to be applied".format(step_to_be_applied, self.number_steps, t_i_plus_1, self.tmax), 
                             style='paraview', 
                             axes=4, 
                             camera=dict(pos=(0., 0., -6.)), 
                             interactive=False).clear() 
            
            time.sleep(4.) 
        """

        # Move mesh and boundary
        ########################
        # Mesh
        #print("\nmoving mesh...")
        fenics.ALE.move(mesh, u)

        # Export mesh characteristics 
        #############################
        """
        if i%10 == 0:
            fenics.File(args.output + "mesh_{}.xml".format(t)) << mesh
            convert_meshformats_to_vtk.xml_to_vtk(args.output + "mesh_{}.xml".format(t), args.output + "mesh_{}.vtk".format(t))
            export_simulation_outputmesh_data.export_resultmesh_data(args.output,
                                                                    args.output + "mesh_{}.vtk".format(t),
                                                                    t,
                                                                    i+1,
                                                                    total_computational_cost,
                                                                    "mesh_{}.txt".format(t))
        """
        

        # Update old fields with new quantities
        #######################################
        #print("\nupdating fields...")
        u_old, v_old, a_old = numerical_scheme_temporal.update_fields(u, u_old, v_old, a_old, beta, gamma, dt)
        
        # Computational time
        ####################
        total_computational_time = time.time () - start_time
        exportTXTfile_name = "simulation_duration_details.txt"
        export_simulation_end_time_and_iterations.export_maximum_time_iterations(args.output + "analytics/",
                                                                                 exportTXTfile_name,
                                                                                 T0, Tmax, Nsteps,
                                                                                 t,
                                                                                 i+1,
                                                                                 total_computational_time)


    # Export final mesh characteristics 
    ###################################
    total_computational_time = time.time () - start_time
    fenics.File(args.output + "mesh_Tmax.xml") << mesh
    convert_meshformats_to_vtk.xml_to_vtk(args.output + "mesh_Tmax.xml", args.output + "mesh_Tmax.vtk")
    export_simulation_outputmesh_data.export_resultmesh_data(args.output + "analytics/",
                                                             args.output + "mesh_Tmax.vtk",
                                                             t,
                                                             i+1,
                                                             total_computational_time,
                                                             "mesh_Tmax.txt")






