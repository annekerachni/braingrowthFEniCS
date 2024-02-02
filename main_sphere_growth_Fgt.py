# elastodynamic brain growth model (dynamic structural mechanics)

import fenics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import vedo.dolfin
import time
import argparse
import json
import time
from mpi4py import MPI
import sys, os

sys.path.append(sys.path[0]) # Add sys.path[0]: ~/braingrowthFEniCS/ from any local path

from FEM_biomechanical_model import preprocessing, numerical_scheme_temporal, numerical_scheme_spatial, mappings, contact, differential_layers, growth, projection
from utils.export_functions import export_simulation_outputmesh_data, export_simulation_end_time_and_iterations, export_XML_PVD_XDMF
from utils.converters import convert_meshformats


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='braingrowthFEniCS: brain growth elastodynamics 3D model')

    parser.add_argument('-i', '--input', help='Input mesh path (xml)', type=str, required=False, 
                        default='./data/sphere.xdmf')                        

    parser.add_argument('-n', '--normalization', help='Is normalization of the input mesh required? (required by braingrowthFEniCS)', type=bool, required=False, default=False)
    
    parser.add_argument('-p', '--parameters', help='Simulation input parameters', type=json.loads, required=False, 
                        default={"H0": 0.05,
                                 "K": 100.0, 
                                 "muCortex":50.0, "muCore": 1.0, 
                                 "rho": 1.0, 
                                 "damping_coef": 0.5,
                                 "alphaTAN": 4.0, "alphaRAD": 0.0, "grTAN": 1.0, "grRAD": 1.0, 
                                 "alphaM": 0.2, "alphaF": 0.4, 
                                 "T0": 0.0, "Tmax": 1.0, "Nsteps": 100,
                                 "linearization_method":"newton", "linear_solver":"gmres", "preconditioner":"sor"})

    parser.add_argument('-o', '--output', help='Output folder path', type=str, required=True, 
                        default='results')
    
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

    # Output file
    #############
    outputpath = os.path.join(args.output, "growth_simulation.xdmf")
    FEniCS_FEM_Functions_file = fenics.XDMFFile(outputpath)
    FEniCS_FEM_Functions_file.parameters["flush_output"] = True
    FEniCS_FEM_Functions_file.parameters["functions_share_mesh"] = True
    FEniCS_FEM_Functions_file.parameters["rewrite_function_mesh"] = True

    # Input mesh
    ############

    # Mesh
    # ----
    # mesh & boundary mesh
    print("\nimporting mesh...")
    
    inputmesh_path = args.input
    inputmesh_format = inputmesh_path.split('.')[-1]

    if inputmesh_format == "xml":
        mesh = fenics.Mesh(inputmesh_path)

    elif inputmesh_format == "xdmf":
        mesh = fenics.Mesh()
        with fenics.XDMFFile(inputmesh_path) as infile:
            infile.read(mesh)

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

    """
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
    """

    # Export the characteristics of mesh_TO 
    # -------------------------------------
    # fenics.File(os.path.join(args.output, "mesh_T0.xml")) << mesh
    with fenics.XDMFFile(MPI.COMM_WORLD, os.path.join(args.output, "mesh_T0.xdmf")) as xdmf:
            xdmf.write(mesh)
            
    convert_meshformats.xdmf_to_vtk(os.path.join(args.output, "mesh_T0.xdmf"), os.path.join(args.output, "mesh_T0.vtk"))
    export_simulation_outputmesh_data.export_resultmesh_data(os.path.join(args.output, "analytics/"),
                                                             os.path.join(args.output, "mesh_T0.vtk"),
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
    export_XML_PVD_XDMF.export_PVDfile(args.output, 'boundaries_T0', boundaries)

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
    #gr = fenics.Function(S, name="gr") 
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
    projection.local_project(h, S, H) # H.assign( fenics.project(h, S) )  
    gm = differential_layers.compute_differential_term_DOF(S, d2s, H, gm) # init 'gm' at t=0.0
    """for dof in S.dofmap().dofs():
        d2s_dof = d2s.vector()[dof]
        gm.vector()[dof] = compute_differential_term(d2s_dof, H.vector()[dof]) """

    # Fg
    # --

    print("\ninitializing growth coefficients: dgTAN & dgRAD...")
    projection.local_project(grTAN * alphaTAN * gm * ((1 - alphaF) * dt), S, dg_TAN) # dg_TAN.assign( fenics.project(grTAN * alphaTAN * gm * ((1 - alphaF) * dt), S) )
    projection.local_project(grRAD * alphaRAD *  ((1 - alphaF) * dt), S, dg_RAD) # dg_RAD.assign( fenics.project(grRAD * alphaRAD *  ((1 - alphaF) * dt), S) ) # = 0.0

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
    projection.local_project(mu_, S, mu)
    #mu_ = fenics.project(mu_, S)
    #mu.assign( mu_ )

    # external forces
    # ---------------
    body_forces_V = fenics.Constant([0.0, 0.0, 0.0])
    tract_V = fenics.Constant([0.0, 0.0, 0.0])

    # penalty forces
    # --------------
    print("\ninitializing contact forces...")
    fcontact_global_V = fenics.Function(V, name="GlobalContactForces") 
    Ft = contact.correct_collisions(mesh, V, K, vertex2dofs_V, vertexB_2_dofsV_mapping)
    fcontact_global_V.assign( Ft ) # init at t=0.0

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
    nonlinearvariationalsolver.parameters['newton_solver']['absolute_tolerance'] = 1E-3 # 1E-8
    nonlinearvariationalsolver.parameters['newton_solver']['relative_tolerance'] = 1E-2 # 1E-7
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

    # Saving functions fixed with time
    FEniCS_FEM_Functions_file.write(d2s, T0)
    FEniCS_FEM_Functions_file.write(H, T0)
    FEniCS_FEM_Functions_file.write(gm, T0)

    FEniCS_FEM_Functions_file.write(BoundaryMesh_Nt, T0)
    FEniCS_FEM_Functions_file.write(Mesh_Nt, T0)

    FEniCS_FEM_Functions_file.write(dg_TAN, T0)
    FEniCS_FEM_Functions_file.write(dg_RAD, T0)
    FEniCS_FEM_Functions_file.write(Fg_T, T0)

    FEniCS_FEM_Functions_file.write(mu, T0)

    start_time = time.time ()
    for i, dt in enumerate( tqdm( np.diff(times), desc='brain is growing...', leave=True) ):

    # collisions (fcontact_global_V) have to be detected at each step

        fenics.set_log_level(fenics.LogLevel.ERROR) # in order not to print solver info logs 

        t = times[i+1]
        t_i_plus_1_minus_alphaF = t - float(alphaF * dt) # Solver should be applied at t_{n+1-alpha_f}=t_{n+1}-alpha_f*dt (alpha-generalized method)

        # Update pre-required entities
        # ----------------------------
        # H
        #print("\nupdating cortical thickness...")
        h.t = t_i_plus_1_minus_alphaF  
        #H.assign( fenics.project(h, S) )# Expression -> scalar Function of the mesh
        projection.local_project(h, S, H) # H.assign( fenics.project(h, S) )  
    
        # d2s
        #print("\nupdating distances to surface...")
        #d2s.assign( differential_layers.compute_distance_to_cortexsurface(vertex2dofs_S, d2s, mesh, bmesh_cortexsurface_bbtree) )
        d2s_ = differential_layers.compute_distance_to_cortexsurface(vertex2dofs_S, d2s, mesh, bmesh_cortexsurface_bbtree) 
        projection.local_project(d2s_, S, d2s)
        
        # gm
        #print("\nupdating differential term function...")
        #gm = differential_layers.compute_differential_term_DOF(S, d2s, H, gm)
        gm_ = differential_layers.compute_differential_term_DOF(S, d2s, H, gm) # init 'gm' at t=0.0
        projection.local_project(gm_, S, gm)

        # Update differential material stiffness mu 
        # -----------------------------------------
        # mu have to be updated at each timestep (material properties evolution with deformation) (So do previously H, d2s, gm)
        #print("\nupdating local stiffness...")       
        mu_ = differential_layers.compute_stiffness(gm, muCore, muCortex)
        projection.local_project(mu_, S, mu)

        # Update growth tensor coefficients
        # ---------------------------------
        #print("\nupdating growth coefficients: dgTAN & dgRAD...")
        projection.local_project(grTAN * gm * alphaTAN * ((1 - alphaF) * dt), S, dg_TAN)
        projection.local_project(grRAD * alphaRAD *  ((1 - alphaF) * dt), S, dg_RAD) 

        # Update growth tensor orientation (adaptative)
        # ---------------------------------------------
        #print("\nupdating normals to boundary and its projections to the whole mesh nodes...")
        boundary_normals = growth.compute_topboundary_normals(mesh, ds, V) 
        mesh_normals = growth.compute_mesh_projected_normals(V, mesh.coordinates(), bmesh.coordinates(), vertexB_2_dofsV_mapping, vertex2dofs_V, BoundaryMesh_Nt) 
        
        projection.local_project(boundary_normals, V, BoundaryMesh_Nt)
        projection.local_project(mesh_normals, V, Mesh_Nt)

        # Final growth tensor
        # -------------------
        #print("\nupdating growth tensor...")
        Fg = growth.compute_growth_tensor(Mesh_Nt, dg_TAN, dg_RAD, gdim)
        projection.local_project(Fg, Vtensor, Fg_T) # projection of Fg onto Vtensor Function Space     


        # Detect and compute penalty forces to include collision correction into the residual form
        ##########################################################################################
        #print("\nupdating contact forces...")
        Ft = contact.correct_collisions(mesh, V, K, vertex2dofs_V, vertexB_2_dofsV_mapping)
        fcontact_global_V.assign( Ft )

        # Solve
        #######       
        nonlinearvariationalsolver.solve() 

        # Export displacement & other FEM functions
        ###########################################
        FEniCS_FEM_Functions_file.write(u, t)
        FEniCS_FEM_Functions_file.write(fcontact_global_V, t)
        
        FEniCS_FEM_Functions_file.write(d2s, t)
        FEniCS_FEM_Functions_file.write(H, t)
        FEniCS_FEM_Functions_file.write(gm, t)

        FEniCS_FEM_Functions_file.write(BoundaryMesh_Nt, t) 
        FEniCS_FEM_Functions_file.write(Mesh_Nt, t) 

        FEniCS_FEM_Functions_file.write(dg_TAN, t)
        FEniCS_FEM_Functions_file.write(dg_RAD, t)
        FEniCS_FEM_Functions_file.write(Fg_T, t)

        FEniCS_FEM_Functions_file.write(mu, t)

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

        # Boundary (if d2s and Mesh_Nt need to be udpated: "solver_Fgt_norm"; "solver_Fgt")
        #print("\nupdating boundarymesh...")
        bmesh = fenics.BoundaryMesh(mesh, "exterior") # cortex envelop
        bmesh_cortexsurface_bbtree = fenics.BoundingBoxTree()
        bmesh_cortexsurface_bbtree.build(bmesh) 

        # Export mesh characteristics 
        #############################
        """
        if i%10 == 0:
            path_xdmf = os.path.join(args.output, "mesh_{}.xdmf".format(t)) 
            path_vtk = os.path.join(args.output, "mesh_{}.vtk".format(t)) 
            
            with fenics.XDMFFile(MPI.COMM_WORLD, path_xdmf) as xdmf:
                xdmf.write(mesh)
            
            convert_meshformats.xdmf_to_vtk(path_xdmf, path_vtk)
            export_simulation_outputmesh_data.export_resultmesh_data(args.output,
                                                                     path_vtk,
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
        export_simulation_end_time_and_iterations.export_maximum_time_iterations(os.path.join(args.output, "analytics/"),
                                                                                 exportTXTfile_name,
                                                                                 T0, Tmax, Nsteps,
                                                                                 t,
                                                                                 i+1,
                                                                                 total_computational_time)


    # Export final mesh characteristics 
    ###################################
    total_computational_time = time.time () - start_time
    
    with fenics.XDMFFile(MPI.COMM_WORLD, os.path.join(args.output, "mesh_Tmax.xdmf")) as xdmf:
        xdmf.write(mesh)
        
    convert_meshformats.xdmf_to_vtk(os.path.join(args.output, "mesh_Tmax.xdmf"), os.path.join(args.output, "mesh_Tmax.vtk"))
    export_simulation_outputmesh_data.export_resultmesh_data(os.path.join(args.output, "analytics/"),
                                                             os.path.join(args.output, "mesh_Tmax.vtk"),
                                                             t,
                                                             i+1,
                                                             total_computational_time,
                                                             "mesh_Tmax.txt")




