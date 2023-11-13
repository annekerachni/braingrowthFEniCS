# elastodynamic brain growth model (dynamic structural mechanics), growth tensor norm updated at each timestep

import fenics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import vedo.dolfin
import time
import argparse
import json

import preprocessing, numerical_scheme_temporal, numerical_scheme_spatial, mappings, contact, outputs, differential_layers, growth


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='braingrowthFEniCS: brain growth elastodynamics 3D model')

    parser.add_argument('-i', '--input', help='Input mesh path (xml)', type=str, required=False, 
                        default='./data/sphere_algoDelaunay1_tets005_refinedcoef5.xml') 
    
    parser.add_argument('-n', '--normalization', help='Is normalization of the input mesh required? (required by braingrowthFEniCS)', type=bool, required=False, default=True)
    
    parser.add_argument('-p', '--parameters', help='Simulation input parameters', type=json.loads, required=False, 
                        default={"H0": 0.04, 
                                 "K": 100.0, 
                                 "muCortex": 20.0, "muCore": 1.0, 
                                 "rho": 0.01, 
                                 "damping_coef": 0.5,
                                 "alphaTAN": 4.0, "alphaRAD": 0.0, "grTAN": 1.0, "grRAD": 1.0, 
                                 "alphaM": 0.2, "alphaF": 0.4, 
                                 "T": 1.0, "Nsteps": 100,
                                 "linearization_method":"newton", "linear_solver":"gmres", "preconditioner":"sor"})
    
    parser.add_argument('-o', '--output', help='Output folder path', type=str, required=False, 
                        default='./results/')
    
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
    outputs.export_PVDfile(args.output, 'boundaries_t0', boundaries)

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
    T       = args.parameters["T"]
    Nsteps  = args.parameters["Nsteps"]
    dt = fenics.Constant(T/Nsteps)
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
    du = fenics.TrialFunction(V)
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
    vertex2dofs_V = mappings.vertex_to_dof_vectors(V, gdim)
    vertex2dofs_B = mappings.vertex_to_dof_vectors(V_cortexsurface, gdim)

    # From the surface mesh (cortex envelop) to the whole mesh (B_2_V_dofmap; vertexB_2_dofsV_mapping --> used to compute Mesh_Nt)
    # --------------------------------------------------------
    B_2_V_dofmap, vertexB_2_dofsV_mapping = mappings.surface_to_mesh(gdim, V, V_cortexsurface, vertex2dofs_B)


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
    vertex2dofs_S = mappings.vertex_to_dof_scalars(S)
    d2s = differential_layers.compute_distance_to_cortexsurface(vertex2dofs_S, d2s, mesh, bmesh_cortexsurface_bbtree) # init at t=0.0

    print("\ninitializing differential term function...")
    H.assign( fenics.project(h, S) )  # helpers.local_project(h, S, H)
    gm = differential_layers.compute_differential_term_DOF(S, d2s, H, gm) # init 'gm' at t=0.0
    """for dof in S.dofmap().dofs():
        d2s_dof = d2s.vector()[dof]
        gm.vector()[dof] = compute_differential_term(d2s_dof, H.vector()[dof]) """

    # Fg
    # --

    print("\ninitializing growth coefficients: dgTAN & dgRAD...")
    
    dg_TAN.assign( fenics.project(grTAN * alphaTAN * gm * ((1 - alphaF) * dt), S) )
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
    outputs.local_project(Fg, Vtensor, Fg_T) # projection of Fg onto Vtensor Function Space

    # mu
    # --
    print("\ninitializing local stiffness...")
    mu_ = differential_layers.compute_stiffness(gm, muCore, muCortex)
    mu_ = fenics.project(mu_, S)
    mu.assign( mu_)

    # external forces
    # ---------------
    body_forces_V = fenics.Constant([0.0, 0.0, 0.0])
    tract_V = fenics.Constant([0.0, 0.0, 0.0])

    # penalty forces
    # --------------
    print("\ninitializing contact forces...")
    fcontact_global_V = fenics.Function(V, name="GlobalContactForces") 
    fcontact_global_V.assign( contact.global_collisions_forces(mesh, V, K, vertex2dofs_V, vertexB_2_dofsV_mapping) ) # init at t=0.0

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
    jacobian = fenics.derivative(res, u, du) # we want to find u that minimize F(u) = 0 (F(u): total potential energy of the system), where F is the residual form of the PDE => dF(u)/du 

    bcs = []
    nonlinearvariationalproblem = fenics.NonlinearVariationalProblem(res, u, bcs, jacobian)   

    # Solver
    ########

    # Parameters
    # ----------
    nonlinearvariationalsolver = fenics.NonlinearVariationalSolver(nonlinearvariationalproblem) 
    # info(nonlinearvariationalsolver.parameters, True) # display the list of available parameters and default values

    # SOLVER PARAMETERS FOR NON-LINEAR PROBLEM 
    nonlinearvariationalsolver.parameters["nonlinear_solver"] = args.parameters["linearization_method"] 
    nonlinearvariationalsolver.parameters['newton_solver']['absolute_tolerance'] = 1E-8 
    nonlinearvariationalsolver.parameters['newton_solver']['relative_tolerance'] = 1E-7
    nonlinearvariationalsolver.parameters['newton_solver']['maximum_iterations'] = 25  
    #nonlinearvariationalsolver.parameters['newton_solver']['convergence_criterion'] = "residual" 
    nonlinearvariationalsolver.parameters['newton_solver']['relaxation_parameter'] = 1.0 

    # DIRECT OR ITERATIVE SOLVER PARAMETERS FOR LINEARIZED PROBLEM 
    nonlinearvariationalsolver.parameters['newton_solver']['linear_solver'] = args.parameters["linear_solver"] # linearized problem: AU=B --> Choose between direct method U=A⁻¹B O(N³) (e.g. 'mumps') or iterative/Krylov subspaces method U=A⁻¹B~(b + Ab + A²b + ...) O(num_iter * N²) (e.g. 'gmres' for non-symmetric problem , 'cg') to compute A⁻¹. 

    nonlinearvariationalsolver.parameters['newton_solver']['preconditioner'] = args.parameters["preconditioner"]
    nonlinearvariationalsolver.parameters['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-9 
    nonlinearvariationalsolver.parameters['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-7 
    nonlinearvariationalsolver.parameters['newton_solver']['krylov_solver']['maximum_iterations'] = 1000 # number of iterations with Krylov subspace method


    # Resolution
    # ----------
    times = np.linspace(0, T, Nsteps+1) 

    xdmf_file = fenics.XDMFFile(args.output + "displacement.xdmf")
    xdmf_file.parameters["flush_output"] = True
    xdmf_file.parameters["functions_share_mesh"] = True
    xdmf_file.parameters["rewrite_function_mesh"] = True

    # Saving functions fixed with time (only Fg norm will evolve but not orientation, fixed at time=0.0)
    xdmf_file.write(BoundaryMesh_Nt, 0.0) 
    xdmf_file.write(Mesh_Nt, 0.0)

    for i, dt in enumerate( tqdm( np.diff(times), desc='brain is growing...', leave=True) ):

    # collisions (fcontact_global_V) have to be detected at each step

        fenics.set_log_level(fenics.LogLevel.ERROR) # in order not to print solver info logs 

        t = times[i+1]
        t_i_plus_1_minus_alphaF = t - float(alphaF * dt) # Solver should be applied at t_{n+1-alpha_f}=t_{n+1}-alpha_f*dt (alpha-generalized method)

        # Update local  material properties 'mu' and growth tensor 'Fg'
        ##############################################################      
        
        # Update pre-required entities
        # ----------------------------
        # H
        #print("\nupdating cortical thickness...")
        h.t = t_i_plus_1_minus_alphaF  
        H.assign( fenics.project(h, S) )# Expression -> scalar Function of the mesh

        # d2s
        #print("\nupdating distances to surface...")
        d2s.assign( differential_layers.compute_distance_to_cortexsurface(vertex2dofs_S, d2s, mesh, bmesh_cortexsurface_bbtree) )
        
        # gm
        #print("\nupdating differential term function...")
        gm = differential_layers.compute_differential_term_DOF(S, d2s, H, gm)

        # Update differential material stiffness mu 
        # -----------------------------------------
        # mu have to be updated at each timestep (material properties evolution with deformation) (So do previously H, d2s, gm)
        #print("\nupdating local stiffness...")
        mu_ = differential_layers.compute_stiffness(gm, muCore, muCortex)
        mu_ = fenics.project(mu_, S)
        mu.assign( mu_ )

        # Update growth tensor coefficients
        # ---------------------------------
        #print("\nupdating growth coefficients: dgTAN & dgRAD...")
        dg_TAN.assign( fenics.project(grTAN * alphaTAN * gm * ((1 - alphaF) * dt), S) )
        dg_RAD.assign( fenics.project(grRAD * alphaRAD *  ((1 - alphaF) * dt), S) )
        """for dof in S.dofmap().dofs():
            dg_TAN.vector()[dof] = grTAN * alphaTAN * gm.vector()[dof] * float((1 - alphaF) * dt)
            dg_RAD.vector()[dof] = grRAD * alphaRAD *  float((1 - alphaF) * dt)"""

        #print("\nupdating growth tensor...")
        Fg = growth.compute_growth_tensor(Mesh_Nt, dg_TAN, dg_RAD, gdim)
        outputs.local_project(Fg, Vtensor, Fg_T) # projection of Fg onto Vtensor Function Space

        # Detect and compute penalty forces to include collision correction into the residual form
        ##########################################################################################
        #print("\nupdating contact forces...")
        fcontact_global_V.assign( contact.global_collisions_forces(mesh, V, K, vertex2dofs_V, vertexB_2_dofsV_mapping) )

        # Solve
        #######

        nonlinearvariationalsolver.solve() 

        # Export displacement 
        #####################
        xdmf_file.write(u, t)

        xdmf_file.write(d2s, t)
        xdmf_file.write(H, t)
        xdmf_file.write(gm, t)

        #xdmf_file.write(BoundaryMesh_Nt, t) # move only for solvertype == "solver_Fgt"
        #xdmf_file.write(Mesh_Nt, t) # move only for solvertype == "solver_Fgt"

        xdmf_file.write(dg_TAN, t)
        xdmf_file.write(dg_RAD, t)
        xdmf_file.write(Fg_T, t)

        xdmf_file.write(mu, t)


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

        # Update old fields with new quantities
        #######################################
        #print("\nupdating fields...")
        u_old, v_old, a_old = numerical_scheme_temporal.update_fields(u, u_old, v_old, a_old, beta, gamma, dt)








