import sys, os
os.environ['OMP_NUM_THREADS'] = '4'  # Set the number of OpenMP CPUs to use (the MUMPS linear solver, which is the FEniCS element based on PETSc using parallelization, is based on OpenMP)
# os.environ['OMP_NUM_THREADS'] required to be imported prior to fenics

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

sys.path.append(sys.path[0]) # BrainGrowth3D
sys.path.append(os.path.dirname(sys.path[0])) # braingrowthFEniCS

from FEM_biomechanical_model import preprocessing, numerical_scheme_spatial, mappings, differential_layers, growth, projection
from utils.export_functions import export_simulation_end_time_and_iterations, export_XML_PVD_XDMF
from utils.converters import convert_meshformats


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='braingrowthFEniCS: FEniCS-based quasistatic sphere growth model.' \
    'The computational model implements a purely solid continuum mechanics conservation law.')

    parser.add_argument('-i', '--input', help='Input sphere mesh path (.xml, .xdmf)', type=str, required=False, 
                        default='./data/sphere/sphere_257Ktets_20Kfaces_refined5.xdmf') # sphere mesh in meters 
   
    parser.add_argument('-p', '--parameters', help='Simulation input parameters', type=json.loads, required=False, 
                        default={"H0": 2e-3, # [m] 
                                 "muCortex": 4500, "muCore": 300, # [Pa] 
                                 "nu": 0.45, # [-]
                                 "alphaTAN": 4e-7, "alphaRAD": 0.0, # [s⁻¹]
                                 "grTAN": 1.0, "grRAD": 1.0, # [-]  
                                 "T0_in_GW": 21.0, "Tmax_in_GW": 36.0, "dt_in_seconds": 86400, # 0.5GW (1GW=168h=604800s)
                                 "linearization_method":"newton", 
                                 "newton_absolute_tolerance":1E-9, "newton_relative_tolerance":1E-6, "max_iter": 15, 
                                 "linear_solver":"mumps"}) 
    
                                # dt_in_seconds
                                # -------------
                                # 1 GW = 604800s 
                                # 0.1 GW = 60480 s 
                                # 1500 s ~ 0.0025 GW 
                                # 3600 s (1h) ~ 0.006 GW
                                # 7200 s (2h) ~ 0.012 GW --> alphaTAN = 7.0e-6
                                # 43200 s (1/2 day) ~ 0.07 GW --> alphaTAN = 1.16e-6
                                # 86400 s (1 day) ~ 0.14 GW
    
    parser.add_argument('-o', '--output', help='Output folder path', type=str, required=False, 
                        default='./results/sphere_growth/')
               
    #parser.add_argument('-v', '--visualization', help='Visualization during simulation', type=bool, required=False, default=False)
    parser.add_argument('-v', '--visualization', help='Visualization during simulation', action='store_true')
    
    args = parser.parse_args() 

    #####################################################
    ###################### Parameters ###################
    #####################################################
    
    # Geometry 
    ##########
    
    # Cortex thickness
    # ----------------
    H0 = args.parameters["H0"]
    cortical_thickness = fenics.Expression('H0 + 0.01*t', H0=H0, t=0.0, degree=0) # eventually modifiy the time function of cortical thickness
    gdim=3
    
    # Mesh
    # ----
    # mesh & boundary mesh
    print("\nimporting mesh...")
    
    inputmesh_path = args.input
    inputmesh_format = inputmesh_path.split('.')[-1]

    if inputmesh_format == "xml":
        mesh0 = fenics.Mesh(inputmesh_path)

    elif inputmesh_format == "xdmf":
        mesh0 = fenics.Mesh()
        with fenics.XDMFFile(inputmesh_path) as infile:
            infile.read(mesh0)
    
    # input mesh was generated with Gmsh --> in meters. 
    # It is required to resize the mesh towards 21GW brain real size range. i.e. brain radius at 21GW ~ 30 mm = 0.03 m => multiply all coords by 0.03
    mesh0.coordinates()[:] = mesh0.coordinates()[:] * 0.03

    bmesh0 = fenics.BoundaryMesh(mesh0, "exterior") # bmesh at t=0.0 (cortex envelop)

    if args.visualization == True:
        fenics.plot(mesh0) 
        plt.title("input mesh")
        plt.show() 

    # mesh characteristics
    characteristics0 = preprocessing.compute_geometrical_characteristics(mesh0, bmesh0)
    center_of_gravity0 = preprocessing.compute_center_of_gravity(characteristics0) 
    min_mesh_spacing0, average_mesh_spacing0, max_mesh_spacing0 = preprocessing.compute_mesh_spacing(mesh0)
    print('input mesh characteristics: {}'.format(characteristics0))
    print('input mesh COG = [xG0:{}, yG0:{}, zG0:{}]'.format(center_of_gravity0[0], center_of_gravity0[1], center_of_gravity0[2]))
    print("input mesh min mesh spacing: {:.3f} mm".format(min_mesh_spacing0))
    print("input mesh mean mesh spacing: {:.3f} mm".format(average_mesh_spacing0))
    print("input mesh max mesh spacing: {:.3f} mm".format(max_mesh_spacing0))


    ###
    # empty sphere mesh with an inner VZ sphere
    sphere_radius_meters = 1.0 * 0.03
    radius_VZ = sphere_radius_meters / 5 
    #dilate_coef_x, dilate_coef_y, dilate_coef_z = sphere_radius_meters, sphere_radius_meters, sphere_radius_meters # --> sphere

    # Mark cells
    cell_markers = fenics.MeshFunction("size_t", mesh0, mesh0.topology().dim())
    cell_markers.set_all(0)

    ####

    class OuterSphere(fenics.SubDomain):
        def __init__(self, inner_radius, COG):
            fenics.SubDomain.__init__(self)
            self.inner_radius = inner_radius
            self.COG = COG

        def inside(self, x, on_boundary):
            return fenics.sqrt( (x[0]-self.COG[0])**2 + (x[1] - self.COG[1])**2 + (x[2]-self.COG[2])**2 ) >= self.inner_radius

    # Mark the domain regions outside from the inner radius
    outer_sphere = OuterSphere(radius_VZ, center_of_gravity0)
    outer_sphere.mark(cell_markers, 2)

    # Generate the new emptied sphere mesh 
    mesh = fenics.SubMesh(mesh0, cell_markers, 2) # cells belonging to the VZ (marked 2) are removed from the sphere mesh.
	
    with fenics.XDMFFile(MPI.COMM_WORLD, os.path.join(args.output, "mesh_T0.xdmf")) as xdmf:
        xdmf.write(mesh)

    ###

    bmesh = fenics.BoundaryMesh(mesh, "exterior")

    # Elastic parameters
    ####################
    muCortex = fenics.Constant(args.parameters["muCortex"])
    muCore = fenics.Constant(args.parameters["muCore"])
    
    nu = fenics.Constant(args.parameters["nu"])
    
    KCortex = fenics.Constant( 2*muCortex.values()[0] * (1 + nu.values()[0]) / (3*(1 - 2*nu.values()[0])) ) # formula for 3D geometries. source: https://en.wikipedia.org/wiki/Lam%C3%A9_parameters
    KCore = fenics.Constant( 2*muCore.values()[0] * (1 + nu.values()[0]) / (3*(1 - 2*nu.values()[0])) ) # formula for 3D geometries.
    
    # Growth parameters
    ###################
    alphaTAN = fenics.Constant(args.parameters["alphaTAN"])
    grTAN = fenics.Constant(args.parameters["grTAN"])

    alphaRAD = fenics.Constant(args.parameters["alphaRAD"])
    grRAD = fenics.Constant(args.parameters["grRAD"])
    
    # Time stepping
    ###############
    T0_in_GW = args.parameters["T0_in_GW"]
    T0_in_seconds = T0_in_GW * 604800 # 1GW=168h=604800s

    Tmax_in_GW = args.parameters["Tmax_in_GW"]
    Tmax_in_seconds = Tmax_in_GW * 604800

    dt_in_seconds = args.parameters["dt_in_seconds"] # --> dt = 1500/2000seconds to 1hour max (for the result not to variate too much)
    dt_in_GW = dt_in_seconds / 604800
    print('\ntime step: {} seconds <-> {:.3f} GW'.format(dt_in_seconds, dt_in_GW)) # in original BrainGrowth: dt = 0,000022361 ~ 2.10⁻⁵

    Nsteps = (Tmax_in_seconds - T0_in_seconds) / dt_in_seconds 
    print('\nNsteps: ~{}'.format( int(Nsteps) ))

    # Form compiler options
    #######################
    # See https://fenicsproject.org/olddocs/dolfin/1.6.0/python/demo/documented/hyperelasticity/python/documentation.html
    
    fenics.parameters["form_compiler"]["optimize"] = True
    fenics.parameters["form_compiler"]["cpp_optimize"] = True # The form compiler to use C++ compiler optimizations when compiling the generated code.
    fenics.parameters["form_compiler"]["quadrature_degree"] = 3 # --> number of node on each edge where to solve the unknown
    #fenics.parameters["allow_extrapolation"] = False 
    fenics.parameters["std_out_all_processes"] = False #  turn off solver logs

    # Output file
    #############
    outputpath = os.path.join(args.output, "growth_simulation.xdmf")
    FEniCS_FEM_Functions_file = fenics.XDMFFile(outputpath)
    FEniCS_FEM_Functions_file.parameters["flush_output"] = True
    FEniCS_FEM_Functions_file.parameters["functions_share_mesh"] = True
    FEniCS_FEM_Functions_file.parameters["rewrite_function_mesh"] = True

    numerical_metrics_path = args.output + "numerical_metrics/"

    try:
        os.makedirs(numerical_metrics_path)
    except OSError as e:
        print(f"Erreur: {e}")
    
    residual_path = os.path.join(numerical_metrics_path, 'residuals.json')
    residual_metrics = {}

    comp_time_path = os.path.join(numerical_metrics_path, 'computational_times.json') 
    comp_time = {} 
    
    energy_internal_path = os.path.join(numerical_metrics_path, 'internal_energies.json') 
    energy_internal = {} 
    
    ##################################################
    ###################### Problem ###################
    ##################################################
    
    # Boundaries
    ############
    print("\ncomputing and marking boundaries...")
    bmesh_bbtree = fenics.BoundingBoxTree()
    bmesh_bbtree.build(bmesh) 

    # initialize boundaries

    #boundaries = fenics.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    volume_facet_marker = fenics.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)  
    volume_facet_marker.set_all(100)

    boundary_face_marker = fenics.MeshFunction('size_t', bmesh, bmesh.topology().dim(), 0)
    boundary_face_marker.set_all(100)
    
    # mark surfaces (cortex + ventricular zone)
    class CortexSurface(fenics.SubDomain): 

        def __init__(self, bmesh_bbtree):
            fenics.SubDomain.__init__(self)
            self.bmesh_bbtree = bmesh_bbtree

        def inside(self, x, on_boundary): 
            _, distance = self.bmesh_bbtree.compute_closest_entity(fenics.Point(*x)) # compute_closest_point() https://fenicsproject.org/olddocs/dolfin/1.5.0/python/programmers-reference/cpp/mesh/GenericBoundingBoxTree.html
            return fenics.near(distance, fenics.DOLFIN_EPS) # returns Points

    cortexsurface = CortexSurface(bmesh_bbtree)
    cortexsurface.mark(volume_facet_marker, 101, check_midpoint=False) # https://fenicsproject.discourse.group/t/how-to-compute-boundary-mesh-and-submesh-from-an-halfdisk-mesh/9812/2
    cortexsurface.mark(boundary_face_marker, 101, check_midpoint=False)
    
    # define part of the boundary to fix (Dirichlet BCs) --> interior of the ellipsoid for the folding not to be impacted (Ventricular Zone empited and fixed)
    min_coords_ellipsoid = np.min(mesh.coordinates()[:, 0])
    max_coords_ellipsoid = np.max(mesh.coordinates()[:, 0])

    # Defining InnerBoundary_VZ (99) enables to compute d2s; gm and Fg properly, only considering external cortex surface (101).
    # ----------------------------------------------------------------------------------------------
    class InnerBoundary_VZ(fenics.SubDomain): # enable to set to 101 only the exterior boundary. Otherwise, d2s; gm and Fg will consider this inner VZ boundary.
        def __init__(self, inner_radius, COG):
            fenics.SubDomain.__init__(self)
            self.inner_radius = inner_radius
            self.COG = COG

        def inside(self, x, on_boundary):
            tol = 1E-14
            return fenics.sqrt( (x[0]-self.COG[0])**2 + (x[1]-self.COG[1])**2 + (x[2]-self.COG[2])**2 ) <= 2*self.inner_radius + tol # and on_boundary
    
    inner_boundary_VZ = InnerBoundary_VZ(radius_VZ, center_of_gravity0)
    inner_boundary_VZ.mark(boundary_face_marker, 99, check_midpoint=False) # and on_boundary needs to be removed for 102 to be marked on bmesh marker!

    # Defining VZ inner boundary using volume facets, which is used by FEniCS to build Dirichlet BCs
    # ----------------------------------------------------------------------------------------------
    class DirichletBCs_VZ(fenics.SubDomain):
        def __init__(self, inner_radius, COG):
            fenics.SubDomain.__init__(self)
            self.inner_radius = inner_radius
            self.COG = COG

        def inside(self, x, on_boundary):
            tol = 1E-14
            return fenics.sqrt( (x[0]-self.COG[0])**2 + (x[1]-self.COG[1])**2 + (x[2]-self.COG[2])**2 ) <= 2*self.inner_radius + tol and on_boundary
    
    dirichletBCs_VZ = DirichletBCs_VZ(radius_VZ, center_of_gravity0)
    dirichletBCs_VZ.mark(volume_facet_marker, 102) # and on_boundary must be added, otherwise, tetraedrons will be fixed and not only faces at the surface of inner VZ boundary

    # export marked boundaries (both are required for the simulation)
    export_XML_PVD_XDMF.export_PVDfile(args.output, 'volume_facet_marker_T0', volume_facet_marker) # "volume_facet_marker" is used to define the surface zone for FEM integration and in particular the surface where the Dirichlet boundary conditions apply.
    export_XML_PVD_XDMF.export_PVDfile(args.output, 'boundary_face_marker_T0', boundary_face_marker) # "boundary_face_marker" is used to build bmesh_cortex and the associated bmesh_cortexsurface_bbtree. And then, bmesh_cortexsurface_bbtree is used to define the FEM functions d2s and gm, required by the simulation (so to define the zone where there will be growth).  
    
    # build cortical surface mesh after having removed the boundary 101 associated with the Dirichlet BCs in the Ventricular Zone. 
    bmesh_cortex = fenics.SubMesh(bmesh, boundary_face_marker, 101) # part of the boundary mesh standing for the cortical surface --> at this stage, 101 corresponds only to all the cortex boundary (ventricular zone boundary, when labelled 102 was removed)

    #print("\ncomputing and marking boundaries...")
    bmesh_cortexsurface_bbtree = fenics.BoundingBoxTree()
    bmesh_cortexsurface_bbtree.build(bmesh_cortex)          

    # Subdomains
    ############
    subdomains = fenics.MeshFunction("size_t", mesh, mesh.topology().dim()) 
    subdomains.set_all(0)
    
    # FEM Function Spaces 
    #####################
    print("\ncreating Lagrange FEM function spaces and functions...")

    # Scalar Function Spaces
    S = fenics.FunctionSpace(mesh, "CG", 1) 
    S_cortexsurface = fenics.FunctionSpace(bmesh_cortex, "CG", 1)

    # Vector Function Spaces
    V = fenics.VectorFunctionSpace(mesh, "CG", 1)
    V_cortexsurface = fenics.VectorFunctionSpace(bmesh_cortex, "CG", 1) 

    # Tensor Function Spaces
    #Vtensor = fenics.TensorFunctionSpace(mesh, "DG", 0)
    Vtensor = fenics.TensorFunctionSpace(mesh,'CG', 1, shape=(3,3)) # https://fenicsproject.discourse.group/t/outer-product-evaluation/2159; https://fenicsproject.discourse.group/t/how-to-choose-projection-space-for-stress-tensor-post-processing/5568/4

    Vtensor_order3 = fenics.VectorFunctionSpace(mesh, "DG", 0, dim=27) # to project grad(Fe) 

    # FEM Functions
    ###############

    # Scalar functions of V
    H = fenics.Function(S, name="H") 
    #fenics.File(os.path.join(args.output, "FEM_functions/H.xml")) << H
    d2s = fenics.Function(S, name="d2s")
    #grGrowthZones = fenics.Function(S, name="grGrowthZones")
    #gr = fenics.Function(S, name="gr") 
    gm = fenics.Function(S, name="gm") 
    mu = fenics.Function(S, name="mu") 
    K = fenics.Function(S, name="K") 

    dg_TAN = fenics.Function(S, name="dgTAN")
    dg_RAD = fenics.Function(S, name="dgRAD") 

    J = fenics.Function(S, name="Jacobian")

    # Vector functions of V
    u = fenics.Function(V, name="Displacement") # Trial function. Current (unknown) displacement
    du = fenics.TrialFunction(V)
    v_test = fenics.TestFunction(V) # Test function

    BoundaryMesh_Nt = fenics.Function(V, name="BoundaryMesh_Nt")
    Mesh_Nt = fenics.Function(V, name="Mesh_Nt")

    # Vector functions of Vtensor
    Fg_T = fenics.Function(Vtensor, name="Fg")
    PK1tot_T = fenics.Function(Vtensor, name="PK1tot") 
    Ee_T = fenics.Function(Vtensor, name="E_strain") 

    F_T = fenics.Function(Vtensor, name="F") 
    Fe_T = fenics.Function(Vtensor, name="Fe") 
    gradFe_T = fenics.Function(Vtensor_order3, name="gradFe") 

    # Mappings
    ##########
    # From vertex to DOF in the whole mesh --> used to compute Mesh_Nt
    # ----------------------------------------------------------------
    vertex2dof_S = mappings.vertex_to_dof_ScalarFunctionSpace(S)
    vertex2dofs_V = mappings.vertex_to_dofs_VectorFunctionSpace(V, gdim)
    vertex2dofs_B = mappings.vertex_to_dofs_VectorFunctionSpace(V_cortexsurface, gdim)

    # From the surface mesh (cortex envelop) to the whole mesh (B_2_V_dofmap; vertexB_2_dofsV_mapping --> used to compute Mesh_Nt)
    # --------------------------------------------------------
    B101_2_V_dofmap, vertexB101_2_dofsV_mapping = mappings.surface_to_mesh_V(gdim, V, V_cortexsurface, vertex2dofs_B)
    Sboundary101_2_S_dofmap, vertexBoundary101Mesh_2_dofScalarFunctionSpaceWholeMesh_mapping = mappings.surface_to_mesh_S(S, S_cortexsurface)
    
    # Residual Form
    ###############

    # Measurement entities 
    # --------------------
    ds = fenics.Measure("ds", domain=mesh, subdomain_data=volume_facet_marker) 

    # prerequisites before computing Fg and mu (H, d2s and gm=f(d2s, H) required)
    # ----------------------------------------
    print("\ninitializing distances to surface...")
    vertex2dof_S = mappings.vertex_to_dof_ScalarFunctionSpace(S)
    d2s_ = differential_layers.compute_distance_to_cortexsurface(vertex2dof_S, d2s, mesh, bmesh_cortexsurface_bbtree) # init at t=0.0
    projection.local_project(d2s_, S, d2s)

    print("\ninitializing differential term function...")
    projection.local_project(cortical_thickness, S, H) # H.assign( fenics.project(h, S) )  
    gm_ = differential_layers.compute_differential_term_DOF(S, d2s, H, gm) # init 'gm' at t=0.0
    projection.local_project(gm_, S, gm)
    """for dof in S.dofmap().dofs():
        d2s_dof = d2s.vector()[dof]
        gm.vector()[dof] = compute_differential_term(d2s_dof, H.vector()[dof]) """

    # Fg
    # --
    print("\ninitializing growth coefficients: dgTAN & dgRAD...")
    #projection.local_project(grTAN * gm * grGrowthZones * alphaTAN * dt_in_seconds, S, dg_TAN) 
    projection.local_project(grTAN * gm * alphaTAN * dt_in_seconds, S, dg_TAN) 
    projection.local_project(grRAD * alphaRAD * dt_in_seconds, S, dg_RAD) 

    print("\ninitializing normals to boundary...")
    boundary_normals = growth.compute_topboundary_normals(mesh, ds(101), V)
    projection.local_project(boundary_normals, V, BoundaryMesh_Nt)

    print("\ninitializing projected normals of nodes of the whole mesh...")
    mesh_normals = growth.compute_mesh_projected_normals(V, mesh.coordinates(), bmesh_cortex.coordinates(), vertexB101_2_dofsV_mapping, vertex2dofs_V, BoundaryMesh_Nt) 
    projection.local_project(mesh_normals, V, Mesh_Nt)

    print("\ninitializing growth tensor...")
    #helpers.local_project( compute_growth_tensor(Mesh_Nt, dg_TAN, dg_RAD), Vtensor, Fg) # init at t=0.0 (local_project equivalent to .assign())"""
    Fg = growth.compute_growth_tensor(Mesh_Nt, dg_TAN, dg_RAD, gdim)
    projection.local_project(Fg, Vtensor, Fg_T) # projection of Fg onto Vtensor Function Space

    # mucontact
    # --
    print("\ninitializing local stiffness...")
    mu_ = differential_layers.compute_shear_and_bulk_stiffnesses(gm, muCore, muCortex)
    projection.local_project(mu_, S, mu)
    #mu_ = fenics.project(mu_, S)
    #mu.assign( mu_ )
    
    K_ = differential_layers.compute_shear_and_bulk_stiffnesses(gm, KCore, KCortex)
    projection.local_project(K_, S, K)

    # external forces
    # ---------------
    """
    body_forces_V = fenics.Constant([0.0, 0.0, 0.0])
    tract_V = fenics.Constant([0.0, 0.0, 0.0])
    """
    
    # Residual Form (ufl)
    # -------------
    print("\ngenerating Residual Form to minimize...")
    
    Id = fenics.Identity(gdim)

    # F: deformation gradient
    F = fenics.variable( Id + fenics.grad(u) ) # F = I₃ + ∇u / [F]: 3x3 matrix
    F_inv = fenics.variable( fenics.inv(F) )

    # Fe: elastic part of the deformation gradient
    Fg_inv = fenics.variable( fenics.inv(Fg) )
    Fe = fenics.variable( F * Fg_inv )# F_t * (F_g)⁻¹

    # Cauchy-Green tensors (elastic part of the deformation only)
    Ce = fenics.variable( Fe.T * Fe )
    Be = fenics.variable( Fe * Fe.T )

    Ee = 0.5*(Ce - Id)  # Green-Lagrange tensor --> visualize strain

    # Invariants 
    Je = fenics.variable( fenics.det(Fe) ) 
    Tre = fenics.variable( fenics.tr(Be) )
    
    # Neo-Hookean strain energy density function
    # ------------------------------------------
    #We = 0.5 * mu * (Tre * pow(Je, -2/3) - 3) + 0.5 * K * (Je - fenics.ln(Je) - 1) # T.Tallinen
    We = 0.5 * mu * (Tre * pow(Je, -2/3) - 3) + 0.5 * K * (Je - 1) * (Je - 1) # X.Wang https://github.com/rousseau/BrainGrowth/

    # Cauchy stress (elastic part)
    # -------------
    """
    Te = fenics.variable( mu * (Be - Tre/3 * Id) * pow(Je, -5/3) \
                        + K * (Je - 1) * Id ) # X.Wang https://github.com/rousseau/BrainGrowth/
    """
    
    # 1st Piola-Kirchhoff stress (elastic part)
    # --------------------------
    """ PK1e = fenics.variable( Je * Te * fenics.inv(Fe.T) ) """# X.Wang https://github.com/rousseau/BrainGrowth/
    #PK1e = fenics.diff(We, Fe)

    # 1st Piola-Kirchhoff stress (total)
    # --------------------------
    # Need to translate Pelastic into Ptotal to appsly the equilibrium balance momentum law.
    PK1tot = fenics.diff(We, F)
      
    res = fenics.inner(PK1tot, fenics.grad(v_test)) * fenics.Measure("dx")

    # Non Linear Problem to solve
    #############################
    print("\nexpressing the non linear variational problem to solve...")
    jacobian = fenics.derivative(res, u, du) # we want to find u that minimize F(u) = 0 (F(u): total potential energy of the system), where F is the residual form of the PDE => dF(u)/du 

    bc_Dirichlet = fenics.DirichletBC(V, fenics.Constant((0., 0., 0.)), volume_facet_marker, 102) # no displacement in x,y,z --> fixed zone to avoid additional solution including Rotations & Translations
    bcs = [bc_Dirichlet]
    nonlinearvariationalproblem = fenics.NonlinearVariationalProblem(res, u, bcs, jacobian)   

    ####################################################
    ###################### Solver ######################
    ####################################################

    # Parameters
    #############
    nonlinearvariationalsolver = fenics.NonlinearVariationalSolver(nonlinearvariationalproblem) 
    # info(nonlinearvariationalsolver.parameters, True) # display the list of available parameters and default values
    # https://home.simula.no/~hpl/homepage/fenics-tutorial/release-1.0-nonabla/fenics_tutorial_1.0.pdf
    # https://link.springer.com/content/pdf/10.1007/978-3-319-52462-7_5.pdf
    # https://fenicsproject.org/qa/5894/nonlinearvariationalsolver-tolerance-what-solver-options/ 

    # SOLVER PARAMETERS FOR NON-LINEAR PROBLEM 
    nonlinearvariationalsolver.parameters["nonlinear_solver"] = args.parameters["linearization_method"] 
    #nonlinearvariationalsolver.parameters['newton_solver']['convergence_criterion'] = "incremental" 
    nonlinearvariationalsolver.parameters['newton_solver']['absolute_tolerance'] = args.parameters["newton_absolute_tolerance"] 
    nonlinearvariationalsolver.parameters['newton_solver']['relative_tolerance'] = args.parameters["newton_relative_tolerance"] 
    nonlinearvariationalsolver.parameters['newton_solver']['maximum_iterations'] = args.parameters["max_iter"] 
    nonlinearvariationalsolver.parameters['newton_solver']['relaxation_parameter'] = 0.8 # means "full" Newton-Raphson iteration expression: u_k+1 = u_k - res(u_k)/res'(u_k) => u_k+1 = u_k - res(u_k)/jacobian(u_k)

    nonlinearvariationalsolver.parameters['newton_solver']['error_on_nonconvergence'] = True
    nonlinearvariationalsolver.parameters['newton_solver']['report'] = True
    
    # CHOOSE AND PARAMETRIZE THE LINEAR SOLVER IN EACH NEWTON ITERATION (LINEARIZED PROBLEM) 
    nonlinearvariationalsolver.parameters['newton_solver']['linear_solver'] = args.parameters["linear_solver"] 
    
    """
    nonlinearvariationalsolver.parameters['newton_solver']['linear_solver'] = args.parameters["linear_solver"] # linearized problem: AU=B --> Choose between direct method U=A⁻¹B O(N³) (e.g. 'mumps') or iterative/Krylov subspaces method U=A⁻¹B~(b + Ab + A²b + ...) O(num_iter * N²) (e.g. 'gmres' for non-symmetric problem , 'cg') to compute A⁻¹. 
    nonlinearvariationalsolver.parameters['newton_solver']['preconditioner'] = args.parameters["preconditioner"]
    
    nonlinearvariationalsolver.parameters['newton_solver']['krylov_solver']['absolute_tolerance'] = args.parameters["krylov_absolute_tolerance"] #1E-4 #1E-9
    nonlinearvariationalsolver.parameters['newton_solver']['krylov_solver']['relative_tolerance'] = args.parameters["krylov_relative_tolerance"] #1E-5 #1E-7
    nonlinearvariationalsolver.parameters['newton_solver']['krylov_solver']['maximum_iterations'] = 1000 # number of iterations with Krylov subspace method
    """
    
    # Reusing previous unknown u_n as the initial guess to solve the next iteration n+1 
    ####################################################################################
    nonlinearvariationalsolver.parameters['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True # Enables to use a non-null initial guess for the Krylov solver (MUMPS) within a Newton-Raphson iteration. https://link.springer.com/content/pdf/10.1007/978-3-319-52462-7_5.pdf --> "Using a nonzero initial guess can be particularly important for timedependent problems or when solving a linear system as part of a nonlinear iteration, since then the previous solution vector U will often be a good initial guess for the solution in the next time step or iteration."
    # parameters['krylov_solver']['monitor_convergence'] = True # https://fenicsproject.org/qa/1124/is-there-a-way-to-set-the-inital-guess-in-the-krylov-solver/
    
    ########################################################
    ###################### Simulation ######################
    ########################################################
    
    times = np.linspace(T0_in_seconds, Tmax_in_seconds, int(Nsteps+1))  # in seconds!
    
    # Export FEM function at T0_in_GW
    FEniCS_FEM_Functions_file.write(d2s, T0_in_GW)
    FEniCS_FEM_Functions_file.write(H, T0_in_GW)
    FEniCS_FEM_Functions_file.write(gm, T0_in_GW)
    #FEniCS_FEM_Functions_file.write(grGrowthZones, T0_in_GW)

    FEniCS_FEM_Functions_file.write(BoundaryMesh_Nt, T0_in_GW)
    FEniCS_FEM_Functions_file.write(Mesh_Nt, T0_in_GW)

    FEniCS_FEM_Functions_file.write(dg_TAN, T0_in_GW)
    FEniCS_FEM_Functions_file.write(dg_RAD, T0_in_GW)
    FEniCS_FEM_Functions_file.write(Fg_T, T0_in_GW)

    FEniCS_FEM_Functions_file.write(mu, T0_in_GW)
    FEniCS_FEM_Functions_file.write(K, T0_in_GW)

    initial_time = time.time() 
    for i, dt in enumerate( tqdm( np.diff(times), desc='brain is growing...', leave=True) ): # dt = dt_in_seconds

        #fenics.set_log_level(fenics.LogLevel.ERROR) # in order not to print solver info logs about Newton solver tolerances
        fenics.set_log_level(fenics.LogLevel.INFO) # in order to print solver info logs about Newton solver tolerances

        t = times[i+1] # in seconds
        t_in_GW = t / 604800 
                
        # Update pre-required entities
        # ----------------------------
        # H
        """h.t = t
        #H.assign( fenics.project(h, S) )# Expression -> scalar Function of the mesh
        projection.local_project(cortical_thickness, S, H) # H.assign( fenics.project(cortical_thickness, S) )  
        """
    
        # d2s
        #d2s.assign( differential_layers.compute_distance_to_cortexsurface(vertex2dof_S, d2s, mesh, bmesh_cortexsurface_bbtree) )
        d2s_ = differential_layers.compute_distance_to_cortexsurface(vertex2dof_S, d2s, mesh, bmesh_cortexsurface_bbtree) 
        #d2s_ = differential_layers.compute_distance_to_cortexsurface_2(vertex2dof_S, d2s, mesh, bmesh_cortexsurface_cKDtree) 
        projection.local_project(d2s_, S, d2s)
        
        # gm
        #gm = differential_layers.compute_differential_term_DOF(S, d2s, H, gm)
        gm_ = differential_layers.compute_differential_term_DOF(S, d2s, H, gm) # init 'gm' at t=0.0
        projection.local_project(gm_, S, gm)

        # Update differential material stiffness mu 
        # -----------------------------------------
        # mu have to be updated at each timestep (material properties evolution with deformation) (So do previously H, d2s, gm)
        mu_ = differential_layers.compute_shear_and_bulk_stiffnesses(gm, muCore, muCortex)
        projection.local_project(mu_, S, mu)
        
        K_ = differential_layers.compute_shear_and_bulk_stiffnesses(gm, KCore, KCortex)
        projection.local_project(K_, S, K)

        # Update growth tensor coefficients
        # ---------------------------------
        #projection.local_project(grTAN * gm * grGrowthZones * alphaTAN * dt, S, dg_TAN)
        projection.local_project(grTAN * gm * alphaTAN * dt, S, dg_TAN)
        projection.local_project(grRAD * alphaRAD * dt, S, dg_RAD) 

        # Update growth tensor orientation (adaptative)
        # ---------------------------------------------
        boundary_normals = growth.compute_topboundary_normals(mesh, ds(101), V) 
        projection.local_project(boundary_normals, V, BoundaryMesh_Nt)
        
        mesh_normals = growth.compute_mesh_projected_normals(V, 
                                                             mesh.coordinates(), 
                                                             bmesh_cortex.coordinates(), 
                                                             vertexB101_2_dofsV_mapping, 
                                                             vertex2dofs_V,
                                                             BoundaryMesh_Nt) 
        projection.local_project(mesh_normals, V, Mesh_Nt)

        # Final growth tensor
        # -------------------
        Fg = growth.compute_growth_tensor(Mesh_Nt, dg_TAN, dg_RAD, gdim)
        projection.local_project(Fg, Vtensor, Fg_T) # projection of Fg onto Vtensor Function Space    

        # Solve
        #######       
        nonlinearvariationalsolver.solve() 

        # Export displacement & other FEM functions
        ###########################################
        # model parameters
        FEniCS_FEM_Functions_file.write(d2s, t_in_GW)
        FEniCS_FEM_Functions_file.write(H, t_in_GW)
        FEniCS_FEM_Functions_file.write(gm, t_in_GW)
        FEniCS_FEM_Functions_file.write(mu, t_in_GW)
        FEniCS_FEM_Functions_file.write(K, t_in_GW)

        projection.local_project(fenics.det(F), S, J) # local volume change (for incompressible material, should be close to 1)
        FEniCS_FEM_Functions_file.write(J, t_in_GW)
        
        # growth tensor components
        FEniCS_FEM_Functions_file.write(BoundaryMesh_Nt, t_in_GW) 
        FEniCS_FEM_Functions_file.write(Mesh_Nt, t_in_GW) 
        
        FEniCS_FEM_Functions_file.write(dg_TAN, t_in_GW)
        FEniCS_FEM_Functions_file.write(dg_RAD, t_in_GW)
        FEniCS_FEM_Functions_file.write(Fg_T, t_in_GW)
        
        # Analysis of the stress and strain
        FEniCS_FEM_Functions_file.write(u, t_in_GW) # displacement field
        
        projection.local_project(Ee, Vtensor, Ee_T)
        FEniCS_FEM_Functions_file.write(Ee_T, t_in_GW) # Green-Lagrange strain field
        
        projection.local_project(PK1tot, Vtensor, PK1tot_T) # Piola-Kirchhoff stress
        FEniCS_FEM_Functions_file.write(PK1tot_T, t_in_GW)

        projection.local_project(F, Vtensor, F_T) # Deformation gradient
        FEniCS_FEM_Functions_file.write(F_T, t_in_GW)

        projection.local_project(Fe, Vtensor, Fe_T) # Elastic deformation gradient
        FEniCS_FEM_Functions_file.write(Fe_T, t_in_GW)

        gradFe_vec = fenics.as_vector([fenics.grad(Fe)[i, j, k] for i in range(3) for j in range(3) for k in range(3)])
        projection.local_project(gradFe_vec, Vtensor_order3, gradFe_T) # ∇Fe
        FEniCS_FEM_Functions_file.write(gradFe_T, t_in_GW)

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
        
        # Assess the numerical validity of the computational model
        ##########################################################
        # residual at each time step
        # --------------------------
        residual_vector = fenics.assemble(res) # Vector (V Space)
        #residual_function.abs() # to get positive values of residual
        residual_DOFs_array = residual_vector.get_local() # get numpy array of residuals (3 dofs per node)

        residual_metrics[t_in_GW] = {}

        # L2 norm of the residual matrix
        residual_metrics[t_in_GW]["residual_vector_L2norm"] = residual_vector.norm('l2') # amplitude moyenne du residu sur tout le domaine

        # infinite norm of the residual matrix
        residual_metrics[t_in_GW]["residual_vector_normInfinite"] = np.max(np.abs(residual_DOFs_array)) # erreur max sur tout le domaine

        # mean value of the residual matrix
        residual_metrics[t_in_GW]["residual_vector_mean"] = np.mean(residual_DOFs_array) # (indication of global error)

        # relative error
        #solution_norm = fenics.assemble(u*u*fenics.dx)**0.5 # L2 Norm of the solution TODO: not working
        #relative_error = residual_metrics["residual_vector_norm"] / solution_norm

        # other metrics
        residual_metrics[t_in_GW]["residual_vector_min"] = np.min(residual_DOFs_array)
        residual_metrics[t_in_GW]["residual_vector_max"] = np.max(residual_DOFs_array)

        with open(residual_path, 'w') as res_json_file:  
            json.dump(residual_metrics, res_json_file, indent=0)

        # cumul computational times at each step
        # --------------------------------------
        time_at_this_step = time.time()
        cumulative_time = time_at_this_step - initial_time
        
        comp_time[t_in_GW] = {}
        
        comp_time[t_in_GW]["iteration"] = i
        comp_time[t_in_GW]["total_iterations"] = Nsteps
        comp_time[t_in_GW]["cumulative_computational_time"] = cumulative_time

        with open(comp_time_path, 'w') as comp_time_json_file:  
            json.dump(comp_time, comp_time_json_file, indent=0)

        # internal energy
        #################
        # if energy int --> 0, displacement field satisfies the ODE
        # Otherwise, there could be an issue in the numerical solving or in the definition of the BCs
         
        energy_internal[t_in_GW] = {}
        
        internal_deformation_energy_at_this_step = fenics.assemble(We * fenics.dx)
        energy_internal[t_in_GW]["elastic_energy"] = internal_deformation_energy_at_this_step
        
        with open(energy_internal_path, 'w') as energy_internal_json_file:  
            json.dump(energy_internal, energy_internal_json_file, indent=0)
            

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
        #mesh.smooth()
        #mesh.smooth_boundary(10, True) # --> to smooth the contact boundary after deformation and avoid irregular mesh surfaces at the interhemisphere https://fenicsproject.discourse.group/t/3d-mesh-generated-from-imported-surface-volume-exhibits-irregularities/3293/6 

        # Boundary (if d2s and Mesh_Nt need to be udpated: "solver_Fgt_norm"; "solver_Fgt")
        #print("\nupdating boundarymesh...")
        bmesh = fenics.BoundaryMesh(mesh, "exterior") # cortex envelop

        bmesh_cortex = fenics.SubMesh(bmesh, boundary_face_marker, 101)
        
        bmesh_cortexsurface_bbtree = fenics.BoundingBoxTree()
        bmesh_cortexsurface_bbtree.build(bmesh_cortex) 
        
        #bmesh_cortexsurface_cKDtree = cKDTree(bmesh_cortex.coordinates())
        
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
        
        # Computational time
        ####################
        """
        total_computational_time = time.time () - start_time
        exportTXTfile_name = "simulation_duration_details.txt"
        export_simulation_end_time_and_iterations.export_maximum_time_iterations(os.path.join(args.output, "analytics"),
                                                                                 exportTXTfile_name,
                                                                                 T0_in_GW, Tmax_in_GW, Nsteps,
                                                                                 t_in_GW,
                                                                                 i+1,
                                                                                 total_computational_time)
        """

    # Export final mesh characteristics 
    ###################################
    """
    total_computational_time = time.time () - start_time
    
    with fenics.XDMFFile(MPI.COMM_WORLD, os.path.join(args.output, "mesh_Tmax.xdmf")) as xdmf:
        xdmf.write(mesh)
        
    convert_meshformats.xdmf_to_vtk(os.path.join(args.output, "mesh_Tmax.xdmf"), os.path.join(args.output, "mesh_Tmax.vtk"))
    export_simulation_outputmesh_data.export_resultmesh_data(os.path.join(args.output, "analytics/"),
                                                             os.path.join(args.output, "mesh_Tmax.vtk"),
                                                             t_in_GW,
                                                             i+1,
                                                             total_computational_time,
                                                             "mesh_Tmax.txt")
    """






