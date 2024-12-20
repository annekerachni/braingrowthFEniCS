# FEniCS-based quasistatic brain growth model (purely solid continuum mechanics conservation law + penalization of contact between the two hemispheres)

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

# Parameters (SI)
############

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='braingrowthFEniCS: brain growth elastodynamics 3D model')

    parser.add_argument('-i', '--input', help='Input mesh path (.xml, .xdmf); mesh unit: either in millimeters, either in meters; mesh orientation: sagittal X, front X+, axial Y, front Y+, coronal Z, top Z+', type=str, required=False, 
                        default="./data/dHCP_surface_vs_volume_RAW/Vf/dhcp_surface_t21_raw_130000faces_455983tets_reoriented_dHCPVolume.xdmf") 
                        # from dHCP surface gifti mesh: "./data/dHCP_surface_vs_volume_RAW/Vf/dhcp_surface_t21_raw_130000faces_455983tets_reoriented_dHCPVolume.xdmf"--> in millimeters
                        # from dHCP volume niftis: "./data/dHCP_surface_vs_volume_RAW/Vf/dhcp_volume_t21_raw_130000faces_480112tets.xdmf" -->  in millimeters

                        # from dHCP volume niftis: "./data/21_28_36GW/transformed_niftis_meshes/transformed_dhcp21GW_isotropic_smoothed_TaubinSmooth50_refinedWidthCoef10.xdmf" (719399 tets)
                        # from dHCP surface gifti mesh: "./data/dHCP_surface/fetal_week21_left_right_merged_V2.xdmf" (455351 tets)
    
    parser.add_argument('-c', '--convertmesh0frommillimetersintometers', help='Convert mesh from millimeters into meters', type=bool, required=False, default=True)
    
    parser.add_argument('-p', '--parameters', help='Simulation input parameters', type=json.loads, required=False, 
                        default={"H0": 1.8e-3, # [m]
                                 "muCortex": 300, "muCore": 100, # [Pa] 
                                 "nu": 0.45, 
                                 "alphaTAN": 2.0e-7, "alphaRAD": 0.0, "grTAN": 1.0, "grRAD": 1.0, # "alphaTAN": 2.0e-7, "alphaRAD": 0.5e-7, "grTAN": 1.0, "grRAD": 1.0, 
                                 "epsilon_n": 5e5, # penalty coefficient (contact mechanics)
                                 "T0_in_GW": 21.0, "Tmax_in_GW": 36.0, "dt_in_seconds": 43200, 
                                 "linearization_method":"newton", 
                                 "newton_absolute_tolerance":1E-9, "newton_relative_tolerance":1E-6, "max_iter": 10, 
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
                        default='./results/brain_growth_OneRigidPlane_penaltyMethodCorrected_BoundaryNt101_103_104_dirichletEllipsoidVZ/') 
                           
    #parser.add_argument('-v', '--visualization', help='Visualization during simulation', type=bool, required=False, default=False)
    parser.add_argument('-v', '--visualization', help='Visualization during simulation', action='store_true')
    
    args = parser.parse_args()
    
    ###################################################
    ###################### Geometry ###################
    ###################################################
    
    # Mesh
    # ----
    # mesh & boundary mesh
    print("\nimporting whole brain mesh...")
    
    inputmesh_path = args.input
    inputmesh_format = inputmesh_path.split('.')[-1]

    if inputmesh_format == "xml":
        mesh_WholeBrain = fenics.Mesh(inputmesh_path)

    elif inputmesh_format == "xdmf":
        mesh_WholeBrain = fenics.Mesh()
        with fenics.XDMFFile(inputmesh_path) as infile:
            infile.read(mesh_WholeBrain)

    # convert initial input whole brain mesh and compute its characteristics
    if args.convertmesh0frommillimetersintometers == True:
        mesh_WholeBrain = preprocessing.converting_mesh_from_millimeters_into_meters(mesh_WholeBrain)
    
    bmesh_WholeBrain = fenics.BoundaryMesh(mesh_WholeBrain, "exterior") 
    characteristics = preprocessing.compute_geometrical_characteristics(mesh_WholeBrain, bmesh_WholeBrain)
    center_of_gravity = preprocessing.compute_center_of_gravity(characteristics) 
    min_mesh_spacing, average_mesh_spacing, max_mesh_spacing = preprocessing.compute_mesh_spacing(mesh_WholeBrain)
    
    print('input mesh characteristics: {}'.format(characteristics))
    print('input mesh COG = [xG0:{}, yG0:{}, zG0:{}]'.format(center_of_gravity[0], center_of_gravity[1], center_of_gravity[2]))
    print("input mesh min mesh spacing: {:.3f} mm".format(min_mesh_spacing))
    print("input mesh mean mesh spacing: {:.3f} mm".format(average_mesh_spacing))
    print("input mesh max mesh spacing: {:.3f} mm".format(max_mesh_spacing)) 
    
    # Brain mesh properties to define Dirichlet inner ellipsoid zone
    brain_maxX, brain_minX = np.max(mesh_WholeBrain.coordinates()[:,0]), np.min(mesh_WholeBrain.coordinates()[:,0])
    brain_maxY, brain_minY = np.max(mesh_WholeBrain.coordinates()[:,1]), np.min(mesh_WholeBrain.coordinates()[:,1])
    brain_maxZ, brain_minZ =  np.max(mesh_WholeBrain.coordinates()[:,2]), np.min(mesh_WholeBrain.coordinates()[:,2]) 
    
    dX = brain_maxX - brain_minX
    dY = brain_maxY - brain_minY
    dZ = brain_maxZ - brain_minZ
    
    hmin_symbolic = fenics.CellDiameter(mesh_WholeBrain) # symbolic expression providing cell diameter [m] for each cell of the mesh 
    Z = fenics.FunctionSpace(mesh_WholeBrain, "DG", 0)
    hmin_symbolic_proj = fenics.project(hmin_symbolic, Z) 
    min_cell_size = np.nanmin(hmin_symbolic_proj.vector()[:])
    average_cell_size = np.nanmean(hmin_symbolic_proj.vector()[:])
    max_cell_size = np.nanmax(hmin_symbolic_proj.vector()[:]) 
    print("max cell size in brain mesh = {} m".format(max_cell_size))

    hmin = min_cell_size
    hmean = average_cell_size
    hmax = max_cell_size
        
    # Export the characteristics of mesh_TO 
    # -------------------------------------
    #fenics.File(args.output + "mesh_WholeBrain_T0.xml") << mesh_WholeBrain
    """
    with fenics.XDMFFile(MPI.COMM_WORLD, os.path.join(args.output, "mesh_WholeBrain_T0.xdmf")) as xdmf:
            xdmf.write(mesh_WholeBrain)
            
    convert_meshformats.xml_to_vtk(args.output + "mesh_WholeBrain_T0.xml", args.output + "mesh_WholeBrain_T0.vtk")
    export_simulation_outputmesh_data.export_resultmesh_data(args.output + "analytics/",
                                                             args.output + "mesh_WholeBrain_T0.vtk",
                                                             args.parameters["T0"],
                                                             0,
                                                             0.0,
                                                             "mesh_WholeBrain_T0.txt")
    """

    #####################################################
    ###################### Parameters ###################
    #####################################################
    
    # Cortex thickness
    ##################
    H0 = args.parameters["H0"]
    cortical_thickness = fenics.Expression('H0 + 0.01*t', H0=H0, t=0.0, degree=0)
    gdim=3
    
    # Elastic parameters
    ####################
    muCortex = fenics.Constant(args.parameters["muCortex"])
    muCore = fenics.Constant(args.parameters["muCore"])
    
    nu = fenics.Constant(args.parameters["nu"])
    
    KCortex = fenics.Constant( 2*muCortex.values()[0] * (1 + nu.values()[0]) / (3*(1 - 2*nu.values()[0])) ) # 3D
    KCore = fenics.Constant( 2*muCore.values()[0] * (1 + nu.values()[0]) / (3*(1 - 2*nu.values()[0])) ) # 3D
    
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
    
    # Subdomains
    ############
    build_mesh_cell_markers = fenics.MeshFunction("size_t", mesh_WholeBrain, mesh_WholeBrain.topology().dim()) 
    build_mesh_cell_markers.set_all(0)
    
    # Cut brain mesh with ellipsoid Ventricular Zone (to define Dirichlet BCs)
    ##########################################################################
    print("\ncutting the brain mesh with an ellipsoid shape standing for the Ventricular Zone (to be used to define the Dirichlet BCs inside of the brain)...")
    
    S_WholeBrain = fenics.FunctionSpace(mesh_WholeBrain, "CG", 1) 
    
    class EllispoidVentricularZone(fenics.UserExpression):
        def __init__(self, radius_VZ, COG_bb, dX, dY, dZ, dilate_coef_x, dilate_coef_y, dilate_coef_z): 
            fenics.UserExpression.__init__(self)
            self.radius_VZ = radius_VZ
            self.COG_bb = COG_bb
            self.dilate_coef_x = dilate_coef_x
            self.dilate_coef_y = dilate_coef_y
            self.dilate_coef_z = dilate_coef_z
            self.dX, self.dY, self.dZ = dX, dY, dZ 
            
        def eval(self, value, x):
            #value[0] = x[0]**2 + x[1]**2 + x[2]**2 - 1  
            #tol = 1E-14
            value[0] = ((x[0]- self.COG_bb[0])/self.dilate_coef_x)**2 + ((x[1] - self.COG_bb[1])/self.dilate_coef_y)**2 + ((x[2]-self.COG_bb[2])/self.dilate_coef_z)**2 - self.radius_VZ**2 #+ tol)

        def value_shape(self):
            return () # returns a scalar
    
    COG_bb = 0.5*(brain_minX + brain_maxX), 0.5*(brain_minY + brain_maxY), 0.5*(brain_minZ + brain_maxZ)
    dilate_coef_x, dilate_coef_y, dilate_coef_z = 0.5*dX, 0.5*dY, 0.5*dZ
    radius_VZ = 1.0/10

    ellipsoid_ventricular_zone = EllispoidVentricularZone(radius_VZ, COG_bb, dX, dY, dZ, dilate_coef_x, dilate_coef_y, dilate_coef_z)
    
    
    """ellipsoid_ventricular_zone_2 = fenics.Expression("((x[0]- COG_bb_X)/dilate_coef_x)**2 + ((x[1] - COG_bb_Y)/dilate_coef_y)**2 + ((x[2]-COG_bb_Z)/dilate_coef_z)**2 - (radius_VZ**2 + tol)",  
                                                    degree=2,
                                                    COG_bb_X = COG_bb[0], dilate_coef_x = dilate_coef_x,
                                                    COG_bb_Y = COG_bb[1], dilate_coef_y = dilate_coef_y,
                                                    COG_bb_Z = COG_bb[2], dilate_coef_z = dilate_coef_z,
                                                    radius_VZ=radius_VZ, tol=1E-14)"""

    # project VZ onto the V FEM space
    zone_to_cut_scalar_FEM_function = fenics.project(ellipsoid_ventricular_zone, S_WholeBrain)
    #zone_to_cut_FEM_function = projection.local_project(ellipsoid_ventricular_zone, V, None)
    
    # 
    """cell_markers = fenics.MeshFunction("size_t", mesh_WholeBrain, mesh_WholeBrain.topology().dim())
    cell_markers.set_all(1)"""

    for cell in fenics.cells(mesh_WholeBrain):
        point = cell.midpoint()
        if zone_to_cut_scalar_FEM_function(point) <= 0:  
            build_mesh_cell_markers[cell] = 10
    # to check labelling of the VZ zone: build_mesh_cell_markers.array(); np.where(build_mesh_cell_markers.array() == 10)
            
    # get submeshes
    ellipsoid_VZ_mesh = fenics.SubMesh(mesh_WholeBrain, build_mesh_cell_markers, 10)
    mesh = fenics.SubMesh(mesh_WholeBrain, build_mesh_cell_markers, 0)
    bmesh = fenics.BoundaryMesh(mesh, "exterior") 
    
    # export brain mesh with ellipsoid VZ inner boundary
    with fenics.XDMFFile("./data/dHCP_surface_vs_volume_RAW/Vf/dhcp_surface_t21_raw_130000faces_455983tets_reoriented_dHCPVolume_in_meters_with_VZboundary.xdmf") as outfile:
        outfile.write(mesh)
        
    # Define FEM Function Spaces associated to the new mesh (with cut VZ)
    #####################################################################
    print("\ncreating Lagrange FEM function spaces and functions associated to the new mesh (with cut VZ)...")

    # Scalar Function Spaces
    S = fenics.FunctionSpace(mesh, "CG", 1) 
    vertex2dofs_S = mappings.vertex_to_dof_ScalarFunctionSpace(S) # mapping: from vertex to DOF in the whole mesh

    # Vector Function Spaces
    V = fenics.VectorFunctionSpace(mesh, "CG", 1)
    vertex2dofs_V = mappings.vertex_to_dofs_VectorFunctionSpace(V, gdim) #  mapping: from vertex to DOFs in the whole mesh --> used to compute Mesh_Nt
    
    # Tensor Function Spaces
    #Vtensor = fenics.TensorFunctionSpace(mesh, "DG", 0)
    Vtensor = fenics.TensorFunctionSpace(mesh,'CG', 1, shape=(3,3)) # https://fenicsproject.discourse.group/t/outer-product-evaluation/2159; https://fenicsproject.discourse.group/t/how-to-choose-projection-space-for-stress-tensor-post-processing/5568/4
    
    # FEM Functions
    ###############
    # Scalar functions of V103
    H = fenics.Function(S, name="H") 
    d2s = fenics.Function(S, name="d2s")
    grGrowthZones = fenics.Function(S, name="grGrowthZones")
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

    nabla_Fe = fenics.Function(V, name="NablaFe") # to express local gradient of the deformation ~ i.e. curvature

    # Vector functions of Vtensor
    Fg_T = fenics.Function(Vtensor, name="Fg")    
    PK1tot_T = fenics.Function(Vtensor, name="PK1tot") 
    Ee_T = fenics.Function(Vtensor, name="E_strain") 
    
    # Boundaries
    ############
    print("\ncomputing and marking boundaries...")
    bmesh_bbtree = fenics.BoundingBoxTree()
    bmesh_bbtree.build(bmesh) 

    # initialize markers   
    volume_facet_marker = fenics.MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # size: facets --> mesh.num_facets() => number of faces in the volume. / Each facet is labelled; in case "on_boundary" is mentionned, only the edges of the face are labelled
    volume_facet_marker.set_all(100) # TODO: replace name by 'volume_marker' or 'cell_marker'
    
    boundary_face_marker = fenics.MeshFunction('size_t', bmesh, bmesh.topology().dim(), 0) # size: n_faces (at surface) --> bmesh.num_faces() => number of faces at the surface / Each surface edge is labelled
    boundary_face_marker.set_all(100) # TODO: replace name by 'boundary_marker' 
    
    # Label boundaries
    ##################
    # mark cortex surface (and at this stage also the VZ inner boundary)
    # ------------------------------------------------------------------
    class CortexSurface(fenics.SubDomain): 

        def __init__(self, bmesh_bbtree):
            fenics.SubDomain.__init__(self)
            self.bmesh_bbtree = bmesh_bbtree

        def inside(self, x, on_boundary): 
            _, distance = self.bmesh_bbtree.compute_closest_entity(fenics.Point(*x)) # compute_closest_point() https://fenicsproject.org/olddocs/dolfin/1.5.0/python/programmers-reference/cpp/mesh/GenericBoundingBoxTree.html
            return fenics.near(distance, fenics.DOLFIN_EPS) # returns Points

    cortexsurface = CortexSurface(bmesh_bbtree)
    cortexsurface.mark(volume_facet_marker, 101, check_midpoint=False) # https://fenicsproject.discourse.group/t/how-to-compute-boundary-mesh-and-submesh-from-an-halfdisk-mesh/9812/2
    #cortexsurface.mark(boundary_face_marker, 101, check_midpoint=False)
    
    # brain regions growth mask
    # -------------------------  
    print("\nmark no-growth brain regions (e.g. 'longitudinal fissure' - 'ventricules' - 'mammilary bodies')...") # code source: T.Tallinen et al. 2016. See detail in https://github.com/rousseau/BrainGrowth/blob/master/geometry.py   
    """
    # initial value from T.Tallinen for delimating growth zones from inner zones: 0.6
    pond_Y = (abs(np.min(mesh.coordinates()[:,1])) + abs(np.max(mesh.coordinates()[:,1])))/2 # |dY|/2
    pond_Z = (abs(np.min(mesh.coordinates()[:,2])) + abs(np.max(mesh.coordinates()[:,2])))/2 # |dZ|/2
    new_critical_value = (pond_Y - H0)#0.6 * pond_Y
    """

    """for vertex, scalarDOF in enumerate(vertex2dofs_S):"""
    """
    # sphere filter but is 
    
    rqp = np.linalg.norm( np.array([  0.714*(mesh.coordinates()[vertex, 0] + 0.1), 
                                                mesh.coordinates()[vertex, 1], 
                                                mesh.coordinates()[vertex, 2] - 0.05  ]))"""
    """
    # sphere filter
    
    rqp = np.linalg.norm( np.array([  (mesh.coordinates()[vertex, 0]), 
                                        mesh.coordinates()[vertex, 1], 
                                        mesh.coordinates()[vertex, 2] - 0.10  ]))"""
    """
    # ellipsoid filter     
                            
    rqp = np.linalg.norm( np.array([    0.714 * mesh.coordinates()[vertex, 0], 
                                                mesh.coordinates()[vertex, 1], 
                                        0.9 *  (mesh.coordinates()[vertex, 2]) - 0.1*pond_Z ])) 

    """

    """
    # ellipsoid filter taking into account also the length of the brain (not to consider growth of inter-hemispheres longitudinal fissure)
        
    rqp = np.linalg.norm( np.array([    0.73 * mesh.coordinates()[vertex, 0], 
                                        0.85 * mesh.coordinates()[vertex, 1], 
                                        0.92 * (mesh.coordinates()[vertex, 2] - 0.15)   ])) """
                                        
    """                                 
    rqp = np.linalg.norm( np.array([    0.8 * characteristics["maxx"]/0.8251 * mesh.coordinates()[vertex, 0], # --> 0.8*maxX/0.8251
                                        0.63 * mesh.coordinates()[vertex, 1] + 0.075, # int(0.6/maxY)
                                        0.8 * characteristics["maxz"]/0.637 * mesh.coordinates()[vertex, 2] - 0.25   ])) # --> 0.8*maxZ/0.637
    """ 

    """
    dX = characteristics['maxx'] - characteristics['minx'] # 2*a
    dY = characteristics['maxy'] - characteristics['miny'] # 2*b
    dZ = characteristics['maxz'] - characteristics['minz'] # 2*c
    #a, b, c = 0.5 * dX, 0.5 * dY, 0.5 * dZ_whole_brain
    a, b, c = 0.9 * 0.5 * dX, 0.9 * 0.5 * dY,  0.9 * 0.5 * dZ 
    
    rqp = np.linalg.norm( np.array([    a/b * (mesh.coordinates()[vertex, 0] - center_of_gravity[0]), 
                                                (mesh.coordinates()[vertex, 1] - center_of_gravity[1]), 
                                        c/b * (mesh.coordinates()[vertex, 2] - center_of_gravity[2]) - 0.75 * pond_Z ])) 

    if rqp < new_critical_value:
        grGrowthZones.vector()[scalarDOF] = max(1.0 - 10.0*(new_critical_value - rqp), 0.0)
    else:
        grGrowthZones.vector()[scalarDOF] = 1.0
    """
    
    #dX = characteristics['maxx'] - characteristics['minx'] 
    #dZ = characteristics['maxz'] - characteristics['minz'] 
    #dY = characteristics['maxy'] - characteristics['miny']
    aX = 0.4 * dX
    bY = 0.35 * dY
    cZ = 0.5 * dZ # BrainGrowth ellipsoid: (1.0, 0.9, 0.7)

    for vertex, scalarDOF in enumerate(vertex2dofs_S):
            
        point_within_NoGrowthellipsoid = (mesh.coordinates()[vertex, 0] - 0.05*dX - center_of_gravity[0])**2 / aX**2 + (mesh.coordinates()[vertex, 1] - center_of_gravity[1])**2 / bY**2 + (mesh.coordinates()[vertex, 2] + 0.15*dZ - center_of_gravity[2])**2 / cZ**2 
                                            # - 0.05*dX --> positive offset X to the frontal lobe
                                            # 0.15*dZ --> negative offset Z to the bottom Z of the mesh
                                                                                    
        if point_within_NoGrowthellipsoid <= 1: 
            grGrowthZones.vector()[scalarDOF] = 0.0
        else:
            grGrowthZones.vector()[scalarDOF] = 1.0
            
    
    FEniCS_FEM_Functions_file.write(grGrowthZones, T0_in_GW) # for debugging
    
    # unilateral contact of each hemisphere onto two associated rigid planes (to correct auto-collisions between the two interhemispheres)
    # ----------------------------------------------------------------------
    # Define interhemisphere contact zone 
    # ---
    print("\ndefining the interhemispheric contact zone...") 
    #Ymin, Ymax = np.min(mesh.coordinates()[:,1]), np.max(mesh.coordinates()[:,1])
    #dY = Ymax - Ymin
    #Y_mean = 0.5 * (Ymin + Ymax)
    y_interhemisphere_plane = 0.5 * (characteristics['miny'] + characteristics['maxy'])
    interHemisphere_Zone_minY_maxY = y_interhemisphere_plane - dY/8, y_interhemisphere_plane + dY/8
    
    """
    def get_surface_node_coordinates(mesh, boundary_markers, surface_id):
        coordinates = []
        for vertex in fenics.vertices(mesh):
            for facet in fenics.facets(vertex):
                if boundary_markers[facet] == surface_id:
                    coordinates.append(vertex.point().array())
                    break
        return coordinates

    coords_103 = get_surface_node_coordinates(mesh, volume_facet_marker, 103) # contact boundary Right 
    coords_104 = get_surface_node_coordinates(mesh, volume_facet_marker, 104) # contact boundary Left 
    
    coords_103 = np.array(coords_103)
    coords_104 = np.array(coords_104)
    
    Ymin_Right = np.min(coords_103[:,1])
    Ymax_Left = np.max(coords_104[:,1])
    """
    
    # Ymin_Right = 0.030504 # 0.0292715 # [m]
    # Ymax_Left = 0.0280936 # 0.0284948 # [m]
    
    #gap_dHCP_28GW = 0.0025 # Ymin_Right - Ymax_Left # min gap between the two hemispheres measured on the real mesh in meters

    #y_interhemisphere_plane_103 =  y_interhemisphere_plane + 0.25 * gap_dHCP_28GW # Right (towards y positive)
    #y_interhemisphere_plane_104 =  y_interhemisphere_plane - 0.25 * gap_dHCP_28GW # Left

    # Mark contact boundaries
    # -----------------------
    class InterHemisphereContactZoneRight(fenics.SubDomain): # y > Y_mean 

        def __init__(self, y_interhemisphere_plane, interHemisphere_Zone, h):
            fenics.SubDomain.__init__(self)
            self.y_interhemisphere_plane = y_interhemisphere_plane
            self.interHemisphere_Zone = interHemisphere_Zone
            self.h = h

        def inside(self, x, on_boundary): 
            return x[1] >= self.y_interhemisphere_plane + self.h and x[1] < self.interHemisphere_Zone[1] and on_boundary

    hRight = 5.0e-4
    interHemisphereRightContactZone = InterHemisphereContactZoneRight(y_interhemisphere_plane, interHemisphere_Zone_minY_maxY, hRight)
    interHemisphereRightContactZone.mark(boundary_face_marker, 103)
    interHemisphereRightContactZone.mark(volume_facet_marker, 103)
    
    """
    bmesh_contact_zone_Right_hemisphere = fenics.SubMesh(bmesh, boundary_face_marker, 103)
    S_cortexsurface_contact_zone_Right = fenics.FunctionSpace(bmesh_contact_zone_Right_hemisphere, "CG", 1) 
    contact_pressure_Right = fenics.Function(S_cortexsurface_contact_zone_Right, name="ContactPressureRight")
    """
    
    ##
    
    class InterHemisphereContactZoneLeft(fenics.SubDomain): # y < Y_mean 

        def __init__(self, y_interhemisphere_plane, interHemisphere_Zone, h):
            fenics.SubDomain.__init__(self)
            self.y_interhemisphere_plane = y_interhemisphere_plane
            self.interHemisphere_Zone = interHemisphere_Zone
            self.h = h

        def inside(self, x, on_boundary): 
            return x[1] > self.interHemisphere_Zone[0] and x[1] <= self.y_interhemisphere_plane - self.h and on_boundary

    hLeft = 5.0e-4
    interHemisphereLeftContactZone = InterHemisphereContactZoneLeft(y_interhemisphere_plane, interHemisphere_Zone_minY_maxY, hLeft)
    interHemisphereLeftContactZone.mark(boundary_face_marker, 104)
    interHemisphereLeftContactZone.mark(volume_facet_marker, 104)
    
    """
    bmesh_contact_zone_Left_hemisphere = fenics.SubMesh(bmesh, boundary_face_marker, 104)
    S_cortexsurface_contact_zone_Left = fenics.FunctionSpace(bmesh_contact_zone_Left_hemisphere, "CG", 1) 
    contact_pressure_Left = fenics.Function(S_cortexsurface_contact_zone_Left, name="ContactPressureLeft")
    """
    
    # Mark zones where to fix nodes (prescribed displacement set to 0.) (Dirichlet BCs) --> https://fenicsproject.org/qa/2989/vertex-on-mesh-boundary/
    # --------------------------------------------------------------------------------
    d2v_S = fenics.dof_to_vertex_map(S)
        
    class DirichletBCs_VZ(fenics.SubDomain): 
        def __init__(self, radius_VZ, COG_bb, dX, dY, dZ, dilate_coef_x, dilate_coef_y, dilate_coef_z): 
            fenics.SubDomain.__init__(self)
            self.radius_VZ = radius_VZ
            self.COG_bb = COG_bb
            self.dilate_coef_x = dilate_coef_x
            self.dilate_coef_y = dilate_coef_y
            self.dilate_coef_z = dilate_coef_z
            self.dX, self.dY, self.dZ = dX, dY, dZ 
            #self.hmean = hmean
                        
        def inside(self, x, on_boundary):
            tol = 1E-14
            return ((x[0]-self.COG_bb[0])/(2.5*self.dilate_coef_x))**2 + ((x[1] - self.COG_bb[1])/(2.5*self.dilate_coef_y))**2 + ((x[2]-self.COG_bb[2])/(2.5*self.dilate_coef_z))**2 <= (self.radius_VZ)**2 and on_boundary
    
    # 2*self.dilate_coef_x; 2*self.dilate_coef_y and 2*self.dilate_coef_z --> to be sure to englobe the VZ boundary
    
    dirichletBCs_VZ = DirichletBCs_VZ(radius_VZ, COG_bb, dX, dY, dZ, dilate_coef_x, dilate_coef_y, dilate_coef_z)
    dirichletBCs_VZ.mark(volume_facet_marker, 102) # https://fenicsproject.discourse.group/t/how-to-compute-boundary-mesh-and-submesh-from-an-halfdisk-mesh/9812/2
    #dirichletBCs_VZ.mark(boundary_face_marker, 102)
    #dirichletBCs_VZ.mark(volume_facet_marker, 102, check_midpoint=False) # https://fenicsproject.discourse.group/t/how-to-compute-boundary-mesh-and-submesh-from-an-halfdisk-mesh/9812/2
    #dirichletBCs_VZ.mark(boundary_face_marker, 102, check_midpoint=False)
    
    # Mark zone between the 2 rigid planes
    # ------------------------------------
    print("\nmarking the boundary zone between the 2 interhemispheric rigid planes (99)...")
    class MyDict(dict): # https://fenicsproject.org/qa/5268/is-that-possible-to-identify-a-facet-by-its-vertices/
            def get(self, key):
                return dict.get(self, sorted(key))

    facet_2_vertices = MyDict((facet.index(), tuple(facet.entities(0))) for facet in fenics.facets(mesh)) # facet to 3 vertices indices
    
    tol = 1E-14 
    """
    for facet in fenics.facets(mesh):
        if facet.exterior() == True: # to select only faces at the exterior surface of the mesh (otherwise, inner faces are also marked)
            vertex1, vertex2, vertex3 = facet_2_vertices[facet.index()]
            #Y_COG_facet = 1/3 * (mesh.coordinates()[vertex1] + mesh.coordinates()[vertex2] + mesh.coordinates()[vertex3]) [1]
            Y_inner_facet = min(abs(mesh.coordinates()[vertex1][1]), min(abs(mesh.coordinates()[vertex2][1]), abs(mesh.coordinates()[vertex3][1])))
            if Y_inner_facet <= y_interhemisphere_plane + hRight + tol and Y_inner_facet <= abs(y_interhemisphere_plane - hLeft - tol): # facet in the interhemispheric zone, between the two rigid planes. This facet should not grow, and should not be taken as reference to build Mesh_Nt.
               if volume_facet_marker.array()[facet.index()] == 102:
                   pass
               else:
                   volume_facet_marker.array()[facet.index()] = 99
    """
    
    for facet in fenics.facets(mesh):
        if facet.exterior() == True: # to select only faces at the exterior surface of the mesh (otherwise, inner faces are also marked)

            vertex1, vertex2, vertex3 = facet_2_vertices[facet.index()]

            Y_COG_facet = 1/3 * (mesh.coordinates()[vertex1] + mesh.coordinates()[vertex2] + mesh.coordinates()[vertex3])[1]

            if Y_COG_facet > 0: # face belongs to Right hemisphere (Y+) 
                Y_inner_facet = min(mesh.coordinates()[vertex1][1], min(mesh.coordinates()[vertex2][1], mesh.coordinates()[vertex3][1]))                        
            else : # face belongs to Left hemisphere (Y-) 
                Y_inner_facet = max(mesh.coordinates()[vertex1][1], max(mesh.coordinates()[vertex2][1], mesh.coordinates()[vertex3][1]))

            if Y_inner_facet <= y_interhemisphere_plane + hRight + tol and Y_inner_facet >= y_interhemisphere_plane - hLeft - tol: # facet in the interhemispheric zone, between the two rigid planes. This facet should not grow, and should not be taken as reference to build Mesh_Nt.
               if volume_facet_marker.array()[facet.index()] == 102:
                   pass
               else:
                   volume_facet_marker.array()[facet.index()] = 99

    # Transfer labels marked on the whole mesh facets (volume) to the surface mesh faces (in order to get valid boundary_surface object)
    # ----------------------------------------------------------------------------------
    print("\ntransferring labels from the whole mesh facets ('volume_facet_marker') to the surface mesh faces ('boundary_face_marker')...")
    # build mapping from whole mesh facets to their associated coordinates (midpoint of the facet) 
    coords_2_meshfacet_index = {}
    
    # parse whole mesh facets (mesh)
    for facet in fenics.facets(mesh):
        midpoint = facet.midpoint()
        facet_midpoint_coords = tuple(np.round(midpoint.array(), decimals=6))  # --> get the coordinates of the midpoint (rounded to avoid precision mismatch)
        coords_2_meshfacet_index[facet_midpoint_coords] = facet.index()
    
    # parse surface mesh faces (bmesh)
    for face in fenics.faces(bmesh):
        midpoint = face.midpoint()
        face_midpoint_coords = tuple(np.round(midpoint.array(), decimals=6))
        
        if face_midpoint_coords in coords_2_meshfacet_index: # if the bmesh face (midpoint) coords corresponds to any of the mesh facets (midpoint) coords 
            
            facet_index = coords_2_meshfacet_index[face_midpoint_coords] # get the associated mesh facet index correspondint to the bmesh face (midpoint) coords
            
            if volume_facet_marker[facet_index] == 102: # facet_index refers to a face in the volume of the mesh
                boundary_face_marker[face.index()] = 102 # face_index refers to a face belonging to the mesh surface
            
            elif volume_facet_marker[facet_index] == 103: # contact boundary related to right hemisphere
                boundary_face_marker[face.index()] = 103
            
            elif volume_facet_marker[facet_index] == 104: # contact boundary related to left hemisphere
                boundary_face_marker[face.index()] = 104
            
            elif volume_facet_marker[facet_index] == 99: # interhemisphere boundary
                boundary_face_marker[face.index()] = 99
                
            elif volume_facet_marker[facet_index] == 101: # cortical surface boundary
                boundary_face_marker[face.index()] = 101
   
            
    """
    face_2_vertices = MyDict((face.index(), tuple(face.entities(0))) for face in fenics.faces(bmesh)) # facet to 3 vertices indices
    
    for face in fenics.faces(bmesh):
        vertex1, vertex2, vertex3 = face_2_vertices[face.index()]
        corresponding_whole_mesh_facet_index = [i for i in range(len(facet_2_vertices)) if (mesh.coordinates()[facet_2_vertices[i][0]].all() == bmesh.coordinates()[vertex1].all() and mesh.coordinates()[facet_2_vertices[i][1]].all() == bmesh.coordinates()[vertex2].all() and mesh.coordinates()[facet_2_vertices[i][2]].all() == bmesh.coordinates()[vertex3].all())][0]
        if volume_facet_marker.array()[corresponding_whole_mesh_facet_index] == 102:
            boundary_face_marker.array()[face.index()] = 102
        elif volume_facet_marker.array()[corresponding_whole_mesh_facet_index] == 103:
            boundary_face_marker.array()[face.index()] = 103
        elif volume_facet_marker.array()[corresponding_whole_mesh_facet_index] == 104:
            boundary_face_marker.array()[face.index()] = 104
    """
    
    # build cortical surface mesh corresponding to the surface elements labelled 101, exclusively 
    # -------------------------------------------------------------------------------------------
    print("\nbuilding the cortical surface bmesh (labels 101, but also 103 and 104)...")
    cortex_surface_labels = {101, 103, 104}

    #cortex_marker_function = fenics.MeshFunction("bool", bmesh, bmesh.topology().dim()-1) 
    cortex_marker_function = fenics.MeshFunction("size_t", bmesh, bmesh.topology().dim(), 0)
    #cortex_marker_function.array()[:] = [label in cortex_surface_labels for label in boundary_face_marker.array()]
    cortex_marker_function.array()[:] =  np.isin(boundary_face_marker.array(), list(cortex_surface_labels)).astype(int) # astype(int) converts boolean into 0 or 1

    bmesh_cortex = fenics.SubMesh(bmesh, cortex_marker_function, 1)

    """bmesh_cortex = fenics.SubMesh(bmesh, boundary_face_marker, 101)""" # part of the boundary mesh standing for the cortical surface --> at this stage, 101 corresponds only to the cortex boundary (contact boundaries 103 and 104, as well as Dirichlet zone 102 were removed)
    # to check values of boundary_face_marker --> boundary_face_marker.array() / np.where(boundary_face_marker.array()[:] == 104) 

    bmesh_cortexsurface_bbtree = fenics.BoundingBoxTree()
    bmesh_cortexsurface_bbtree.build(bmesh_cortex)  
    
    # export marked boundaries (both are required for the simulation)
    # ------------------------
    print("\nexporting marked boundaries ('volume_facet_marker' and 'boundary_face_marker')...")
    #export_XML_PVD_XDMF.export_PVDfile(args.output, 'no_growth_vertices', vertices_withNoGrowth)
    export_XML_PVD_XDMF.export_PVDfile(args.output, 'volume_facet_marker_T0', volume_facet_marker) # "volume_facet_marker" is used to define the surface zone for FEM integration (ds(xxx)) and in particular the surface where the Dirichlet boundary conditions apply.
    export_XML_PVD_XDMF.export_PVDfile(args.output, 'boundary_face_marker_T0', boundary_face_marker) # "boundary_face_marker" is used to build bmesh_cortex and the associated bmesh_cortexsurface_bbtree. And then, bmesh_cortexsurface_bbtree is used to define the FEM functions d2s and gm, required by the simulation (so to define the zone where there will be growth).  
    export_XML_PVD_XDMF.export_PVDfile(args.output, 'cortex_boundary_face_marker_T0', cortex_marker_function)
    """
    - boundary_volume --> 101, 102, 103, 104
    - boundary_surface --> 101 (to build 2D submeshes, here, 'bmesh_cortex')
    """

    # Measurement entities 
    # --------------------
    ds = fenics.Measure("ds", domain=mesh, subdomain_data=volume_facet_marker) 
    
    # FEM function spaces associated with the cortex surface
    ########################################################
    print("\ndefining FEM function spaces associated to the cortical surface mesh...")
    S_cortexsurface = fenics.FunctionSpace(bmesh_cortex, "CG", 1) 
    V_cortexsurface = fenics.VectorFunctionSpace(bmesh_cortex, "CG", 1) 
    
    # Additional mappings
    #####################
    print("\ncomputing mappings...")
    # From vertex to DOF in the whole mesh --> used to compute Mesh_Nt
    # ----------------------------------------------------------------
    vertex2dofs_B101 = mappings.vertex_to_dofs_VectorFunctionSpace(V_cortexsurface, gdim)

    # From the surface mesh (cortex envelop) to the whole mesh (B101_2_V_dofmap; vertexB101_2_dofsV_mapping --> used to compute Mesh_Nt)
    # --------------------------------------------------------
    B101_2_V_dofmap, vertexB101_2_dofsV_mapping = mappings.surface_to_mesh_V(gdim, V, V_cortexsurface, vertex2dofs_B101)
    Sboundary101_2_S_dofmap, vertexBoundary101Mesh_2_dofScalarFunctionSpaceWholeMesh_mapping = mappings.surface_to_mesh_S(S, S_cortexsurface)
    
    # From the whole mesh to the surface mesh (to be use for projections onto surface in contact process)
    # ---------------------------------------
    #vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping_t0 = mappings.mesh_to_surface_V(mesh.coordinates(), bmesh.coordinates()) # at t=0. (keep reference projection before contact for proximity node to deep tetrahedra vertices)
    
    # Residual Form
    ###############

    # prerequisites before computing Fg and mu (H, d2s and gm=f(d2s, H) required)
    # ----------------------------------------
    print("\ninitializing distances to surface...")
    d2s_ = differential_layers.compute_distance_to_cortexsurface(vertex2dofs_S, d2s, mesh, bmesh_cortexsurface_bbtree) # init at t=0.0
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
    projection.local_project(grTAN * gm * grGrowthZones * alphaTAN * dt_in_seconds, S, dg_TAN) 
    #projection.local_project(grTAN * gm * alphaTAN * dt_in_seconds, S, dg_TAN) 
    projection.local_project(grRAD * (1 - gm) * alphaRAD * dt_in_seconds, S, dg_RAD) 

    print("\ninitializing normals to boundary...")
    boundary_normals = growth.compute_topboundary_normals(mesh, ds(101) + ds(103) + ds(104), V) # ds(101) + ds(103) + ds(104) only cortical surface (labelled 101 but also including the contact zones 103 and 104) must be identified as the reference boundary where to compute the normals for the growth tensor orientation 
    projection.local_project(boundary_normals, V, BoundaryMesh_Nt)

    print("\ninitializing projected normals of nodes of the whole mesh...")
    mesh_normals = growth.compute_mesh_projected_normals(V, mesh.coordinates(), bmesh_cortex.coordinates(), vertexB101_2_dofsV_mapping, vertex2dofs_V, BoundaryMesh_Nt) 
    projection.local_project(mesh_normals, V, Mesh_Nt)

    print("\ninitializing growth tensor...")
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

    # F: deformation gradient
    # -----------------------
    Id = fenics.Identity(gdim)
    F = fenics.variable( Id + fenics.grad(u) ) # F = I₃ + ∇u / [F]: 3x3 matrix

    # Fe: elastic part of the deformation gradient
    # --------------------------------------------
    Fg_inv = fenics.variable( fenics.inv(Fg) )
    Fe = fenics.variable( F * Fg_inv )# F_t * (F_g)⁻¹

    # Cauchy-Green tensors (elastic part of the deformation only)
    # --------------------
    Ce = fenics.variable( Fe.T * Fe )
    Be = fenics.variable( Fe * Fe.T )
    
    Ee = 0.5*(Ce - Id)  # Green-Lagrange tensor --> visualize strain

    # Invariants 
    # ----------
    Je = fenics.variable( fenics.det(Fe) ) 
    Tre = fenics.variable( fenics.tr(Ce) )
    
    # Neo-Hookean strain energy density function
    # ------------------------------------------
    #We = 0.5 * mu * (Tre * pow(Je, -2/3) - 3) + 0.5 * K * (Je - fenics.ln(Je) - 1) # T.Tallinen
    We = 0.5 * mu * (Tre * pow(Je, -2/3) - 3) + 0.5 * K * (Je - 1) * (Je - 1) # X.Wang https://github.com/rousseau/BrainGrowth/ incompressible Neo-Hookean (with penalization term to force incompressibility)
    
    """
    lmbda = K - 2*mu/3
    We = 0.5 * mu * (Tre - 3) - mu * fenics.ln(Je) + (lmbda / 2) * (fenics.ln(Je))**2 # compressible Neo-Hookean
    """
    
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
    PK1tot = fenics.diff(We, F) # -> visualize stress
    
    # Elastic part of the residual form (conservation law)
    # ---------------------------------
    print("\ngenerating Residual Form to minimize...")
    res = fenics.inner(PK1tot, fenics.grad(v_test)) * fenics.Measure("dx")

    # Pre-requisites to implement the penalty method
    # ----------------------------------------------
    print("\nsetting parameters and functions to apply the penalty method...")
    # epsilon = fenics.Constant(1e2) # penalization coefficient--> h/E
    #h = mesh.hmin()
    #E = 9 * KCortex.values()[0] * muCortex.values()[0] / (3 * KCortex.values()[0] + muCortex.values()[0])
    #epsilon = h / E 
    epsilon = args.parameters["epsilon_n"] # 1e5, 1e6
    print("epsilon (penalization coefficient): {}".format(epsilon)) 
    #h = 0. #0.00005 # since max displacement when contact between lobes occurs at 23GW is ~0.0001 (with no Contact Mechanics added in residual form <-- not true)
    
    # Here self-contact is replaced by a contact between a deformable solid (Left or Right hemisphere) and a rigid plane (the interhemisphere plane)
    
    def normal_gap_Y(u, y_plane, mesh): # Definition of gap function (if gap < 0 => penalization)
        x = fenics.SpatialCoordinate(mesh) # in order to recompute x at each new reference configuration
        return (x[1] + u[1]) - y_plane # compute each time new virtual configuration at t+1 
        # --> = gn
    
    """
    def normal_gap(u, y_plane, BoundaryMesh_Nt, mesh): # Definition of gap function (if gap < 0 => penalization)
        x = fenics.SpatialCoordinate(mesh) # in order to recompute x at each new reference configuration
        return fenics.dot((y_plane - (x[1] + u[1])) * fenics.Constant((0., 1., 0.)) , BoundaryMesh_Nt) # compute each time new virtual configuration at t+1
        # --> = gn
    """
        
    def mackauley(x):
        return (x + abs(x))/2
    
    # Compute the normal gaps for left and right hemispheres 
    # ------------------------------------------------------
    print("\nexpressing the normal gaps for left and right hemispheres...")
    # compute normal gap gn between brain surface (on the Left or Right contact zone) and the interhemisphere plane
    #gn = normal_gap_Y(u, y_interhemisphere_plane, mesh) # xS - xM
    gn = normal_gap_Y(u, y_interhemisphere_plane, mesh) # xS - xM# Right
    #gn = normal_gap(u, y_interhemisphere_plane, BoundaryMesh_Nt, mesh) # xS - xM
    
    # Penalize collision of Right-hemisphere onto interhemisphere plane (Y) (Add contact to the residual form)
    # ---------------------------------------------------------------------
    y_axis_referentiel = fenics.as_vector((0., 1., 0.))
    
    #res += epsilon * fenics.dot( mackauley(gn) * BoundaryMesh_Nt, v_test ) * ds(103) 
    res -= epsilon * fenics.dot( fenics.conditional(- gn > 0, - gn, 0) * y_axis_referentiel, v_test ) * ds(103) 
    # res += epsilon * fenics.dot( mackauley(-gn) * BoundaryMesh_Nt, v_test ) * ds(103)
    
    # Penalize collision of Left-hemisphere onto interhemisphere plane (Y) (Add contact to the residual form)
    # --------------------------------------------------------------------
    #res += epsilon * fenics.dot( mackauley(-gn) * BoundaryMesh_Nt, v_test ) * ds(104) 
    res -= epsilon * fenics.dot( fenics.conditional(gn > 0, gn, 0) * (-y_axis_referentiel), v_test ) * ds(104) 
    # res += epsilon * fenics.dot( mackauley(-gn) * BoundaryMesh_Nt, v_test ) * ds(104)

    # Non Linear Problem to solve
    # ---------------------------
    print("\nexpressing the non linear variational problem to solve...")
    jacobian = fenics.derivative(res, u, du) 

    bc_Dirichlet = fenics.DirichletBC(V, fenics.Constant((0., 0., 0.)), volume_facet_marker, 102) # no growth zones (grGrowthZones = 0) are fixed. no displacement in x,y,z --> fixed zone to avoid additional solution including Rotations & Translations
    bcs = [bc_Dirichlet]
    
    nonlinearvariationalproblem = fenics.NonlinearVariationalProblem(res, u, bcs, jacobian)   

    ######################################
    ############### Solver ###############
    ######################################

    # Parameters
    ############
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
    ###################################################################################
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
    FEniCS_FEM_Functions_file.write(grGrowthZones, T0_in_GW)

    FEniCS_FEM_Functions_file.write(BoundaryMesh_Nt, T0_in_GW)
    FEniCS_FEM_Functions_file.write(Mesh_Nt, T0_in_GW)

    FEniCS_FEM_Functions_file.write(dg_TAN, T0_in_GW)
    FEniCS_FEM_Functions_file.write(dg_RAD, T0_in_GW)
    FEniCS_FEM_Functions_file.write(Fg_T, T0_in_GW)

    FEniCS_FEM_Functions_file.write(mu, T0_in_GW)
    FEniCS_FEM_Functions_file.write(K, T0_in_GW)

    initial_time = time.time()
    for i, dt in enumerate( tqdm( np.diff(times), desc='brain is growing...', leave=True) ): # dt = dt_in_seconds

    # collisions (fcontact_global_V) have to be detected at each step

        #fenics.set_log_level(fenics.LogLevel.ERROR) # in order not to print solver info logs about Newton solver tolerances
        fenics.set_log_level(fenics.LogLevel.INFO) # in order to print solver info logs about Newton solver tolerances
        
        t = times[i+1] # in seconds
        t_in_GW = t / 604800 
        print("\n")
        print("tGW = {:.2f}".format(t_in_GW))
        
        # Update pre-required entities
        # ----------------------------
        # H
        """h.t = t
        #H.assign( fenics.project(h, S) )# Expression -> scalar Function of the mesh
        projection.local_project(cortical_thickness, S, H) # H.assign( fenics.project(cortical_thickness, S) )  
        """
        
        # d2s
        #d2s.assign( differential_layers.compute_distance_to_cortexsurface(vertex2dofs_S, d2s, mesh, bmesh_cortexsurface_bbtree) )
        d2s_ = differential_layers.compute_distance_to_cortexsurface(vertex2dofs_S, d2s, mesh, bmesh_cortexsurface_bbtree) 
        projection.local_project(d2s_, S, d2s)
        
        # gm
        #gm = differential_layers.compute_differential_term_DOF(S, d2s, H, gm)
        gm_ = differential_layers.compute_differential_term_DOF(S, d2s, H, gm) 
        projection.local_project(gm_, S, gm)

        # Update differential material stiffness mu 
        # -----------------------------------------
        # mu have to be updated at each timestep (material properties evolution with deformation) 
        mu_ = differential_layers.compute_shear_and_bulk_stiffnesses(gm, muCore, muCortex)
        projection.local_project(mu_, S, mu)

        K_ = differential_layers.compute_shear_and_bulk_stiffnesses(gm, KCore, KCortex)
        projection.local_project(K_, S, K)

        # Update growth tensor coefficients
        # ---------------------------------
        projection.local_project(grTAN * gm * grGrowthZones * alphaTAN * dt, S, dg_TAN)
        #projection.local_project(grTAN * gm * alphaTAN * dt, S, dg_TAN)
        projection.local_project(grRAD * (1 - gm) * alphaRAD * dt, S, dg_RAD) 
        
        # Update growth tensor orientation (adaptative)
        # ---------------------------------------------
        boundary_normals = growth.compute_topboundary_normals(mesh, ds(101) + ds(103) + ds(104), V) 
        projection.local_project(boundary_normals, V, BoundaryMesh_Nt)
        
        mesh_normals = growth.compute_mesh_projected_normals(V, mesh.coordinates(), bmesh_cortex.coordinates(), vertexB101_2_dofsV_mapping, vertex2dofs_V, BoundaryMesh_Nt) 
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
        
        projection.local_project(fenics.det(F), S, J) # local volume change (for incompressible material, should be close to 1)
        FEniCS_FEM_Functions_file.write(J, t_in_GW)
        
        # contact pressure
        """
        projection.local_project(epsilon * gn, S_cortexsurface_contact_zone_Right, contact_pressure_Right)
        projection.local_project(epsilon * (-gn), S_cortexsurface_contact_zone_Left, contact_pressure_Left)
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
        fenics.ALE.move(mesh, u)
        #mesh.smooth()
        #mesh.smooth_boundary(10, True) # --> to smooth the contact boundary after deformation and avoid irregular mesh surfaces at the interhemisphere https://fenicsproject.discourse.group/t/3d-mesh-generated-from-imported-surface-volume-exhibits-irregularities/3293/6 

        # Boundary (if d2s and Mesh_Nt need to be udpated: "solver_Fgt_norm"; "solver_Fgt")       
        bmesh = fenics.BoundaryMesh(mesh, "exterior") # whole mesh envelop

        bmesh_cortex = fenics.SubMesh(bmesh, cortex_marker_function, 1)
        """bmesh_cortex = fenics.SubMesh(bmesh, boundary_face_marker, 101)""" # cortex surface including also the contact boundary (that are growing) but not he noGrowth Zone nor the inner zone labelled 99
        
        bmesh_cortexsurface_bbtree = fenics.BoundingBoxTree()
        bmesh_cortexsurface_bbtree.build(bmesh_cortex) 

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






