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
from scipy.spatial import cKDTree
import nibabel as nib
import meshio

sys.path.append(sys.path[0]) # BrainGrowth3D
sys.path.append(os.path.dirname(sys.path[0])) # braingrowthFEniCS

from FEM_biomechanical_model import preprocessing, numerical_scheme_spatial, mappings, differential_layers, growth, projection
from utils.export_functions import export_simulation_end_time_and_iterations, export_XML_PVD_XDMF
from utils.converters import convert_meshformats
from utils import mesh_refiner


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='braingrowthFEniCS: halfsphere growth quasistatic 3D model')

    parser.add_argument('-i', '--input', help='Input mesh path (xml)', type=str, required=False, 
                        default="./data/dHCP_surface_vs_volume_RAW/Vf/dhcp_surface_t21_raw_130000faces_455983tets_reoriented_dHCPVolume.xdmf")     
                        # half sphere mesh: './data/halfsphere/halfsphere_radius1meter_refinedWidthCoef5.xdmf' # meters
                        # from dHCP surface gifti mesh: "./data/dHCP_surface_vs_volume_RAW/Vf/dhcp_surface_t21_raw_130000faces_455983tets_reoriented_dHCPVolume.xdmf"--> in millimeters
                        # from dHCP volume niftis: "./data/dHCP_surface_vs_volume_RAW/Vf/dhcp_volume_t21_raw_130000faces_480112tets.xdmf" -->  in millimeters

    parser.add_argument('-c', '--convertmesh0frommillimetersintometers', help='Convert mesh from millimeters into meters', type=bool, required=False, default=True)

    parser.add_argument('-p', '--parameters', help='Simulation input parameters', type=json.loads, required=False, 
                        default={"H0": 3.0e-3, # [m] 
                                 "muCortex": 1000, "muCore": 100, # [Pa] 
                                 "nu": 0.45,
                                 "alphaTAN": 4.0e-7, "alphaRAD": 0.5e-7, "grTAN": 1.0, "grRAD": 1.0, # alphaTAN: [s⁻¹], alphaRAD: [s⁻¹] 
                                 "T0_in_GW": 21.0, "Tmax_in_GW": 36.0, "dt_in_seconds": 43200, # 0.5GW (1GW=168h=604800s)
                                 "linearization_method":"newton", 
                                 "newton_absolute_tolerance":1E-9, "newton_relative_tolerance":1E-6, "max_iter": 10, 
                                 "linear_solver":"mumps"}) 
    
    parser.add_argument('-o', '--output', help='Output folder path', type=str, required=False, 
                        default='./results/halfellipsoid_EXTRUDED_growth_DirichletBCsFixed/')
               
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
    cortical_thickness = fenics.Expression('H0 + 0.01*t', H0=H0, t=0.0, degree=0)
    gdim=3
    
    # Mesh
    # ----
    # mesh & boundary mesh
    print("\nimporting brain mesh...")
    
    inputmesh_path = args.input
    inputmesh_format = inputmesh_path.split('.')[-1]

    if inputmesh_format == "xml":
        brain_mesh = fenics.Mesh(inputmesh_path)

    elif inputmesh_format == "xdmf":
        brain_mesh = fenics.Mesh()
        with fenics.XDMFFile(inputmesh_path) as infile:
            infile.read(brain_mesh)
            
    # convert initial input whole brain mesh and compute its characteristics
    if args.convertmesh0frommillimetersintometers == True:
        brain_mesh = preprocessing.converting_mesh_from_millimeters_into_meters(brain_mesh)
        
    # input mesh was generated with Gmsh --> in meters. 
    # It is required to resize the mesh towards 21GW brain real size range. i.e. brain radius at 21GW ~ 30 mm = 0.03 m => multiply all coords by 0.03
    #mesh.coordinates()[:] = mesh.coordinates()[:] * 0.03
    
    # Build ellipsoid bounding box for brain mesh
    #############################################
    print("\nbuild ellipsoid bounding box mesh...")
    brain_maxX, brain_minX = np.max(brain_mesh.coordinates()[:,0]), np.min(brain_mesh.coordinates()[:,0])
    brain_maxY, brain_minY = np.max(brain_mesh.coordinates()[:,1]), np.min(brain_mesh.coordinates()[:,1])
    brain_maxZ, brain_minZ =  np.max(brain_mesh.coordinates()[:,2]), np.min(brain_mesh.coordinates()[:,2])

    dX = brain_maxX - brain_minX
    dY = brain_maxY - brain_minY
    dZ = brain_maxZ - brain_minZ
    print("dX = {}mm, dY = {}mm, dZ = {}mm".format(dX*1000, dY*1000, dZ*1000))

    hmin_symbolic = fenics.CellDiameter(brain_mesh) # symbolic expression providing cell diameter [m] for each cell of the mesh 
    Z = fenics.FunctionSpace(brain_mesh, "DG", 0)
    hmin_symbolic_proj = fenics.project(hmin_symbolic, Z) 
    min_cell_size = np.nanmin(hmin_symbolic_proj.vector()[:])
    average_cell_size = np.nanmean(hmin_symbolic_proj.vector()[:])
    max_cell_size = np.nanmax(hmin_symbolic_proj.vector()[:]) 
    print("max cell size in brain mesh = {} m".format(max_cell_size))

    hmin = min_cell_size
    hmean = average_cell_size
    hmax = max_cell_size
    
    # parameters
    mesh_bb_MSH_path = "./data/dHCP_surface_vs_volume_RAW/Vf/halfellipsoidBB_EXTRUDED_dhcp_surface_t21_raw_reoriented_dHCPVolume.msh"
    visualization = False

    COG_bb = 0.5*(brain_minX + brain_maxX), 0.5*(brain_minY + brain_maxY), 0.5*(brain_minZ + brain_maxZ)

    # create and write (.msh) tetraedral box
    from utils.mesh_creator import create_MSH_ellipsoid, create_MSH_halfellipsoid

    """mesh = create_MSH_ellipsoid.create_MSH_ellipsoid_mesh(mesh_bb_MSH_path, 
                                                         COG_bb[0], COG_bb[1], COG_bb[2],
                                                         1, 
                                                         0.5*dX, 0.5*dY, 0.5*dZ, # (dX + 4e-3)*0.75, (dY + 4e-3)*0.75, (dZ + 4e-3)*0.75, 
                                                         hmean, hmean, 
                                                         1,
                                                         visualization)"""
    
    box_X, box_Y, box_Z = brain_minX, brain_minY, brain_minZ
    box_dX, box_dY, box_dZ = dX, dY, dZ/2
    
    mesh = create_MSH_halfellipsoid.create_MSH_halfellipsoid_emptiedVentricularZone_mesh(mesh_bb_MSH_path, 
                                                                                        COG_bb[0], COG_bb[1], COG_bb[2],
                                                                                        1,
                                                                                        0.5*dX, 0.5*dY, 0.5*dZ, 
                                                                                        hmean, hmean, 
                                                                                        box_X, box_Y, box_Z, box_dX, box_dY, box_dZ,
                                                                                        1,
                                                                                        visualization) # N.B. output mesh is a GMSH object


    # convert mesh into .xdmf format
    mesh_bb_XDMF_path = "./data/dHCP_surface_vs_volume_RAW/Vf/halfellipsoidBB_EXTRUDED_dhcp_surface_t21_raw_reoriented_dHCPVolume.xdmf"
    #mesh = meshio.read(mesh_bb_MSH_path) 
    meshio.write(mesh_bb_XDMF_path, meshio.Mesh(points=mesh.points, cells={'tetra': mesh.cells_dict['tetra']}))

    mesh = fenics.Mesh()
    with fenics.XDMFFile(mesh_bb_XDMF_path) as infile: 
        infile.read(mesh) 
        
    # refine mesh near by surface
    #############################
    mesh_spacing = mesh.hmin()

    brainsurface_bmesh = fenics.BoundaryMesh(mesh, "exterior") # get required args for refinement function

    brainsurface_bmesh_bbtree = fenics.BoundingBoxTree()
    brainsurface_bmesh_bbtree.build(brainsurface_bmesh) 
    
    refinement_width_coef = 2
    
    mesh = mesh_refiner.refine_mesh_on_brainsurface_boundary(mesh, brainsurface_bmesh_bbtree, mesh_spacing, refinement_width_coef) # N.B. output mesh is a FEniCS (.xdmf) object
    
    # export refined mesh into .xdmf format
    halfellipsoid_mesh_refined_XDMF_path = "./data/dHCP_surface_vs_volume_RAW/Vf/halfellipsoidBB_EXTRUDED_refinedCoef2_dhcp_surface_t21_raw_reoriented_dHCPVolume.xdmf"
    #mesh = meshio.read(mesh_bb_MSH_path) 
    meshio.write(halfellipsoid_mesh_refined_XDMF_path, meshio.Mesh(points=mesh.coordinates(), cells={'tetra': mesh.cells()}))
    
    ####
    bmesh = fenics.BoundaryMesh(mesh, "exterior") # bmesh at t=0.0 (cortex envelop)

    if args.visualization == True:
        fenics.plot(mesh) 
        plt.title("input mesh")
        plt.show() 

    # reorient gifti mesh
    #preprocessing.reorient_gifti_mesh('./data/brain/dHCP_21GW/fetal.week21.right.pial.surf.gii')

    # mesh characteristics
    characteristics0 = preprocessing.compute_geometrical_characteristics(mesh, bmesh)
    center_of_gravity0 = preprocessing.compute_center_of_gravity(characteristics0) 
    min_mesh_spacing0, average_mesh_spacing0, max_mesh_spacing0 = preprocessing.compute_mesh_spacing(mesh)
    print('input mesh characteristics: {}'.format(characteristics0))
    print('input mesh COG = [xG0:{}, yG0:{}, zG0:{}]'.format(center_of_gravity0[0], center_of_gravity0[1], center_of_gravity0[2]))
    print("input mesh min mesh spacing: {:.3f} mm".format(min_mesh_spacing0))
    print("input mesh mean mesh spacing: {:.3f} mm".format(average_mesh_spacing0))
    print("input mesh max mesh spacing: {:.3f} mm".format(max_mesh_spacing0))

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
    
    # Boundaries
    ############
    print("\ncomputing and marking boundaries...")
    #bmesh_cortexsurface_bbtree = fenics.BoundingBoxTree()
    #bmesh_cortexsurface_bbtree.build(bmesh) 

    # initialize boundaries
    regions = fenics.MeshFunction('size_t', mesh, mesh.topology().dim())
    regions.set_all(0)

    #boundaries = fenics.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    boundaries_volume = fenics.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)  
    boundaries_volume.set_all(100)

    boundaries_surface = fenics.MeshFunction('size_t', bmesh, bmesh.topology().dim(), 0)
    boundaries_surface.set_all(100)
    
    # define part of the boundary to fix (Dirichlet BCs) 

    """
    class CortexSurface(fenics.SubDomain): 
        def __init__(self, Z_midsphere_plane):
            fenics.SubDomain.__init__(self)
            self.Z_midsphere_plane = Z_midsphere_plane

        def inside(self, x, on_boundary):  
            tol = 1E-14
            return x[2] > self.Z_midsphere_plane # need to remove "on_boundary" to build submesh of a boundary mesh --> https://fenicsproject.discourse.group/t/how-to-build-vectorfunctionspace-on-part-of-a-2d-mesh-boundary/9648; returns all boundary points except those on interhemisphere plane, supposed to be fixed.
    
    Z_midsphere_plane = 0.0
    cortexsurface = CortexSurface(Z_midsphere_plane)
    cortexsurface.mark(boundaries_volume, 101, check_midpoint=False) # https://fenicsproject.discourse.group/t/how-to-compute-boundary-mesh-and-submesh-from-an-halfdisk-mesh/9812/2
    cortexsurface.mark(boundaries_surface, 101, check_midpoint=False) # # https://fenicsproject.discourse.group/t/how-to-compute-boundary-mesh-and-submesh-from-an-halfdisk-mesh/9812/2
    # cortexsurface.mark(regions, 1)
    """

    class DirichletBCs(fenics.SubDomain): 
        def __init__(self, brain_minZ, dZ):
            fenics.SubDomain.__init__(self)
            self.brain_minZ = brain_minZ
            self.dZ = dZ
            
         # midsphere plane: Z=0.0
        def inside(self, x, on_boundary):
            tol = 1E-14
            return fenics.near(x[2], self.brain_minZ + self.dZ/2, tol) #and on_boundary
        
    dirichletBCs = DirichletBCs(brain_minZ, dZ)
    dirichletBCs.mark(boundaries_volume, 102, check_midpoint=False) # https://fenicsproject.discourse.group/t/how-to-compute-boundary-mesh-and-submesh-from-an-halfdisk-mesh/9812/2
    dirichletBCs.mark(boundaries_surface, 102, check_midpoint=False)

    bmesh_cortex = fenics.SubMesh(bmesh, boundaries_surface, 100) # part of the boundary mesh standing for the cortical surface
    with fenics.XDMFFile(MPI.COMM_WORLD, args.output + "bmesh_cortex.xdmf") as xdmf:
        xdmf.write(bmesh_cortex)
    """
    halfsphere = fenics.SubMesh(mesh, regions, 1)
    """

    #print("\ncomputing and marking boundaries...")
    bmesh_cortexsurface_bbtree = fenics.BoundingBoxTree()
    bmesh_cortexsurface_bbtree.build(bmesh_cortex)                
    
    # export marked boundaries
    export_XML_PVD_XDMF.export_PVDfile(args.output, 'boundaries_volume_T0', boundaries_volume)

    # Subdomains
    ############
    #subdomains = fenics.MeshFunction("size_t", mesh, mesh.topology().dim()) 
    #subdomains.set_all(0)
    
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

    """
    u_old = fenics.Function(V) # Fields from previous time step (displacement, velocity, acceleration)
    v_old = fenics.Function(V)
    a_old = fenics.Function(V)
    """

    BoundaryMesh_Nt = fenics.Function(V, name="BoundaryMesh_Nt")
    Mesh_Nt = fenics.Function(V, name="Mesh_Nt")

    # Vector functions of Vtensor
    Fg_T = fenics.Function(Vtensor, name="Fg")
    PK1tot_T = fenics.Function(Vtensor, name="PK1tot") 
    Ee_T = fenics.Function(Vtensor, name="E_strain") 

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
    ds = fenics.Measure("ds", domain=mesh, subdomain_data=boundaries_volume) 

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
        
    # brain regions growth mask (avoid growth in Dirichlet boundary zone)
    # ------------------------- 
    # add growth for all mesh nodes
    """
    grGrowthZones.vector()[:] = 1.0 
    
    for vertex, dof in enumerate(vertex2dof_S):
        if mesh.coordinates()[vertex, 2] < -0.7 * 0.03:
            grGrowthZones.vector()[dof] = 0.0
    """
        
    # remove growth for Dirichlet fixed nodes
    """
    class MyDict(dict): # https://fenicsproject.org/qa/5268/is-that-possible-to-identify-a-facet-by-its-vertices/
        def get(self, key):
            return dict.get(self, sorted(key))

    f_2_v = MyDict((facet.index(), tuple(facet.entities(0))) for facet in fenics.facets(mesh))
    
    for face in fenics.facets(mesh):
        vertex1, vertex2, vertex3 = f_2_v[face.index()]
        if boundaries_volume.array()[face.index()] == 102: # Dirichlet boundary (face is supposed to be fixed --> no growth)
            grGrowthZones.vector()[vertex2dof_S[vertex1]] = 0.0
            grGrowthZones.vector()[vertex2dof_S[vertex2]] = 0.0
            grGrowthZones.vector()[vertex2dof_S[vertex3]] = 0.0
    """

    # Fg
    # --
    print("\ninitializing growth coefficients: dgTAN & dgRAD...")
    #projection.local_project(grTAN * gm * grGrowthZones * alphaTAN * dt_in_seconds, S, dg_TAN) 
    projection.local_project(grTAN * gm * alphaTAN * dt_in_seconds, S, dg_TAN) 
    projection.local_project(grRAD * (1 - gm) * alphaRAD * dt_in_seconds, S, dg_RAD)  # projection.local_project(grRAD * alphaRAD * dt_in_seconds, S, dg_RAD) 

    print("\ninitializing normals to boundary...")
    boundary_normals = growth.compute_topboundary_normals(mesh, ds(100), V)
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
    F_inv = fenics.inv(F)

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

    # Rolling BCS on the ellipsoid halfplane z=0
    # ------------------------------------------
    zero = fenics.Constant(0.0)
    #bc_Dirichlet_rolling_fixedX = fenics.DirichletBC(V.sub(0), zero, dirichletBCs)     # uX = 0
    #bc_Dirichlet_rolling_fixedY = fenics.DirichletBC(V.sub(1), zero, dirichletBCs)   # uY = 0
    """bc_Dirichlet_rolling_fixedZ = fenics.DirichletBC(V.sub(2), zero, dirichletBCs)"""    # uZ = 0
        
    # Fully-fixed BCS
    # ---------------
    bc_Dirichlet_fixedXYZ = fenics.DirichletBC(V, fenics.Constant((0., 0., 0.)), boundaries_volume, 102) # no displacement in x,y,z --> fixed zone to avoid additional solution including Rotations & Translations
    
    bcs = [bc_Dirichlet_fixedXYZ]
    nonlinearvariationalproblem = fenics.NonlinearVariationalProblem(res, u, bcs, jacobian)   

    ####################################################
    ###################### Solver ######################
    ####################################################

    # Parameters
    # ----------
    nonlinearvariationalsolver = fenics.NonlinearVariationalSolver(nonlinearvariationalproblem) 
    # info(nonlinearvariationalsolver.parameters, True) # display the list of available parameters and default values
    # https://home.simula.no/~hpl/homepage/fenics-tutorial/release-1.0-nonabla/fenics_tutorial_1.0.pdf
    #https://link.springer.com/content/pdf/10.1007/978-3-319-52462-7_5.pdf
    # https://fenicsproject.org/qa/5894/nonlinearvariationalsolver-tolerance-what-solver-options/ (first used)

    # SOLVER PARAMETERS FOR NON-LINEAR PROBLEM 
    nonlinearvariationalsolver.parameters["nonlinear_solver"] = args.parameters["linearization_method"] 
    #nonlinearvariationalsolver.parameters['newton_solver']['convergence_criterion'] = "incremental" 
    nonlinearvariationalsolver.parameters['newton_solver']['absolute_tolerance'] = args.parameters["newton_absolute_tolerance"] 
    nonlinearvariationalsolver.parameters['newton_solver']['relative_tolerance'] = args.parameters["newton_relative_tolerance"] 
    nonlinearvariationalsolver.parameters['newton_solver']['maximum_iterations'] = args.parameters["max_iter"] 
    nonlinearvariationalsolver.parameters['newton_solver']['relaxation_parameter'] = 1.0 # means "full" Newton-Raphson iteration expression: u_k+1 = u_k - res(u_k)/res'(u_k) => u_k+1 = u_k - res(u_k)/jacobian(u_k)

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
    #nonlinearvariationalsolver.parameters['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True # https://link.springer.com/content/pdf/10.1007/978-3-319-52462-7_5.pdf --> "Using a nonzero initial guess can be particularly important for timedependent problems or when solving a linear system as part of a nonlinear iteration, since then the previous solution vector U will often be a good initial guess for the solution in the next time step or iteration."
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

    """start_time = time.time ()"""
    """
    energies = np.zeros((int(Nsteps+1), 4))
    E_damp = 0
    E_ext = 0
    """
    
    initial_time = time.time()
    for i, dt in enumerate( tqdm( np.diff(times), desc='brain is growing...', leave=True) ): # dt = dt_in_seconds

        fenics.set_log_level(fenics.LogLevel.ERROR) # in order not to print solver info logs 

        """
        t_in_GW = times[i+1]
        t = t_in_GW * 604800 # in seconds
        """
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
        projection.local_project(grRAD * (1 - gm) * alphaRAD * dt_in_seconds, S, dg_RAD)  # projection.local_project(grRAD * alphaRAD * dt, S, dg_RAD) 

        # Update growth tensor orientation (adaptative)
        # ---------------------------------------------
        boundary_normals = growth.compute_topboundary_normals(mesh, ds(100), V) 
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
        
        # Save energies https://fenicsproject.org/olddocs/dolfin/2019.1.0/python/demos/elastodynamics/demo_elastodynamics.py.html
        ###############          
        """
        E_elas = fenics.assemble(0.5 * numerical_scheme_spatial.k(numerical_scheme_temporal.avg(u_old, u, alphaF), v_test, Fg_T, mu, K, gdim) ) 
        E_kin = fenics.assemble(0.5 * numerical_scheme_spatial.m(rho, numerical_scheme_temporal.avg(a_old, a_new, alphaM), v_test) )
        E_damp += dt * fenics.assemble( numerical_scheme_spatial.c(damping_coef, numerical_scheme_temporal.avg(v_old, v_new, alphaF), v_test) )
        # E_ext += assemble( Wext(u-u_old) )
        E_tot = E_elas + E_kin + E_damp #-E_ext
        
        energies[i+1, :] = np.array([E_elas, E_kin, E_damp, E_tot])
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

        bmesh_cortex = fenics.SubMesh(bmesh, boundaries_surface, 100)
        
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






