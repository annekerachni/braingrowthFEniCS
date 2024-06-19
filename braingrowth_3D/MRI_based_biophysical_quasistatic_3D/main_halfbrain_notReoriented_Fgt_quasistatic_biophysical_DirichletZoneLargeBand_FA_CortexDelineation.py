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

sys.path.append(sys.path[0])
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from FEM_biomechanical_model_quasistatic import preprocessing, numerical_scheme_spatial, mappings, differential_layers, growth, projection
from utils.export_functions import export_simulation_outputmesh_data, export_simulation_end_time_and_iterations, export_XML_PVD_XDMF
from utils.converters import convert_meshformats
from utils import mesh_refiner

from niftitomesh.niftivalues2meshnodes import transfer_niftivalues_to_meshnodes_withITK_ECCOMAS # to load MRI data onto mesh nodes
from MRI_driven_parameters.FA_2_growthcoef import load_mesh_with_growthcoef_ECCOMAS_XDMF # to compute alphaTAN from FA nodal values
from MRI_driven_parameters.Segmentation_to_grTAN_for_cortical_thickness import load_mesh_with_grTAN_ECCOMAS # to get Cortical delineation from Segmentation label values


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='braingrowthFEniCS: halfbrain growth quasistatic 3D model + MRI-based parameters')

    parser.add_argument('-i', '--input', help='Input mesh path (.xml, .xdmf)', type=str, required=False, 
                        default='./data/dHCP_raw/dhcpRight21GW_masked_20000faces_98000tets_refinedWidthCoef5.xdmf') 
    
    parser.add_argument('-fa', '--fractionalanisotropynifti', help='Path to the fetal dchp atlas FA niftis at 21GW (whole brain)', type=str, required=False, 
                        default='./data/dHCP_raw/fa-t21.00.nii.gz')
    
    parser.add_argument('-seg', '--segmentationnifti', help='Path to the fetal dchp atlas Segmentation niftis at 21GW (whole brain)', type=str, required=False, 
                        default='./data/dHCP_raw/tissue-t21.00_dhcp-19.nii.gz')
    
    parser.add_argument('-r', '--reorientation', help='Is reorientation of the input mesh required?', type=bool, required=False, default=False)
   
    parser.add_argument('-p', '--parameters', help='Simulation input parameters', type=json.loads, required=False, 
                        default={#"H0": 2.25e-3, # [m] 
                                 "muCortex": 300, "muCore": 100, # [Pa]
                                 "nu": 0.45,
                                 "FA_to_alphaTAN_coef": 10.0e-6, # FA_to_alphaTAN_coef: [s⁻¹]
                                 "alphaRAD": 0.0, "grRAD": 1.0, # alphaRAD: [s⁻¹]
                                 "T0_in_GW": 21.0, "Tmax_in_GW": 29.0, "dt_in_seconds": 3600, # 0.5GW (1GW=168h=604800s)
                                 "linearization_method":"newton", "linear_solver":"gmres", "preconditioner":"sor",
                                 "newton_absolute_tolerance":1E-3, "newton_relative_tolerance":1E-4, 
                                 "krylov_absolute_tolerance":1E-4, "krylov_relative_tolerance":1E-5}) 
    
    parser.add_argument('-o', '--output', help='Output folder path', type=str, required=False, 
                        default='./results/halfbrain_notReoriented_Fgt_quasistatic_alphaTANFAbased_H0segmentation/')
               
    #parser.add_argument('-v', '--visualization', help='Visualization during simulation', type=bool, required=False, default=False)
    parser.add_argument('-v', '--visualization', help='Visualization during simulation', action='store_true')
    
    args = parser.parse_args() 
    
    #####################################################
    ###################### Parameters ###################
    #####################################################
    
    # Geometry 
    ##########
    """H0 = args.parameters["H0"]
    cortical_thickness = fenics.Expression('H0 + 0.01*t', H0=H0, t=0.0, degree=0)"""
    gdim=3
    
    # Mesh
    # ----
    # mesh & boundary mesh
    print("\nimporting mesh...")
    
    inputmesh_path = args.input
    inputmesh_format = inputmesh_path.split('.')[-1]

    if inputmesh_format == "xml":
        mesh0 = fenics.Mesh(inputmesh_path) # mesh0 (for projection of nifti values)
        mesh = fenics.Mesh(inputmesh_path)

    elif inputmesh_format == "xdmf":
        mesh0 = fenics.Mesh() # mesh0 (for projection of nifti values)
        with fenics.XDMFFile(inputmesh_path) as infile:
            infile.read(mesh0)
            
        mesh = fenics.Mesh()
        with fenics.XDMFFile(inputmesh_path) as infile:
            infile.read(mesh)

    # mesh lengths in meters (SI)
    print("\nconvert half brain Right mesh into meters (SI)...")
    mesh = preprocessing.converting_mesh_from_millimeters_into_meters(mesh)

    print("\ncomputing boundary mesh...")
    bmesh = fenics.BoundaryMesh(mesh, "exterior") # bmesh at t=0.0 (cortex envelop)

    if args.visualization == True:
        fenics.plot(mesh) 
        plt.title("input mesh")
        plt.show() 

    # reorient gifti mesh
    #preprocessing.reorient_gifti_mesh('./data/brain/dHCP_21GW/fetal.week21.right.pial.surf.gii')

    # mesh characteristics
    characteristics = preprocessing.compute_geometrical_characteristics(mesh, bmesh)
    center_of_gravity = preprocessing.compute_center_of_gravity(characteristics) 
    min_mesh_spacing, average_mesh_spacing, max_mesh_spacing = preprocessing.compute_mesh_spacing(mesh)
    print('input mesh characteristics: {}'.format(characteristics))
    print('input mesh COG = [xG0:{}, yG0:{}, zG0:{}]'.format(center_of_gravity[0], center_of_gravity[1], center_of_gravity[2]))
    print("input mesh min mesh spacing: {:.3f} mm".format(min_mesh_spacing))
    print("input mesh mean mesh spacing: {:.3f} mm".format(average_mesh_spacing))
    print("input mesh max mesh spacing: {:.3f} mm".format(max_mesh_spacing)) 

    # normalization and boundary mesh update, reoriented mesh characteristics
    if args.reorientation == True: # args.normalization
        print("\nreorienting mesh + placing COG at origin...")
        mesh = preprocessing.reorient_mesh(mesh, characteristics, center_of_gravity) # since netgen change x<>y axis compared to .stl orientation
        
        print("\nplacing interhemisphere at z=0.0...")
        COG_Z = 0.5*(abs(np.min(mesh.coordinates()[:,2])) + abs(np.max(mesh.coordinates()[:,2]))) 
        mesh.coordinates()[:,2] = mesh.coordinates()[:,2] - COG_Z # translate mesh, so that COG be at z=0.0
        
        #print("\ncomputing BLL vector...")
        #mesh_BLL_vector = preprocessing.compute_mesh_BLL_vector(mesh)

        #print("\nreorienting mesh making the BLL vector the Y direction...")
        #mesh = preprocessing.register_mesh_towards_Y_axis(mesh, mesh_BLL_vector)

        print("\ncomputing boundary mesh...")
        bmesh = fenics.BoundaryMesh(mesh, "exterior") # update bmesh

        if args.visualization == True:
            fenics.plot(mesh) 
            plt.title("reoriented mesh")
            plt.show()  
            #vedo.dolfin.plot(mesh, wireframe=False, text='reoriented mesh', style='paraview', axes=4).close()
        
        characteristics = preprocessing.compute_geometrical_characteristics(mesh, bmesh) # n_nodes, coordinates, n_tets, n_faces_Surface, n_faces_Volume, minx, maxx, miny, maxy, minz, maxz 
        center_of_gravity = preprocessing.compute_center_of_gravity(characteristics) 
        min_mesh_spacing, average_mesh_spacing, max_mesh_spacing = preprocessing.compute_mesh_spacing(mesh)
        print('reoriented mesh characteristics: {}'.format(characteristics))
        print('reoriented mesh COG = [xG:{}, yG:{}, zG:{}]'.format(center_of_gravity[0], center_of_gravity[1], center_of_gravity[2]))
        print("reoriented min_mesh_spacing: {:.3f} m".format(min_mesh_spacing))
        print("reoriented mesh mean mesh spacing: {:.3f} m".format(average_mesh_spacing))
        print("reoriented mesh max mesh spacing: {:.3f} m".format(max_mesh_spacing))

    # Export the characteristics of mesh_TO 
    # -------------------------------------
    #fenics.File(args.output + "mesh_T0.xml") << mesh

    with fenics.XDMFFile(MPI.COMM_WORLD, os.path.join(args.output, "mesh_T0.xdmf")) as xdmf:
        xdmf.write(mesh)

    """
            
    convert_meshformats.xml_to_vtk(args.output + "mesh_T0.xml", args.output + "mesh_T0.vtk")
    export_simulation_outputmesh_data.export_resultmesh_data(args.output + "analytics/",
                                                             args.output + "mesh_T0.vtk",
                                                             args.parameters["T0"],
                                                             0,
                                                             0.0,
                                                             "mesh_T0.txt")
    """
    
    # Elastic parameters
    ####################
    muCortex = fenics.Constant(args.parameters["muCortex"])
    muCore = fenics.Constant(args.parameters["muCore"])
    
    nu = fenics.Constant(args.parameters["nu"])
    
    KCortex = fenics.Constant( 2*muCortex.values()[0] * (1 + nu.values()[0]) / (3*(1 - 2*nu.values()[0])) ) # 3D
    KCore = fenics.Constant( 2*muCore.values()[0] * (1 + nu.values()[0]) / (3*(1 - 2*nu.values()[0])) ) # 3D

    # Growth parameters
    ###################
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
    
    ##################################################
    ###################### Problem ###################
    ##################################################
    
    # Boundaries
    ############
    #print("\ncomputing and marking boundaries...")
    #bmesh_cortexsurface_bbtree = fenics.BoundingBoxTree()
    #bmesh_cortexsurface_bbtree.build(bmesh) 

    # initialize boundaries --> See: https://fenicsproject.discourse.group/t/how-to-mark-boundaries-to-a-circle-and-change-solution-from-cartesian-to-polar/5492/4
    # N.B. 'boundaries_volume' is used to defined Dirichlet BCs & 'boundaries_surface' is used to define cortical surface submesh
    boundaries_volume = fenics.MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # size: facets --> mesh.num_facets() => number of faces in the volume
    boundaries_volume.set_all(100)
    
    boundaries_surface = fenics.MeshFunction('size_t', bmesh, bmesh.topology().dim(), 0) # size: n_faces (at surface) => number of faces at the surface
    boundaries_surface.set_all(100)


    # define part of the boundary to fix (Dirichlet BCs) 
    """
    class InterHemisphereContactZone(fenics.SubDomain): # https://fenicsproject.discourse.group/t/on-boundary-clarification/3342/3
        def inside(self, x, on_boundary): 
            tol = 1E-14
            interHemisphere_zone_Xmin, interHemisphere_zone_Xmax = -0.0191863, 0.0369359
            interHemisphere_zone_Zmin, interHemisphere_zone_Zmax = -0.0200173, 0.0238999
            interHemisphere_zone_Ymin, interHemisphere_zone_Ymax = characteristics['miny'], 0.00101497

            interHemisphere_zone_Center_COORDS_XYZ = 0.5*(interHemisphere_zone_Xmax + interHemisphere_zone_Xmin), 0.5*(interHemisphere_zone_Ymax + interHemisphere_zone_Ymin), 0.5*(interHemisphere_zone_Zmax + interHemisphere_zone_Zmin)
            aX, bZ = 0.5*(interHemisphere_zone_Xmax - interHemisphere_zone_Xmin), 0.5*(interHemisphere_zone_Zmax - interHemisphere_zone_Zmin)

            return ((x[0]-interHemisphere_zone_Center_COORDS_XYZ[0])**2/aX**2 + (x[2]-interHemisphere_zone_Center_COORDS_XYZ[2])**2/bZ**2 <= 1.25) and x[1] < interHemisphere_zone_Ymax and x[1] > interHemisphere_zone_Ymin
    """
    
    class InterHemisphereZone(fenics.SubDomain): # https://fenicsproject.discourse.group/t/on-boundary-clarification/3342/3
        def inside(self, x, on_boundary): 
            tol = 1E-14
            interHemisphere_zone_Xmin, interHemisphere_zone_Xmax = -0.0282783, 0.0369359 # values measures with Paraview on the dHCP halfmesh Right 21GW built from original .gii
            interHemisphere_zone_Zmin, interHemisphere_zone_Zmax = -0.0200173, 0.0238999
            interHemisphere_zone_Ymin, interHemisphere_zone_Ymax = characteristics['miny'], 0.008 #0.0150209

            interHemisphere_zone_Center_COORDS_XYZ = 0.5*(interHemisphere_zone_Xmax + interHemisphere_zone_Xmin), 0.5*(interHemisphere_zone_Ymax + interHemisphere_zone_Ymin), 0.5*(interHemisphere_zone_Zmax + interHemisphere_zone_Zmin)
            aX, bZ, cY = 0.5*(interHemisphere_zone_Xmax - interHemisphere_zone_Xmin), 0.5*(interHemisphere_zone_Zmax - interHemisphere_zone_Zmin), 0.5*(interHemisphere_zone_Ymax - interHemisphere_zone_Ymin)

            return ((x[0]-interHemisphere_zone_Center_COORDS_XYZ[0])**2/aX**2 + (x[2]-interHemisphere_zone_Center_COORDS_XYZ[2])**2/bZ**2 + (x[1]-interHemisphere_zone_Center_COORDS_XYZ[1])**2/cY**2 <= 1.35)

    interHemisphereZone = InterHemisphereZone()
    interHemisphereZone.mark(boundaries_surface, 102)
    interHemisphereZone.mark(boundaries_volume, 102) 

    # export marked boundaries
    # ------------------------
    #export_XML_PVD_XDMF.export_PVDfile(args.output, 'boundaries_surface_T0', boundaries_surface) # export at t=0.0 by default
    export_XML_PVD_XDMF.export_PVDfile(args.output, 'boundaries_volume_T0', boundaries_volume) 

    # Build cortical surface (boundary) submesh
    # -----------------------------------------
    bmesh_cortex = fenics.SubMesh(bmesh, boundaries_surface, 100) # part of the boundary mesh standing for the cortical surface
    with fenics.XDMFFile(MPI.COMM_WORLD, args.output + "cortical_surface_mesh.xdmf") as xdmf:
        xdmf.write(bmesh_cortex)

    print("\ncomputing and marking boundaries...")
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

    # FEM Functions
    ###############
    
    # Scalar functions of V
    #H = fenics.Function(S, name="H") 
    #d2s = fenics.Function(S, name="d2s")
    grGrowthZones = fenics.Function(S, name="grGrowthZones")
    #gm = fenics.Function(S, name="gm") 
    mu = fenics.Function(S, name="mu") 
    K = fenics.Function(S, name="K") 

    fa = fenics.Function(S, name="FA")
    normalized_FA = fenics.Function(S, name="NormalizedFA")
    alphaTAN = fenics.Function(S, name="alphaTAN")
    
    grTAN = fenics.Function(S, name="grTANCortexDelineation")
    
    dg_TAN = fenics.Function(S, name="dgTAN")
    dg_RAD = fenics.Function(S, name="dgRAD") 

    # Vector functions of V
    u = fenics.Function(V, name="Displacement") # Trial function. Current (unknown) displacement
    du = fenics.TrialFunction(V)
    v_test = fenics.TestFunction(V) # Test function

    BoundaryMesh_Nt = fenics.Function(V, name="BoundaryMesh_Nt")
    Mesh_Nt = fenics.Function(V, name="Mesh_Nt")

    # Vector functions of Vtensor
    Fg_T = fenics.Function(Vtensor, name="Fg")    
    PK1tot_T = fenics.Function(Vtensor, name="PK1tot") 
    
    # Mappings
    ##########
    print("\ncomputing mappings...")
    # From vertex to DOF in the whole mesh --> used to compute Mesh_Nt
    # ----------------------------------------------------------------
    vertex2dofs_V = mappings.vertex_to_dofs_VectorFunctionSpace(V, gdim)
    vertex2dofs_B = mappings.vertex_to_dofs_VectorFunctionSpace(V_cortexsurface, gdim)

    # From the surface mesh (cortex envelop) to the whole mesh (B100_2_V_dofmap; vertexB100_2_dofsV_mapping --> used to compute Mesh_Nt)
    # --------------------------------------------------------
    B100_2_V_dofmap, vertexB100_2_dofsV_mapping = mappings.surface_to_mesh_V(gdim, V, V_cortexsurface, vertex2dofs_B)
    Sboundary100_2_S_dofmap, vertexBoundary100Mesh_2_dofScalarFunctionSpaceWholeMesh_mapping = mappings.surface_to_mesh_S(S, S_cortexsurface)
    
    # From the whole mesh to the surface mesh (to be use for projections onto surface in contact process)
    # ---------------------------------------
    vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping_t0 = mappings.mesh_to_surface_V(mesh.coordinates(), bmesh.coordinates()) # at t=0. (keep reference projection before contact for proximity node to deep tetrahedra vertices)
    
    # Residual Form
    ###############

    # prerequisites before computing Fg and mu (H, d2s and gm=f(d2s, H) required)
    # ----------------------------------------
    print("\ninitializing distances to surface...")
    vertex2dofs_S = mappings.vertex_to_dof_ScalarFunctionSpace(S)
    """d2s_ = differential_layers.compute_distance_to_cortexsurface(vertex2dofs_S, d2s, mesh, bmesh_cortexsurface_bbtree) # init at t=0.0"""
    #d2s_ = differential_layers.compute_distance_to_cortexsurface_2(vertex2dofs_S, d2s, mesh, bmesh_cortexsurface_cKDtree) # init at t=0.0
    """projection.local_project(d2s_, S, d2s)"""

    print("\ninitializing differential term function...")
    """projection.local_project(cortical_thickness, S, H)""" # H.assign( fenics.project(h, S) )  
    #gm_ = differential_layers.compute_differential_term_DOF(S, d2s, H, gm) # init 'gm' at t=0.0
    #projection.local_project(gm_, S, gm)
    """for dof in S.dofmap().dofs():
        d2s_dof = d2s.vector()[dof]
        gm.vector()[dof] = compute_differential_term(d2s_dof, H.vector()[dof]) """

    # brain regions growth mask
    # -------------------------  
    print("mark no-growth brain regions (e.g. 'longitudinal fissure' - 'ventricules' - 'mammilary bodies')...") # code source: T.Tallinen et al. 2016. See detail in https://github.com/rousseau/BrainGrowth/blob/master/geometry.py   
    # initial value from T.Tallinen for delimating growth zones from inner zones: 0.6
    pond_Y = (abs(np.min(mesh.coordinates()[:,1])) + abs(np.max(mesh.coordinates()[:,1])))/2
    pond_X = (abs(np.min(mesh.coordinates()[:,0])) + abs(np.max(mesh.coordinates()[:,0])))/2
    new_critical_value = (pond_Y - 2.25e-3) # 0.6 * pond_Y
    #new_critical_value = pond_Y - 10*H0

    for vertex, scalarDOF in enumerate(vertex2dofs_S):

        dX = characteristics['maxx'] - characteristics['minx'] # 2*a
        dY = characteristics['maxy'] - characteristics['miny'] # 2*b
        dZ_whole_brain = 2 * (characteristics['maxz'] - characteristics['minz']) # 2*c
        #a, b, c = 0.5 * dX, 0.5 * dY, 0.5 * dZ_whole_brain
        a, b, c = 0.7 * 0.5 * dX, 0.7 * 0.5 * dY,  0.7 * 0.5 * dZ_whole_brain 
        
        rqp = np.linalg.norm( np.array([    a/b * (mesh.coordinates()[vertex, 0] - center_of_gravity[0]) + 0.75*pond_X, 
                                                  (mesh.coordinates()[vertex, 1] - center_of_gravity[1]), 
                                            c/b * (mesh.coordinates()[vertex, 2] - center_of_gravity[2]) ])) 
        

         
        """
        if rqp < new_critical_value:
            grGrowthZones.vector()[scalarDOF] = max(1.0 - 10.0*(new_critical_value - rqp), 0.0)
        else:
            grGrowthZones.vector()[scalarDOF] = 1.0
        """
            
        if mesh.coordinates()[vertex, 0] > center_of_gravity[0] + 0.5 * (0.5 * dX) :
            grGrowthZones.vector()[scalarDOF] = 1.0
        else:
            grGrowthZones.vector()[scalarDOF] = 0.0
    
    # Fg
    # --
    print("\ninitializing growth coefficients: dgTAN & dgRAD...")
    
    # 0. 
    ####
    # Load FA and Segmentation from MRI nifti at T0 (e.g. 21GW) to mesh nodes
    FA_nifti_file_path_T0 = args.fractionalanisotropynifti 
    Segmentation_nifti_file_path_T0 = args.segmentationnifti 

    # transfer Labels and FA values onto initial mesh
    mesh_meshio_FA = transfer_niftivalues_to_meshnodes_withITK_ECCOMAS.load_FA_onto_mesh_nodes(mesh0, 
                                                                                               FA_nifti_file_path_T0, 
                                                                                               interpolation_mode='linear') # interpolation_mode: 'nearest_neighbor'; 'linear'
    
    mesh_meshio_FA.write(args.output + "mesh_FA_values_{}GW".format(int(T0_in_GW)) + '.vtk') 
    
    # nead to revert again mesh0 coords before applyin 2nd projection (of parcellation)  
    mesh_coordinates = mesh0.coordinates() # list of nodes coordinates
    X = mesh_coordinates[:,0].copy()
    Y = mesh_coordinates[:,1].copy()
    mesh_coordinates[:,0] = Y
    mesh_coordinates[:,1] = X

    mesh_meshio_Labels = transfer_niftivalues_to_meshnodes_withITK_ECCOMAS.load_Segmentation_onto_mesh_nodes(mesh0, 
                                                                                                             Segmentation_nifti_file_path_T0, 
                                                                                                             interpolation_mode='nearest_neighbor') # interpolation_mode: 'nearest_neighbor'; 'linear'
    
    mesh_meshio_Labels.write(args.output + "mesh_Segmentation_values_{}GW".format(int(T0_in_GW)) + '.vtk') 
    
    # 1. get nodal label of Cortex or Core belonging from dHCP (whole brain) Segmentation nifti 
    #### 
    ####
    grTAN.assign( load_mesh_with_grTAN_ECCOMAS.bilayer_tangential_growth_ponderation_from_SegmentationLabels(grTAN, 
                                                                                                             vertex2dofs_S, 
                                                                                                             mesh_meshio_Labels) )
    FEniCS_FEM_Functions_file.write(grTAN, T0_in_GW)

    # Density of FA in the Cortex volume "normalized" to 1
    """
    class Cortex(fenics.SubDomain):
        def inside(self, x, on_boundary):
            for vertex, coords in enumerate(mesh.coordinates()):
                if grTAN.vector()[vertex2dofs_S[]] == 1.:
                    return x
    """

    """
    d2v_S = fenics.dof_to_vertex_map(S)
    vertices_Cortex = d2v_S[grTAN.vector()[:] == 1.0] # indexation in the whole mesh

    subdomains.array()[vertices_Cortex[:]] = 1
    """

    ###
    """
    class MyDict(dict): # https://fenicsproject.org/qa/5268/is-that-possible-to-identify-a-facet-by-its-vertices/
        def get(self, key):
            return dict.get(self, sorted(key))

    tet_2_v = MyDict((tet.index(), tuple(tet.entities(0))) for tet in fenics.cells(mesh))
    
    for tet in fenics.cells(mesh):
        vertex1, vertex2, vertex3, vertex4 = tet_2_v[tet.index()]
        if grTAN.vector()[vertex2dofs_S[vertex1]] == 1. and \
           grTAN.vector()[vertex2dofs_S[vertex2]] == 1. and \
           grTAN.vector()[vertex2dofs_S[vertex3]] == 1. and \
           grTAN.vector()[vertex2dofs_S[vertex4]] == 1. :
           subdomains.array()[tet.index()] = 1
    """
    
    ###

    # export marked boundaries and subdomains
    # ---------------------------------------
    export_XML_PVD_XDMF.export_PVDfile(args.output, 'boundaries_volume_T0', boundaries_volume)
    #export_XML_PVD_XDMF.export_PVDfile(args.output, 'subdomains_T0_grTAN', subdomains) 

    # Measurement entities 
    # --------------------
    ds = fenics.Measure("ds", domain=mesh, subdomain_data=boundaries_volume) 
    #dx = fenics.Measure("dx", domain=mesh, subdomain_data=subdomains) # Volume measurement entities 

    ###
    ###

    # 2. Compute alphaTAN from  dHCP (whole brain) fractional anisotropy (FA) nifti at T0=21GW. 
    ####
    ####
    # a. Normalize FA (FA value should be between 0. and 1.)
    fa, normalized_FA = load_mesh_with_growthcoef_ECCOMAS_XDMF.normalize_FA(fa, normalized_FA, vertex2dofs_S, mesh_meshio_FA)
    FEniCS_FEM_Functions_file.write(normalized_FA, T0_in_GW)

    # b. find coef such that int(coef * fa)dx = int(1)dx <=> discrete integration on all mesh volume nodes : Sum(alphaTAN)_{n_nodes} = coef_lissage_alphaTAN_FA * Sum(alphaTAN_FA)_{n_nodes} for whole growth to be equal in numerical versus FA-informed cases
    """
    fa_with_density_normalize_to_one_in_Cortex_volume = fenics.Function(S, name="FA_such_that_density_in_Cortex_equal_to_one")
    coef = fenics.Function(S, name="coef_for_FA_density_in_Cortex_be_one") # unkown

    dv = fenics.TrialFunction(S) 
    v_ = fenics.TestFunction(S)

    volume_Cortex = fenics.assemble(fenics.Constant(1) * dx(1))
    
    # find dv (=coef) such that int(dv * fa)dx = int(1)dx
    a_proj = (fenics.inner(fenics.dot(dv, fa), v_)) * dx #* dx(1) # integration on the Cortex layer volume / dv --> unknown, namely "coef"
    b_proj = (fenics.inner(fenics.Constant(1), v_)) * dx # * dx(1) 
    solver = fenics.LocalSolver(a_proj, b_proj)
    solver.factorize()
    solver.solve_local_rhs(coef)

    #A = fenics.assemble(fenics.dot(coef, fa) * dx(1))
    #b = fenics.assemble(1)
    #solver.solve(A, coef, b)

    fa_with_density_normalize_to_one_in_Cortex_volume.vector()[:] = fa.vector()[:] * coef.vector()[:] 
    FEniCS_FEM_Functions_file.write(coef, T0_in_GW)
    FEniCS_FEM_Functions_file.write(fa_with_density_normalize_to_one_in_Cortex_volume, T0_in_GW)
    """

    # b. alphaTAN: Compute tangential growth coefficient from FA values
    FA_to_alphaTAN_coef = args.parameters["FA_to_alphaTAN_coef"]
    alphaTAN_normalizedFA = fenics.Function(S, name="alphaTAN_normalizedFA_based")
    alphaTAN_normalizedFA.assign( load_mesh_with_growthcoef_ECCOMAS_XDMF.tangential_growth_coef_from_FA(FA_to_alphaTAN_coef, 
                                                                                           alphaTAN_normalizedFA, 
                                                                                           vertex2dofs_S, 
                                                                                           normalized_FA) )# normalized_FA; fa_with_density_normalize_to_one_in_Cortex_volume
    FEniCS_FEM_Functions_file.write(alphaTAN_normalizedFA, T0_in_GW)

    # c. find coef such that int(coef * fa)dx = int(1)dx <=> discrete integration on all mesh volume nodes : Sum(alphaTAN)_{n_nodes_in_Cortex} = coef_lissage_alphaTAN_FA * Sum(alphaTAN_FA)_{n_nodes_in_Cortex} for whole growth to be equal in numerical versus FA-informed cases
    list_of_scalarDOF_in_Cortex = np.where(grTAN.vector()[:] != 0)
    print("list coef".format(list_of_scalarDOF_in_Cortex))
    list_alphaTAN_homogeneous = [] # Sum(alphaTAN)_{n_nodes_in_Cortex}
    list_alphaTAN_normalizedFAbased = [] # Sum(alphaTAN_FA)_{n_nodes_in_Cortex}
    for scalarDOF in list_of_scalarDOF_in_Cortex:
        list_alphaTAN_homogeneous.append(FA_to_alphaTAN_coef * grTAN.vector()[scalarDOF])
        list_alphaTAN_normalizedFAbased.append(alphaTAN_normalizedFA.vector()[scalarDOF])
    coef_lissage_alphaTAN_FA = np.sum(list_alphaTAN_homogeneous) / np.sum(list_alphaTAN_normalizedFAbased)
    
    #coef_lissage_alphaTAN_FA = mesh.num_vertices() * FA_to_alphaTAN_coef / np.sum(alphaTAN_normalizedFA.vector()[:])
    
    # d. deduce FA-based alphaTAN, that preserves the density of growth in the Cortex, compared to the homogeneous case.
    alphaTAN.assign(coef_lissage_alphaTAN_FA * alphaTAN_normalizedFA)
    FEniCS_FEM_Functions_file.write(alphaTAN, T0_in_GW)
    
    # e. dgTAN and dgRAD
    # projection.local_project(grTAN * grGrowthZones * alphaTAN * dt_in_seconds, S, dg_TAN) 
    projection.local_project(grTAN * alphaTAN * dt_in_seconds, S, dg_TAN)
    projection.local_project(grRAD * alphaRAD * dt_in_seconds, S, dg_RAD) 

    print("\ninitializing normals to boundary...")
    #BoundaryMesh_Nt.assign( growth.compute_topboundary_normals(mesh, ds, V) )
    boundary_normals = growth.compute_topboundary_normals(mesh, ds(100), V) 
    projection.local_project(boundary_normals, V, BoundaryMesh_Nt)

    print("\ninitializing projected normals of nodes of the whole mesh...")
    #Mesh_Nt.assign( growth.compute_mesh_projected_normals(V, mesh.coordinates(), bmesh.coordinates(), vertexB100_2_dofsV_mapping, vertex2dofs_V, BoundaryMesh_Nt) )
    mesh_normals = growth.compute_mesh_projected_normals(V, mesh.coordinates(), bmesh_cortex.coordinates(), vertexB100_2_dofsV_mapping, vertex2dofs_V, BoundaryMesh_Nt) 
    projection.local_project(mesh_normals, V, Mesh_Nt)

    print("\ninitializing growth tensor...")
    #helpers.local_project( compute_growth_tensor(Mesh_Nt, dg_TAN, dg_RAD), Vtensor, Fg) # init at t=0.0 (local_project equivalent to .assign())"""
    Fg = growth.compute_growth_tensor(Mesh_Nt, dg_TAN, dg_RAD, gdim)
    projection.local_project(Fg, Vtensor, Fg_T) # projection of Fg onto Vtensor Function Space

    # mucontact
    # --
    print("\ninitializing local stiffness...")   
    mu_ = differential_layers.compute_shear_and_bulk_stiffnesses(grTAN, muCore, muCortex)
    projection.local_project(mu_, S, mu)
    
    #mu_ = fenics.project(mu_, S)
    #mu.assign( mu_ )
    K_ = differential_layers.compute_shear_and_bulk_stiffnesses(grTAN, KCore, KCortex)
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

    # Fe: elastic part of the deformation gradient
    Fg_inv = fenics.variable( fenics.inv(Fg) )
    Fe = fenics.variable( F * Fg_inv )# F_t * (F_g)⁻¹

    # Cauchy-Green tensors (elastic part of the deformation only)
    Ce = fenics.variable( Fe.T * Fe )
    Be = fenics.variable( Fe * Fe.T )

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

    bc_Dirichlet = fenics.DirichletBC(V, fenics.Constant((0., 0., 0.)), boundaries_volume, 102) # no displacement in x,y,z --> fixed zone to avoid additional solution including Rotations & Translations
    bcs = [bc_Dirichlet]
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
    nonlinearvariationalsolver.parameters["nonlinear_solver"] = args.parameters["linearization_method"] # newton
    #nonlinearvariationalsolver.parameters['newton_solver']['convergence_criterion'] = "incremental" 
    nonlinearvariationalsolver.parameters['newton_solver']['absolute_tolerance'] = args.parameters["newton_absolute_tolerance"] #1E-3 # 1E-7 # 1E-10 for unknown (displacement) in mm
    nonlinearvariationalsolver.parameters['newton_solver']['relative_tolerance'] = args.parameters["newton_relative_tolerance"]  # 1E-8 # 1E-11 for unknown (displacement) in mm
    nonlinearvariationalsolver.parameters['newton_solver']['maximum_iterations'] = 25 # 50 (25)
    nonlinearvariationalsolver.parameters['newton_solver']['relaxation_parameter'] = 1.0 # means "full" Newton-Raphson iteration expression: u_k+1 = u_k - res(u_k)/res'(u_k) => u_k+1 = u_k - res(u_k)/jacobian(u_k)

    # CHOOSE AND PARAMETRIZE THE LINEAR SOLVER IN EACH NEWTON ITERATION (LINEARIZED PROBLEM) 
    nonlinearvariationalsolver.parameters['newton_solver']['linear_solver'] = args.parameters["linear_solver"] # linearized problem: AU=B --> Choose between direct method U=A⁻¹B O(N³) (e.g. 'mumps') or iterative/Krylov subspaces method U=A⁻¹B~(b + Ab + A²b + ...) O(num_iter * N²) (e.g. 'gmres' for non-symmetric problem , 'cg') to compute A⁻¹. 
    nonlinearvariationalsolver.parameters['newton_solver']['preconditioner'] = args.parameters["preconditioner"]

    nonlinearvariationalsolver.parameters['newton_solver']['krylov_solver']['absolute_tolerance'] = args.parameters["krylov_absolute_tolerance"] #1E-4 #1E-9
    nonlinearvariationalsolver.parameters['newton_solver']['krylov_solver']['relative_tolerance'] = args.parameters["krylov_relative_tolerance"] #1E-5 #1E-7
    nonlinearvariationalsolver.parameters['newton_solver']['krylov_solver']['maximum_iterations'] = 1000 # number of iterations with Krylov subspace method
    
    # Reusing previous unknown u_n as the initial guess to solve the next iteration n+1 
    #nonlinearvariationalsolver.parameters['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True # https://link.springer.com/content/pdf/10.1007/978-3-319-52462-7_5.pdf --> "Using a nonzero initial guess can be particularly important for timedependent problems or when solving a linear system as part of a nonlinear iteration, since then the previous solution vector U will often be a good initial guess for the solution in the next time step or iteration."
    # parameters['krylov_solver']['monitor_convergence'] = True # https://fenicsproject.org/qa/1124/is-there-a-way-to-set-the-inital-guess-in-the-krylov-solver/
    
    ########################################################
    ###################### Simulation ######################
    ########################################################
    
    times = np.linspace(T0_in_seconds, Tmax_in_seconds, int(Nsteps+1))  # in seconds!
    
    # Export FEM function at T0_in_GW
    #FEniCS_FEM_Functions_file.write(d2s, T0_in_GW)
    #FEniCS_FEM_Functions_file.write(H, T0_in_GW)
    #FEniCS_FEM_Functions_file.write(gm, T0_in_GW)
    #FEniCS_FEM_Functions_file.write(grGrowthZones, T0_in_GW)

    FEniCS_FEM_Functions_file.write(BoundaryMesh_Nt, T0_in_GW)
    FEniCS_FEM_Functions_file.write(Mesh_Nt, T0_in_GW)

    #FEniCS_FEM_Functions_file.write(normalized_FA, T0_in_GW)
    #FEniCS_FEM_Functions_file.write(alphaTAN, T0_in_GW)
    #FEniCS_FEM_Functions_file.write(grTAN, T0_in_GW)
    FEniCS_FEM_Functions_file.write(dg_TAN, T0_in_GW)
    FEniCS_FEM_Functions_file.write(dg_RAD, T0_in_GW)
    FEniCS_FEM_Functions_file.write(Fg_T, T0_in_GW)

    FEniCS_FEM_Functions_file.write(mu, T0_in_GW)
    FEniCS_FEM_Functions_file.write(K, T0_in_GW)

    """start_time = time.time ()"""
    energies = np.zeros((int(Nsteps+1), 4))
    E_damp = 0
    E_ext = 0
    
    for i, dt in enumerate( tqdm( np.diff(times), desc='brain is growing...', leave=True) ): # dt = dt_in_seconds

        fenics.set_log_level(fenics.LogLevel.ERROR) # in order not to print solver info logs 

        """
        t_in_GW = times[i+1]
        t = t_in_GW * 604800 # in seconds
        """
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
        #d2s.assign( differential_layers.compute_distance_to_cortexsurface(vertex2dofs_S, d2s, mesh, bmesh_cortexsurface_bbtree) )
        """
        d2s_ = differential_layers.compute_distance_to_cortexsurface(vertex2dofs_S, d2s, mesh, bmesh_cortexsurface_bbtree) 
        #d2s_ = differential_layers.compute_distance_to_cortexsurface_2(vertex2dofs_S, d2s, mesh, bmesh_cortexsurface_cKDtree) 
        projection.local_project(d2s_, S, d2s)
        """
        
        # gm
        #gm = differential_layers.compute_differential_term_DOF(S, d2s, H, gm)
        #gm_ = differential_layers.compute_differential_term_DOF(S, d2s, H, gm) # init 'gm' at t=0.0
        #projection.local_project(gm_, S, gm)

        # Update differential material stiffness mu 
        # -----------------------------------------
        # mu have to be updated at each timestep (material properties evolution with deformation) (So do previously H, d2s, gm)
        mu_ = differential_layers.compute_shear_and_bulk_stiffnesses(grTAN, muCore, muCortex)
        projection.local_project(mu_, S, mu)
        
        K_ = differential_layers.compute_shear_and_bulk_stiffnesses(grTAN, KCore, KCortex)
        projection.local_project(K_, S, K)

        # Update growth tensor coefficients
        # ---------------------------------
        #projection.local_project(grTAN * grGrowthZones * alphaTAN * dt, S, dg_TAN)
        projection.local_project(grTAN * alphaTAN * dt, S, dg_TAN)
        projection.local_project(grRAD * alphaRAD * dt, S, dg_RAD) 

        # Update growth tensor orientation (adaptative)
        # ---------------------------------------------
        boundary_normals = growth.compute_topboundary_normals(mesh, ds(100), V) 
        projection.local_project(boundary_normals, V, BoundaryMesh_Nt)
        
        mesh_normals = growth.compute_mesh_projected_normals(V, mesh.coordinates(), bmesh_cortex.coordinates(), vertexB100_2_dofsV_mapping, vertex2dofs_V, BoundaryMesh_Nt) 
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
        FEniCS_FEM_Functions_file.write(u, t_in_GW)
        
        projection.local_project(PK1tot, Vtensor, PK1tot_T) # export Piola-Kirchhoff stress
        FEniCS_FEM_Functions_file.write(PK1tot_T, t_in_GW)
                
        #FEniCS_FEM_Functions_file.write(d2s, t_in_GW)
        #FEniCS_FEM_Functions_file.write(H, t_in_GW)
        #FEniCS_FEM_Functions_file.write(gm, t_in_GW)

        FEniCS_FEM_Functions_file.write(BoundaryMesh_Nt, t_in_GW) 
        FEniCS_FEM_Functions_file.write(Mesh_Nt, t_in_GW) 

        FEniCS_FEM_Functions_file.write(dg_TAN, t_in_GW)
        FEniCS_FEM_Functions_file.write(dg_RAD, t_in_GW)
        FEniCS_FEM_Functions_file.write(Fg_T, t_in_GW)

        FEniCS_FEM_Functions_file.write(mu, t_in_GW)
        FEniCS_FEM_Functions_file.write(K, t_in_GW)

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
        
        #bmesh_cortexsurface_cKDtree = cKDTree(bmesh_sub_10.coordinates())
        
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






