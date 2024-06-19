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
import sys, os

sys.path.append(sys.path[0])
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from FEM_biomechanical_model import contact_penalty, preprocessing, numerical_scheme_temporal, numerical_scheme_spatial, mappings, differential_layers, growth, projection
from utils.export_functions import export_simulation_outputmesh_data, export_simulation_end_time_and_iterations, export_XML_PVD_XDMF
from utils.converters import convert_meshformats


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='braingrowthFEniCS: brain growth elastodynamics 3D model')

    parser.add_argument('-i', '--input', help='Input mesh path (xml)', type=str, required=False, 
                        default='./data/brainmesh.xdmf') 
                        
    parser.add_argument('-n', '--normalization', help='Is normalization of the input mesh required? (required by braingrowthFEniCS)', type=bool, required=False, default=True)
    
    parser.add_argument('-p', '--parameters', help='Simulation input parameters', type=json.loads, required=False, 
                        default={"H0": 0.07, # 0.04
                                 "K": 100.0, # 50
                                 "muCortex": 30.0, "muCore": 1.0, # 50, 1
                                 "rho": 1.0, 
                                 "damping_coef": 10., # 0.5
                                 "alphaTAN": 5.0, "alphaRAD": 0.0, "grTAN": 1.0, "grRAD": 1.0, # alphaTAN 7
                                 "penalty_coefficient": 80, 
                                 "alphaM": 0.2, "alphaF": 0.4, 
                                 "T0": 0.0, "Tmax": 1.0, "Nsteps": 100,
                                 "linearization_method":"newton", "linear_solver":"gmres", "preconditioner":"sor"}) 
    
    parser.add_argument('-o', '--output', help='Output folder path', type=str, required=False, 
                        default='./results_brainFgt_2Hemispheres_DISTANCE_PENALTY_25mars/')
               
    parser.add_argument('-v', '--visualization', help='Visualization during simulation', type=bool, required=False, default=False)
    #parser.add_argument('-v', '--visualization', help='Visualization during simulation', action='store_true')
    
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

    # Output file
    #############
    FEniCS_FEM_Functions_file = fenics.XDMFFile(args.output + "growth_simulation.xdmf")
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
    """
    fenics.File(args.output + "mesh_T0.xml") << mesh
    convert_meshformats.xml_to_vtk(args.output + "mesh_T0.xml", args.output + "mesh_T0.vtk")
    export_simulation_outputmesh_data.export_resultmesh_data(args.output + "analytics/",
                                                             args.output + "mesh_T0.vtk",
                                                             args.parameters["T0"],
                                                             0,
                                                             0.0,
                                                             "mesh_T0.txt")
    """

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

    # Subdomains
    # ----------
    subdomains = fenics.MeshFunction("size_t", mesh, mesh.topology().dim()) 
    subdomains.set_all(0)

    # 2 Hemispheres
    ###############

    # Labelize 2 hemispheres meshes and boundaries
    # --------------------------------------------

    # define and mark brainHemisphere1 and submesh_Hemisphere2 (add label to the whole mesh nodes/DOFs)
    class BrainHemisphere1(fenics.SubDomain):

        def __init__(self, inter_hemispheres_position_X_0): 
            fenics.SubDomain.__init__(self)
            self.inter_hemispheres_position_X_0 = inter_hemispheres_position_X_0

        def inside(self, x, on_boundary):
            return x[0] <= self.inter_hemispheres_position_X_0 # axis of the 2 hemispheres in normalized brain should be X

    class BrainHemisphere2(fenics.SubDomain):

        def __init__(self, inter_hemispheres_position_X_0): 
            fenics.SubDomain.__init__(self)
            self.inter_hemispheres_position_X_0 = inter_hemispheres_position_X_0

        def inside(self, x, on_boundary):
            return x[0] >= self.inter_hemispheres_position_X_0 # axis of the 2 hemispheres in normalized brain should be X
    
    # compute interhemisphere position
    inter_hemispheres_position_X_0 = 0.0 # since normalized brain mesh
    """interhemispheres_position_y_0 = 0.5 * (np.min(bmesh.coordinates()[:,1]) + np.max(bmesh.coordinates()[:,1]))"""

    """
    # if we consider "half" brain simulation
    if halforwholebrain == "half":
        if leftorrightlobe == "left": 
            interlobes_pos = np.max(bmesh.coordinates()[:,1]) # = characteristics['maxy'] for Left lobe
        elif leftorrightlobe == "right": 
            interlobes_pos = np.min(bmesh.coordinates()[:,1]) # = characteristics['miny'] for Right lobe 

    lobe_extremity_Y = characteristics['miny']
    lobe_extremity_Y = characteristics['maxy']
    
    halfbrain_width = abs(lobe_extremity_Y - interlobes_position)
    interlobes_boundary_width = 0.2 * halfbrain_width # e.g. 0.05 if halfbrain width = 0.5
    """

    BrainHemisphere1(inter_hemispheres_position_X_0) .mark(subdomains, 1) # mark subdomain "brain hemisphere 1" with 1
    BrainHemisphere2(inter_hemispheres_position_X_0).mark(subdomains, 2) 

    # define Hemisphere1 & Hemisphere2 meshes and boundary meshes (FEniCS mesh object)
    # ------------------------------------------------------------
    submesh_Hemisphere1 = fenics.SubMesh(mesh, subdomains, 1) # submesh_Hemisphere1 = fenics.MeshView.create(subdomains, 1) 
    submesh_Hemisphere2 = fenics.SubMesh(mesh, subdomains, 2)

    bmesh_Hemisphere1 = fenics.BoundaryMesh(submesh_Hemisphere1, "exterior") 
    bmesh_Hemisphere2 = fenics.BoundaryMesh(submesh_Hemisphere2, "exterior") 

    boundaries_H1 = fenics.MeshFunction("size_t", submesh_Hemisphere1, submesh_Hemisphere1.topology().dim() - 1) # --> full boundaries (!= from ds(102))for BoundaryMesh_Nt_H1 computation
    boundaries_H2 = fenics.MeshFunction("size_t", submesh_Hemisphere2, submesh_Hemisphere2.topology().dim() - 1)  

    # define and mark boundaries of the 2 hemispheres (add label to surface nodes/DOFs)
    # -----------------------------------------------
    class BrainHemisphere1Boundary(fenics.SubDomain):
            def __init__(self, submesh_Hemisphere1):
                fenics.SubDomain.__init__(self)
                self.bmesh_H1 = fenics.BoundaryMesh(submesh_Hemisphere1, "exterior") 
                self.bmesh_H1_bbtree = fenics.BoundingBoxTree() # rewrite bbtree for new bmesh_cortexsurface (partial)
                self.bmesh_H1_bbtree.build(self.bmesh_H1)

            def inside(self, x, on_boundary):
                _, distance = self.bmesh_H1_bbtree.compute_closest_entity(fenics.Point(*x)) # compute_closest_point() 
                return fenics.near(distance, fenics.DOLFIN_EPS) # returns Points
        
    hemisphere1_mark = 102
    BrainHemisphere1Boundary(submesh_Hemisphere1).mark(boundaries, hemisphere1_mark, check_midpoint=False)

    class BrainHemisphere2Boundary(fenics.SubDomain):
            def __init__(self, submesh_Hemisphere2):
                fenics.SubDomain.__init__(self)
                self.bmesh_H2 = fenics.BoundaryMesh(submesh_Hemisphere2, "exterior") 
                self.bmesh_H2_bbtree = fenics.BoundingBoxTree() # rewrite bbtree for new bmesh_cortexsurface (partial)
                self.bmesh_H2_bbtree.build(self.bmesh_H2)

            def inside(self, x, on_boundary):
                _, distance = self.bmesh_H2_bbtree.compute_closest_entity(fenics.Point(*x)) # compute_closest_point() 
                return fenics.near(distance, fenics.DOLFIN_EPS) # returns Points
        
    hemisphere2_mark = 103 # for integration of contact force on the sub-boundaries of the whole mesh
    BrainHemisphere2Boundary(submesh_Hemisphere2).mark(boundaries, hemisphere2_mark, check_midpoint=False)
    
    # export marked boundaries
    export_XML_PVD_XDMF.export_PVDfile(args.output, 'boundaries_T0', boundaries)
    
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

    # Penalty coefficient (self-contact mechanics)
    #penalty_method = args.parameters["penalty_method"]
    penalty_coefficient = args.parameters["penalty_coefficient"]


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

    S_Hemisphere1 = fenics.FunctionSpace(submesh_Hemisphere1, "CG", 1) 
    S_Hemisphere1_surface = fenics.FunctionSpace(bmesh_Hemisphere1, "CG", 1)

    S_Hemisphere2 = fenics.FunctionSpace(submesh_Hemisphere2, "CG", 1) 
    S_Hemisphere2_surface = fenics.FunctionSpace(bmesh_Hemisphere2, "CG", 1)

    # Vector Function Spaces
    V = fenics.VectorFunctionSpace(mesh, "CG", 1)

    V_cortexsurface = fenics.VectorFunctionSpace(bmesh, "CG", 1) 
    
    V_BrainHemisphere1 = fenics.VectorFunctionSpace(submesh_Hemisphere1, "CG", 1)
    VH1_dofs = V_BrainHemisphere1.dofmap().dofs()

    V_Hemisphere1_surface = fenics.VectorFunctionSpace(bmesh_Hemisphere1, "CG", 1)
    
    V_BrainHemisphere2 = fenics.VectorFunctionSpace(submesh_Hemisphere2, "CG", 1)
    VH2_dofs = V_BrainHemisphere2.dofmap().dofs()

    V_Hemisphere2_surface = fenics.VectorFunctionSpace(bmesh_Hemisphere2, "CG", 1)

    # Tensor Function Spaces
    #Vtensor = fenics.TensorFunctionSpace(mesh, "DG", 0)
    Vtensor = fenics.TensorFunctionSpace(mesh,'CG', 1, shape=(3,3)) # https://fenicsproject.discourse.group/t/outer-product-evaluation/2159; https://fenicsproject.discourse.group/t/how-to-choose-projection-space-for-stress-tensor-post-processing/5568/4

    # FEM Functions
    ###############

    # Scalar functions of V
    H = fenics.Function(S, name="H") 
    d2s = fenics.Function(S, name="d2s")
    grNoGrowthZones = fenics.Function(S, name="grNoGrowthZones")
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

    #BoundaryMesh_Nt = fenics.Function(V, name="BoundaryMesh_Nt")
    #Mesh_Nt = fenics.Function(V, name="Mesh_Nt")

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
    vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping_T0 = mappings.mesh_to_surface_V(mesh.coordinates(), bmesh.coordinates()) # at t=0. (keep reference projection before contact for proximity node to deep tetrahedra vertices)

    # From Hemisphere 1 V Function Space to Whole Mesh V Function Space (common to reference and deformed mesh)
    # -----------------------------------------------------------------
    VH1_2_V_dofmap = mappings.submeshFunctionSpace_2_meshFunctionSpace_dofmap(V, V_BrainHemisphere1)
    SH1_2_S_dofmap = mappings.submeshFunctionSpace_2_meshFunctionSpace_dofmap_Scalar(S, S_Hemisphere1)
    vertexidxH1_to_DOF_SH1_mapping = fenics.vertex_to_dof_map(S_Hemisphere1)
    vertexidxV_to_DOFsV_mapping_H1 = fenics.vertex_to_dof_map(V_BrainHemisphere1).reshape((-1, 3))
    vertex2dofs_B_H1 = mappings.vertex_to_dofs_VectorFunctionSpace(V_Hemisphere1_surface, gdim)
    B_2_V_dofmap_H1, vertexBH1_2_dofsVH1_mapping = mappings.surface_to_mesh_V(gdim, V_BrainHemisphere1, V_Hemisphere1_surface, vertex2dofs_B_H1)
    SboundaryHemisphere1_2_SWholeMesh_dofmap, vertexboundaryHemisphere1_2_dofScalarFunctionSpaceWholeMesh_mapping = mappings.surface_to_mesh_S(S, S_Hemisphere1_surface)

    # From Hemisphere 2 V Function Space to Whole Mesh V Function Space (common to reference and deformed mesh)
    # -----------------------------------------------------------------
    VH2_2_V_dofmap = mappings.submeshFunctionSpace_2_meshFunctionSpace_dofmap(V, V_BrainHemisphere2)
    SH2_2_S_dofmap = mappings.submeshFunctionSpace_2_meshFunctionSpace_dofmap_Scalar(S, S_Hemisphere2)
    vertexidxH2_to_DOF_SH2_mapping = fenics.vertex_to_dof_map(S_Hemisphere2)
    vertexidxV_to_DOFsV_mapping_H2 = fenics.vertex_to_dof_map(V_BrainHemisphere2).reshape((-1, 3))
    vertex2dofs_B_H2 = mappings.vertex_to_dofs_VectorFunctionSpace(V_Hemisphere2_surface, gdim)
    B_2_V_dofmap_H2, vertexBH2_2_dofsVH2_mapping = mappings.surface_to_mesh_V(gdim, V_BrainHemisphere2, V_Hemisphere2_surface, vertex2dofs_B_H2)
    SboundaryHemisphere2_2_SWholeMesh_dofmap, vertexboundaryHemisphere2_2_dofScalarFunctionSpaceWholeMesh_mapping = mappings.surface_to_mesh_S(S, S_Hemisphere2_surface)

    # Residual Form
    ###############

    # Measurement entities 
    # --------------------
    ds = fenics.Measure("ds", domain=mesh, subdomain_data=boundaries) 
    ds_H1 = fenics.Measure("ds")(subdomain_data=boundaries_H1) # fenics.Measure("ds")[boundaries_H1] # source: https://fenicsproject.discourse.group/t/how-to-define-a-function-in-a-submesh-based-on-function-values-present-in-an-adjoining-submesh/4047/3
    ds_H2 = fenics.Measure("ds")(subdomain_data=boundaries_H2) # fenics.Measure("ds")[boundaries_H2]

    dx = fenics.Measure("dx", domain=mesh, subdomain_data=subdomains) 

    # a_new, v_new
    # ------------
    print("\nexpressing a_new and v_new thanks to β-Newmark time integration approximation...")
    a_new = numerical_scheme_temporal.update_acceleration(u, u_old, v_old, a_old, beta, dt, ufl=True)
    v_new = numerical_scheme_temporal.update_velocity(a_new, u_old, v_old, a_old, gamma, dt, ufl=True)

    # prerequisites before computing Fg and mu (H, d2s and gm=f(d2s, H) required)
    # ----------------------------------------
    print("\ninitializing distances to surface...")
    vertex2dofs_S = mappings.vertex_to_dof_ScalarFunctionSpace(S)
    d2s_ = differential_layers.compute_distance_to_cortexsurface(vertex2dofs_S, d2s, mesh, bmesh_cortexsurface_bbtree) # init at t=0.0
    projection.local_project(d2s_, S, d2s)

    print("\ninitializing differential term function...")
    projection.local_project(h, S, H) # H.assign( fenics.project(h, S) )  
    gm_ = differential_layers.compute_differential_term_DOF(S, d2s, H, gm) # init 'gm' at t=0.0
    projection.local_project(gm_, S, gm)
    """for dof in S.dofmap().dofs():
        d2s_dof = d2s.vector()[dof]
        gm.vector()[dof] = compute_differential_term(d2s_dof, H.vector()[dof]) """

    # brain regions growth mask
    # -------------------------  
    print("mark no-growth brain regions (e.g. 'longitudinal fissure' - 'ventricules' - 'mammilary bodies')...") # code source: T.Tallinen et al. 2016. See detail in https://github.com/rousseau/BrainGrowth/blob/master/geometry.py   
    for vertex, scalarDOF in enumerate(vertex2dofs_S):
        # sphere filter but is 
        """
        rqp = np.linalg.norm( np.array([  0.714*(mesh.coordinates()[vertex, 0] + 0.1), 
                                                 mesh.coordinates()[vertex, 1], 
                                                 mesh.coordinates()[vertex, 2] - 0.05  ]))"""
        
        # sphere filter
        """ 
        rqp = np.linalg.norm( np.array([  (mesh.coordinates()[vertex, 0]), 
                                           mesh.coordinates()[vertex, 1], 
                                           mesh.coordinates()[vertex, 2] - 0.10  ]))
        """ 

        # ellipsoid filter       
            
        rqp = np.linalg.norm( np.array([    0.714 * mesh.coordinates()[vertex, 0], 
                                                    mesh.coordinates()[vertex, 1], 
                                            0.9 *  (mesh.coordinates()[vertex, 2] - 0.10)   ]))  
        
        # ellipsoid filter taking into account also the length of the brain (not to consider growth of inter-hemispheres longitudinal fissure)
        """ 
        rqp = np.linalg.norm( np.array([    0.73 * mesh.coordinates()[vertex, 0], 
                                            0.85 * mesh.coordinates()[vertex, 1], 
                                            0.92 * (mesh.coordinates()[vertex, 2] - 0.15)   ])) """
                                            
        
        """                                 
        rqp = np.linalg.norm( np.array([    0.8*characteristics["maxx"]/0.8251 * mesh.coordinates()[vertex, 0], # --> 0.8*maxX/0.8251
                                            0.63 * mesh.coordinates()[vertex, 1] + 0.075, # int(0.6/maxY)
                                            0.8*characteristics["maxz"]/0.637 * mesh.coordinates()[vertex, 2] - 0.25   ])) # --> 0.8*maxZ/0.637
        """ 
        
        if rqp < 0.6:
            grNoGrowthZones.vector()[scalarDOF] = max(1.0 - 10.0*(0.6 - rqp), 0.0)
        else:
            grNoGrowthZones.vector()[scalarDOF] = 1.0
    
    # Fg
    # --
    print("\ninitializing growth coefficients: dgTAN & dgRAD...")
    projection.local_project(grTAN * gm * grNoGrowthZones * alphaTAN * ((1 - alphaF) * dt), S, dg_TAN) # dg_TAN.assign( fenics.project(grTAN * alphaTAN * gm * ((1 - alphaF) * dt), S) )
    projection.local_project(grRAD * alphaRAD *  ((1 - alphaF) * dt), S, dg_RAD) # dg_RAD.assign( fenics.project(grRAD * alphaRAD *  ((1 - alphaF) * dt), S) ) # = 0.0

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

    print("\ninitializing normals to boundary for the Whole Mesh...")
    BoundaryMesh_Nt = fenics.Function(V, name="BoundaryMesh_Nt")
    #BoundaryMesh_Nt.assign( growth.compute_topboundary_normals(mesh, ds, V) )
    boundary_normals = growth.compute_topboundary_normals(mesh, ds, V) 
    projection.local_project(boundary_normals, V, BoundaryMesh_Nt)

    print("\ninitializing normals to boundary for Hemisphere 1...")
    BoundaryMesh_Nt_Hemisphere1 = fenics.Function(V_BrainHemisphere1, name="BoundaryMesh_Nt_Hemisphere1")
    #BoundaryMesh_Nt.assign( growth.compute_topboundary_normals(mesh, ds, V) )
    boundary_normals_Hemipshere1 = growth.compute_topboundary_normals(submesh_Hemisphere1, ds_H1, V_BrainHemisphere1) 
    projection.local_project(boundary_normals_Hemipshere1, V_BrainHemisphere1, BoundaryMesh_Nt_Hemisphere1)

    print("\ninitializing normals to boundary for Hemisphere 2...")
    BoundaryMesh_Nt_Hemisphere2 = fenics.Function(V_BrainHemisphere2, name="BoundaryMesh_Nt_Hemisphere2")
    #BoundaryMesh_Nt.assign( growth.compute_topboundary_normals(mesh, ds, V) )
    boundary_normals_Hemipshere2 = growth.compute_topboundary_normals(submesh_Hemisphere2, ds_H2, V_BrainHemisphere2) 
    projection.local_project(boundary_normals_Hemipshere2, V_BrainHemisphere2, BoundaryMesh_Nt_Hemisphere2)

    print("\ninitializing projected normals of nodes of the whole mesh...")
    Mesh_Nt = fenics.Function(V, name="Mesh_Nt")
    #Mesh_Nt.assign( growth.compute_mesh_projected_normals(V, mesh.coordinates(), bmesh.coordinates(), vertexB_2_dofsV_mapping, vertex2dofs_V, BoundaryMesh_Nt) )
    mesh_normals = growth.compute_mesh_projected_normals(V, mesh.coordinates(), bmesh.coordinates(), vertexB_2_dofsV_mapping, vertex2dofs_V, BoundaryMesh_Nt) 
    projection.local_project(mesh_normals, V, Mesh_Nt)

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
    Contact_Force_Hemisphere1_VH1 = fenics.Function(V_BrainHemisphere1, name="PenaltyContactForcesHemisphere1_VBrainHemisphere1") 
    #Contact_Force_Hemisphere1_S = fenics.Function(S, name="PenaltySurfaceContactForcesHemisphere1_S") 
    Contact_Force_Hemisphere1_V = fenics.Function(V, name="PenaltyContactForcesHemisphere1_V") 

    Contact_Force_Hemisphere2_VH2 = fenics.Function(V_BrainHemisphere2, name="PenaltyContactForcesHemisphere2_VBrainHemisphere2") 
    #Contact_Force_Hemisphere2_S = fenics.Function(S, name="PenaltySurfaceContactForcesHemisphere2_S") 
    Contact_Force_Hemisphere2_V = fenics.Function(V, name="PenaltyContactForcesHemisphere2_V") 

    """
    fcontact_V_Hemisphere1, fcontact_V_Hemisphere2 = contact_penalty.contact_mechanics_algo_2HemispheresSubmeshes_DISTANCE_PENALTY(mesh, V, numerical_scheme_temporal.avg(u_old, u, alphaF), average_mesh_spacing, grNoGrowthZones,
                                                                                                                                    subdomains, V_BrainHemisphere1, V_BrainHemisphere2, 
                                                                                                                                    BoundaryMesh_Nt_Hemisphere1, BoundaryMesh_Nt_Hemisphere2,
                                                                                                                                    VH1_2_V_dofmap, VH2_2_V_dofmap,
                                                                                                                                    vertexBH1_2_dofsVH1_mapping, vertexBH2_2_dofsVH2_mapping,
                                                                                                                                    vertexboundaryHemisphere2_2_dofScalarFunctionSpaceWholeMesh_mapping,
                                                                                                                                    penalty_coefficient) 
 
    """

    fcontact_V_Hemisphere1, fcontact_V_Hemisphere2 = contact_penalty.contact_mechanics_algo_2HemispheresSubmeshes_COLLISIONS_PENALTY(mesh, BoundaryMesh_Nt, numerical_scheme_temporal.avg(u_old, u, alphaF), V, grNoGrowthZones,
                                                                                                                                    subdomains, V_BrainHemisphere1, V_BrainHemisphere2, 
                                                                                                                                    BoundaryMesh_Nt_Hemisphere1, BoundaryMesh_Nt_Hemisphere2,
                                                                                                                                    vertexidxV_to_DOFsV_mapping_H1, vertexidxV_to_DOFsV_mapping_H2,
                                                                                                                                    VH1_2_V_dofmap, vertexidxV_to_DOFsV_mapping_H1, VH2_2_V_dofmap, vertexidxV_to_DOFsV_mapping_H2,
                                                                                                                                    vertexboundaryHemisphere1_2_dofScalarFunctionSpaceWholeMesh_mapping, vertexboundaryHemisphere2_2_dofScalarFunctionSpaceWholeMesh_mapping,
                                                                                                                                    penalty_coefficient) 


    Contact_Force_Hemisphere1_VH1.assign ( fcontact_V_Hemisphere1 )
    Contact_Force_Hemisphere2_VH2.assign ( fcontact_V_Hemisphere2 )                


    #  Update interlobes contact forces (defined on V and used in the variational form) 
    # ---------------------------------------------------------------------------------

    # convert contact forces in VLeft space into forces in V space
    for dofVsub in VH1_dofs:
        Contact_Force_Hemisphere1_V.vector()[ VH1_2_V_dofmap[dofVsub] ] = Contact_Force_Hemisphere1_VH1.vector()[ dofVsub ]  

    #print("\nmin Left interlobes contact force: {}".format( np.min(fcontact_interlobes_Left_V.vector()[:]) ) )
    #print("max Left interlobes contact force: {}".format( np.max(fcontact_interlobes_Left_V.vector()[:]) ) )

    # convert contact forces in VRight space into forces in V space
    for dofVsub in VH2_dofs:
        Contact_Force_Hemisphere2_V.vector()[ VH2_2_V_dofmap[dofVsub] ] = Contact_Force_Hemisphere2_VH2.vector()[ dofVsub ] 

    #print("min Right interlobes contact force: {}".format( np.min(fcontact_interlobes_Right_V.vector()[:]) ) )
    #print("max Right interlobes contact force: {}\n".format( np.max(fcontact_interlobes_Right_V.vector()[:]) ) )

    # export contact forces Left and Right at each step
    FEniCS_FEM_Functions_file.write(Contact_Force_Hemisphere1_V, T0) 
    FEniCS_FEM_Functions_file.write(Contact_Force_Hemisphere2_V, T0) 

    """
    if len( np.where(Contact_Force_Hemisphere1_V.vector()[:] != 0.)[0] ) != 0 : # if interlobes corrective forces detected 
        
        #print( "Contact forces will be applied on {} volume DOFs to avoid interlobes collison...".format(len(np.where(fcontact_interlobes_Left_V.vector()[:] != 0.)[0]) + len(np.where(fcontact_interlobes_Right_V.vector()[:] != 0.)[0])) )

        if args.visualization == True:

            vedo.dolfin.plot(mesh,
                            Contact_Force_Hemisphere1_V, 
                            mode='mesh arrows', 
                            wireframe=True, 
                            style='bw', # meshlab; bw; matplotlib
                            scale=1.0, 
                            camera=dict(pos=(0., 0., -6.)),
                            at=0, 
                            N=2) 
            
            vedo.dolfin.plot(mesh,
                            Contact_Force_Hemisphere2_V, 
                            mode='mesh arrows', 
                            wireframe=True, 
                            style='bw', 
                            scale=1.0, 
                            camera=dict(pos=(0., 0., -6.)),
                            at=1,
                            N=2).clear() 

            time.sleep(4.)
    """
        
    # Residual Form (ufl)
    # -------------
    print("\ngenerating Residual Form to minimize...")
    res = ( numerical_scheme_spatial.m(rho, numerical_scheme_temporal.avg(a_old, a_new, alphaM), v_test) 
          + numerical_scheme_spatial.c(damping_coef, numerical_scheme_temporal.avg(v_old, v_new, alphaF), v_test) 
          + numerical_scheme_spatial.k(numerical_scheme_temporal.avg(u_old, u, alphaF), v_test, Fg_T, mu, K, gdim) # Fg
        # + numerical_scheme_spatial.contact_penalty_method_residual_term(ForceS_penalty_method_FEM, ds) 
        #  + numerical_scheme_spatial.contact_penalty_method_residual_term_Hertzian(Contact_Force_Hemisphere1_V, ds, v_test)
          + numerical_scheme_spatial.contact_residual_term_2Hemispheres_PENALTY_2Hemispheres(Contact_Force_Hemisphere1_V, Contact_Force_Hemisphere2_V, ds, v_test)
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
    nonlinearvariationalsolver.parameters['newton_solver']['maximum_iterations'] = 50 # 50 (25)
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
    
    # Export FEM function at T0
    FEniCS_FEM_Functions_file.write(d2s, T0)
    FEniCS_FEM_Functions_file.write(H, T0)
    FEniCS_FEM_Functions_file.write(gm, T0)
    FEniCS_FEM_Functions_file.write(grNoGrowthZones, T0)

    FEniCS_FEM_Functions_file.write(BoundaryMesh_Nt, T0)
    FEniCS_FEM_Functions_file.write(BoundaryMesh_Nt_Hemisphere1, T0)
    FEniCS_FEM_Functions_file.write(BoundaryMesh_Nt_Hemisphere2, T0)
    FEniCS_FEM_Functions_file.write(Mesh_Nt, T0)

    FEniCS_FEM_Functions_file.write(dg_TAN, T0)
    FEniCS_FEM_Functions_file.write(dg_RAD, T0)
    FEniCS_FEM_Functions_file.write(Fg_T, T0)

    FEniCS_FEM_Functions_file.write(mu, T0)
    
    # Resolution
    # ----------
    times = np.linspace(T0, Tmax, Nsteps+1) 

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
        projection.local_project(grTAN * gm * grNoGrowthZones * alphaTAN * ((1 - alphaF) * dt), S, dg_TAN)
        projection.local_project(grRAD * alphaRAD *  ((1 - alphaF) * dt), S, dg_RAD) 

        # Update growth tensor orientation (adaptative)
        # ---------------------------------------------
        #print("\nupdating normals to boundary and its projections to the whole mesh nodes...")
        boundary_normals = growth.compute_topboundary_normals(mesh, ds, V) 
        boundary_normals_Hemipshere1 = growth.compute_topboundary_normals(submesh_Hemisphere1, ds_H1, V_BrainHemisphere1) 
        boundary_normals_Hemipshere2 = growth.compute_topboundary_normals(submesh_Hemisphere2, ds_H2, V_BrainHemisphere2) 

        mesh_normals = growth.compute_mesh_projected_normals(V, mesh.coordinates(), bmesh.coordinates(), vertexB_2_dofsV_mapping, vertex2dofs_V, BoundaryMesh_Nt) 
        
        projection.local_project(boundary_normals, V, BoundaryMesh_Nt)
        projection.local_project(mesh_normals, V, Mesh_Nt)
        projection.local_project(boundary_normals_Hemipshere1, V_BrainHemisphere1, BoundaryMesh_Nt_Hemisphere1)
        projection.local_project(boundary_normals_Hemipshere2, V_BrainHemisphere2, BoundaryMesh_Nt_Hemisphere2)

        # Final growth tensor
        # -------------------
        #print("\nupdating growth tensor...")
        Fg = growth.compute_growth_tensor(Mesh_Nt, dg_TAN, dg_RAD, gdim)
        projection.local_project(Fg, Vtensor, Fg_T) # projection of Fg onto Vtensor Function Space     


        # Detect and compute penalty forces to include collision correction into the residual form
        ##########################################################################################
        #print("\nupdating contact forces...")  
        """
        fcontact_V_Hemisphere1, fcontact_V_Hemisphere2 = contact_penalty.contact_mechanics_algo_2HemispheresSubmeshes_DISTANCE_PENALTY(mesh, V, numerical_scheme_temporal.avg(u_old, u, alphaF), average_mesh_spacing, grNoGrowthZones,
                                                                                                                               subdomains, V_BrainHemisphere1, V_BrainHemisphere2, 
                                                                                                                               BoundaryMesh_Nt_Hemisphere1, BoundaryMesh_Nt_Hemisphere2,
                                                                                                                               VH1_2_V_dofmap, VH2_2_V_dofmap,
                                                                                                                               vertexBH1_2_dofsVH1_mapping, vertexBH2_2_dofsVH2_mapping,
                                                                                                                               vertexboundaryHemisphere2_2_dofScalarFunctionSpaceWholeMesh_mapping,
                                                                                                                               penalty_coefficient) 
        """
        fcontact_V_Hemisphere1, fcontact_V_Hemisphere2 = contact_penalty.contact_mechanics_algo_2HemispheresSubmeshes_COLLISIONS_PENALTY(mesh, BoundaryMesh_Nt, numerical_scheme_temporal.avg(u_old, u, alphaF), V, grNoGrowthZones,
                                                                                                                                        subdomains, V_BrainHemisphere1, V_BrainHemisphere2, 
                                                                                                                                        BoundaryMesh_Nt_Hemisphere1, BoundaryMesh_Nt_Hemisphere2,
                                                                                                                                        vertexidxV_to_DOFsV_mapping_H1, vertexidxV_to_DOFsV_mapping_H2,
                                                                                                                                        VH1_2_V_dofmap, vertexidxV_to_DOFsV_mapping_H1, VH2_2_V_dofmap, vertexidxV_to_DOFsV_mapping_H2,
                                                                                                                                        vertexboundaryHemisphere1_2_dofScalarFunctionSpaceWholeMesh_mapping, vertexboundaryHemisphere2_2_dofScalarFunctionSpaceWholeMesh_mapping,
                                                                                                                                        penalty_coefficient) 
                                    
        Contact_Force_Hemisphere1_VH1.assign ( fcontact_V_Hemisphere1 )
        Contact_Force_Hemisphere2_VH2.assign ( fcontact_V_Hemisphere2 )                

        #  Update interlobes contact forces (defined on V and used in the variational form) 
        # ---------------------------------------------------------------------------------

        # convert contact forces in VLeft space into forces in V space
        for dofVsub in VH1_dofs:
            Contact_Force_Hemisphere1_V.vector()[ VH1_2_V_dofmap[dofVsub] ] = Contact_Force_Hemisphere1_VH1.vector()[ dofVsub ]  

        #print("\nmin Left interlobes contact force: {}".format( np.min(fcontact_interlobes_Left_V.vector()[:]) ) )
        #print("max Left interlobes contact force: {}".format( np.max(fcontact_interlobes_Left_V.vector()[:]) ) )

        # convert contact forces in VRight space into forces in V space
        for dofVsub in VH2_dofs:
            Contact_Force_Hemisphere2_V.vector()[ VH2_2_V_dofmap[dofVsub] ] = Contact_Force_Hemisphere2_VH2.vector()[ dofVsub ] 

        #print("min Right interlobes contact force: {}".format( np.min(fcontact_interlobes_Right_V.vector()[:]) ) )
        #print("max Right interlobes contact force: {}\n".format( np.max(fcontact_interlobes_Right_V.vector()[:]) ) )

        # Solve
        #######       
        nonlinearvariationalsolver.solve() 

        # Export displacement & other FEM functions
        ###########################################
        FEniCS_FEM_Functions_file.write(u, t)
        FEniCS_FEM_Functions_file.write(Contact_Force_Hemisphere1_V, t) 
        FEniCS_FEM_Functions_file.write(Contact_Force_Hemisphere2_V, t) 
        
        FEniCS_FEM_Functions_file.write(d2s, t)
        FEniCS_FEM_Functions_file.write(H, t)
        FEniCS_FEM_Functions_file.write(gm, t)

        FEniCS_FEM_Functions_file.write(BoundaryMesh_Nt, t) 
        #FEniCS_FEM_Functions_file.write(fenics.project(BoundaryMesh_Nt_Hemisphere1, V), t)
        #FEniCS_FEM_Functions_file.write(fenics.project(BoundaryMesh_Nt_Hemisphere2, V), t)
        FEniCS_FEM_Functions_file.write(Mesh_Nt, t) 

        FEniCS_FEM_Functions_file.write(dg_TAN, t)
        FEniCS_FEM_Functions_file.write(dg_RAD, t)
        FEniCS_FEM_Functions_file.write(Fg_T, t)

        FEniCS_FEM_Functions_file.write(mu, t)

        """
        if visualization == True:
            vedo.dolfin.plot(u, 
                             mode='displace', 
                             text="Step {} / {}:\nMesh at time {} / tmax={}\nDisplacement to be applied".format(step_to_be_applied, number_steps, t_i_plus_1, tmax), 
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
        #fenics.ALE.move(submesh_Hemisphere1, u)
        #fenics.ALE.move(submesh_Hemisphere2, u)
        #submesh_Hemisphere1 = fenics.SubMesh(mesh, subdomains, 1)
        #submesh_Hemisphere2 = fenics.SubMesh(mesh, subdomains, 2)

        if args.visualization == True:
            """
            fenics.plot(mesh)
            plt.show()
            """

            """
            vedo.dolfin.plot(mesh, # u
                             mode="mesh", # mode="displace"
                             style="paraview",
                             axes=4,
                             camera=dict(pos=(-6., -8., -6.)),
                             text="step {}".format(i),
                             interactive=False).clear()
            """
            
            vedo.dolfin.plot(Contact_Force_Hemisphere1_VH1, # Contact_Force_Hemisphere1_V, Contact_Force_Hemisphere2_V, fenics.project(Contact_Force_Hemisphere1_V + Contact_Force_Hemisphere2_V, V)
                             mode="mesh arrows", scale=0.2,
                             style="paraview",
                             axes=4,
                             camera=dict(pos=(0., 0., -10.)), # camera=dict(pos=(-6., -8., -6.)); camera=dict(pos=(0., 0., -15.))
                             text="Contact Forces in Hemisphere 1. Step {}".format(i),
                             interactive=False).clear()
            time.sleep(1)

            vedo.dolfin.plot(Contact_Force_Hemisphere2_VH2, # Contact_Force_Hemisphere1_V, Contact_Force_Hemisphere2_V, fenics.project(Contact_Force_Hemisphere1_V + Contact_Force_Hemisphere2_V, V)
                             mode="mesh arrows", scale=0.02,
                             style="paraview",
                             axes=4,
                             camera=dict(pos=(0., 0., -10.)), # camera=dict(pos=(-6., -8., -6.)); camera=dict(pos=(0., 0., -15.))
                             text="Contact Forces in Hemisphere 2. Step {}".format(i),
                             interactive=False).clear()
            time.sleep(1)

            vedo.dolfin.plot(BoundaryMesh_Nt_Hemisphere1, # Contact_Force_Hemisphere1_V, Contact_Force_Hemisphere2_V, fenics.project(Contact_Force_Hemisphere1_V + Contact_Force_Hemisphere2_V, V)
                             mode="mesh arrows", scale=0.2,
                             style="paraview",
                             axes=4,
                             camera=dict(pos=(0., 0., -10.)), # camera=dict(pos=(-6., -8., -6.)); camera=dict(pos=(0., 0., -15.))
                             text="Normals at boundary in Hemisphere 1. Step {}".format(i),
                             interactive=False).clear()
            time.sleep(1)

            vedo.dolfin.plot(BoundaryMesh_Nt_Hemisphere2, # Contact_Force_Hemisphere1_V, Contact_Force_Hemisphere2_V, fenics.project(Contact_Force_Hemisphere1_V + Contact_Force_Hemisphere2_V, V)
                             mode="mesh arrows", scale=0.2,
                             style="paraview",
                             axes=4,
                             camera=dict(pos=(0., 0., -10.)), # camera=dict(pos=(-6., -8., -6.)); camera=dict(pos=(0., 0., -15.))
                             text="Normals at boundary in Hemisphere 2. Step {}".format(i),
                             interactive=False).clear()
            time.sleep(1)

        # Boundary (if d2s and Mesh_Nt need to be udpated: "solver_Fgt_norm"; "solver_Fgt")
        #print("\nupdating boundarymesh...")
        bmesh = fenics.BoundaryMesh(mesh, "exterior") # cortex envelop
        bmesh_cortexsurface_bbtree = fenics.BoundingBoxTree()
        bmesh_cortexsurface_bbtree.build(bmesh) 

        # Export mesh characteristics 
        #############################
        """
        if i%10 == 0:
            fenics.File(args.output + "mesh_{}.xml".format(t)) << mesh
            convert_meshformats.xml_to_vtk(args.output + "mesh_{}.xml".format(t), args.output + "mesh_{}.vtk".format(t))
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
    convert_meshformats.xml_to_vtk(args.output + "mesh_Tmax.xml", args.output + "mesh_Tmax.vtk")
    export_simulation_outputmesh_data.export_resultmesh_data(args.output + "analytics/",
                                                             args.output + "mesh_Tmax.vtk",
                                                             t,
                                                             i+1,
                                                             total_computational_time,
                                                             "mesh_Tmax.txt")






