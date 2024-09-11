import fenics
import argparse
import json
#import vedo.dolfin
from mpi4py import MPI
import numpy as np
from tqdm import tqdm

import os, sys
sys.path.append(sys.path[0]) 
sys.path.append(os.path.dirname(sys.path[0]))

from utils.export_functions import export_XML_PVD_XDMF
from FEM_biomechanical_model_2D import mappings, differential_layers, projection, growth


#####################################################
###################### Parameters ###################
#####################################################

# Geometry 
##########
gdim = 2
mesh_path = './data/rectangle.xml'

Xmin, Xmax = 0., 5.0e-2 # in meters (brain length at 21GW ~ 60mm = 6.10⁻²m)
Ymin, Ymax = 0., 2.5e-2 # in meters (brain cortex thickness at 21GW ~ 2.5e-3 = 10% total height => 2.5.10⁻²m)
num_cell_x, num_cell_y = 100, 50 # 60, 60

H0 = fenics.Constant(2.0e-3) # 2.5e-3 (~10%) 
# cortical_thickness = fenics.Expression('H0 + 0.01*t', H0=H0, t=0.0, degree=0)

refinement_coef_in_Cortex = 20

# Elastic parameters
####################
muCortex = fenics.Constant(10000) # 300 [Pa]
muCore = fenics.Constant(1000) # 100 [Pa]

nu = 0.48
#KCortex = fenics.Constant( muCortex.values()[0] * (1 + nu) / (1 - nu) ) # 2D
#KCore = fenics.Constant( muCore.values()[0] * (1 + nu) / (1 - nu) ) # 2D
KCortex = fenics.Constant( 2*muCortex.values()[0] * (1 + nu) / (3*(1 - 2*nu)) ) # 3D
KCore = fenics.Constant( 2*muCore.values()[0] * (1 + nu) / (3*(1 - 2*nu)) ) # 3D
#KCortex  = fenics.Constant(20000) # 3000 [Pa] (nu=0.45)
#KCore  = fenics.Constant(2000) # 1000 [Pa] (nu=0.45)"

# Growth parameters
###################
alphaTAN = fenics.Constant(1.0e-5) # 5.0e-6
alphaRAD = fenics.Constant(0.)

# Time stepping
###############
T0_in_GW = 21 # GW
T0_in_seconds = T0_in_GW * 604800 # 1GW=168h=604800s

Tmax_in_GW = 29 # GW
Tmax_in_seconds = Tmax_in_GW * 604800

dt_in_seconds = 3600 # --> dt = 1500/2000 seconds to 1 hour max (for the result not to variate too much)
# 1 GW = 604800s 
# 0.1 GW = 60480 s
# 1500 s ~ 0.0025 GW (dt advised by S.Urcun)
# 3600 s (1h) ~ 0.006 GW
# 7200 s (2h) ~ 0.012 GW --> alphaTAN = 7.0e-6
# 43200 s (1/2 day) ~ 0.07 GW --> alphaTAN = 1.16e-6
# 86400 s (1 day) ~ 0.14 GW
dt_in_GW = dt_in_seconds / 604800
print('\ntime step: {} seconds <-> {:.3f} GW'.format(dt_in_seconds, dt_in_GW)) # in original BrainGrowth: dt = 0,000022361 ~ 2.10⁻⁵

Nsteps = (Tmax_in_seconds - T0_in_seconds) / dt_in_seconds 
print('\nNsteps: ~{}'.format( int(Nsteps) ))

# Form compiler options
#######################
"""See https://fenicsproject.org/olddocs/dolfin/1.6.0/python/demo/documented/hyperelasticity/python/documentation.html"""
fenics.parameters["form_compiler"]["optimize"] = True
fenics.parameters["form_compiler"]["cpp_optimize"] = True # The form compiler to use C++ compiler optimizations when compiling the generated code.
fenics.parameters["form_compiler"]["quadrature_degree"] = 3 # --> number of node on each edge where to solve the unknown
#fenics.parameters["allow_extrapolation"] = False 
fenics.parameters["std_out_all_processes"] = False #  turn off solver logs

# Output folder & file
######################
output_folder_path = './results/'

outputpath = os.path.join(output_folder_path, "growth_simulation.xdmf")
FEniCS_FEM_Functions_file = fenics.XDMFFile(outputpath)
FEniCS_FEM_Functions_file.parameters["flush_output"] = True
FEniCS_FEM_Functions_file.parameters["functions_share_mesh"] = True
FEniCS_FEM_Functions_file.parameters["rewrite_function_mesh"] = True

#####################################################
###################### Problem ######################
#####################################################

# Build rectangle geometry
##########################
mesh = fenics.RectangleMesh(fenics.Point(Xmin, Ymin), fenics.Point(Xmax, Ymax + refinement_coef_in_Cortex*H0.values()[0]), 
                            num_cell_x, num_cell_y, 
                            "right/left")

fenics.File(mesh_path) << mesh

#vedo.dolfin.plot(mesh, wireframe=False, text='mesh', style='paraview', axes=4).close()

# Refine mesh in Cortex
#######################
# Cortex / Core delineation
Y_cortex = Ymax - H0.values()[0]

# Densify
X = mesh.coordinates()

vertices_belonging_to_Cortex = np.where(X[:,1] > Y_cortex)[0]

for vertex in list(vertices_belonging_to_Cortex):
    #-0.5*(x[0]-0.5) 
    X[vertex,1] = Y_cortex + (1/refinement_coef_in_Cortex) * (X[vertex,1] - Y_cortex) #https://fenicsproject.org/qa/13422/refining-mesh-near-boundaries/
    #X[vertex,1] = Ymax + 0.5*(X[vertex,1] - H0.values()[0])

#vedo.dolfin.plot(mesh, wireframe=False, text='mesh', style='paraview', axes=4).close()

# boundary mesh
###############
bmesh = fenics.BoundaryMesh(mesh, "exterior")

# Define subdomains & boundaries
################################
# initialize boundaries --> See: https://fenicsproject.discourse.group/t/how-to-mark-boundaries-to-a-circle-and-change-solution-from-cartesian-to-polar/5492/4
# N.B. 'boundaries' is used to defined Dirichlet BCs & 'boundaries_cortexsurface' is used to define cortical surface submesh
boundaries = fenics.MeshFunction("size_t", mesh, mesh.topology().dim() - 1) # size: facets --> mesh.num_facets() => number of faces in the volume
boundaries.set_all(100)

boundaries_cortexsurface = fenics.MeshFunction('size_t', bmesh, bmesh.topology().dim(), 0) # size: n_faces (at surface) => number of faces at the surface
boundaries_cortexsurface.set_all(100)

# define and mark cortex surface boundary
"""
class CortexSurface(fenics.SubDomain): 
    def __init__(self, ymax): 
        fenics.SubDomain.__init__(self)
        self.ymax = ymax

    def inside(self, x, on_boundary):
        return fenics.near(x[1], self.ymax)  # https://fenicsproject.discourse.group/t/how-to-build-vectorfunctionspace-on-part-of-a-2d-mesh-boundary/9648/4  

cortex_surface_boundary = CortexSurface(Ymax)
""" 
class CortexSurface(fenics.SubDomain): 
    def __init__(self, X): 
        fenics.SubDomain.__init__(self)
        self.X = X

    def inside(self, x, on_boundary):
        return fenics.near(x[1], np.max(self.X[:,1]))
    
cortex_surface_boundary = CortexSurface(X)
cortex_surface_boundary.mark(boundaries, 101, check_midpoint=False)  # Mark brainsurface boundary https://fenicsproject.discourse.group/t/how-to-compute-boundary-mesh-and-submesh-from-an-halfdisk-mesh/9812/2
cortex_surface_boundary.mark(boundaries_cortexsurface, 101, check_midpoint=False)  

# define and mark other boundaries (for Dirichlet BCs)
class Left(fenics.SubDomain): 
    def __init__(self, xmin): 
        fenics.SubDomain.__init__(self)
        self.xmin = xmin

    def inside(self, x, on_boundary):
        return on_boundary and fenics.near(x[0], self.xmin)  
    
class Right(fenics.SubDomain): 
    def __init__(self, xmax): 
        fenics.SubDomain.__init__(self)
        self.xmax = xmax

    def inside(self, x, on_boundary):
        return on_boundary and fenics.near(x[0], self.xmax)  
    
class Bottom(fenics.SubDomain): 
    def __init__(self, ymin): 
        fenics.SubDomain.__init__(self)
        self.ymin = ymin

    def inside(self, x, on_boundary):
        return on_boundary and fenics.near(x[1], self.ymin)  
    
left_boundary = Left(Xmin)
left_boundary.mark(boundaries, 102)

right_boundary = Right(Xmax)
right_boundary.mark(boundaries, 103)

bottom_boundary = Bottom(Ymin)
bottom_boundary.mark(boundaries, 104)

# export marked boundaries
output_folder_path = './results/'
export_XML_PVD_XDMF.export_PVDfile(output_folder_path, 'boundaries_T0', boundaries)

# Compute Cortex surface bmesh
##############################
bmesh = fenics.BoundaryMesh(mesh, "exterior")

bmesh_sub_101 = fenics.SubMesh(bmesh, boundaries_cortexsurface, 101) # part of the boundary mesh standing for the cortical surface
with fenics.XDMFFile(MPI.COMM_WORLD, output_folder_path + "cortical_bmesh.xdmf") as xdmf:
    xdmf.write(bmesh_sub_101)

print("\ncomputing and marking boundaries...")
bmesh_cortexsurface_bbtree = fenics.BoundingBoxTree()
bmesh_cortexsurface_bbtree.build(bmesh_sub_101) 
    
# Build FEM spaces
##################
# Scalar Function Spaces
S = fenics.FunctionSpace(mesh, "CG", 1) 
S_cortexsurface = fenics.FunctionSpace(bmesh_sub_101, "CG", 1) # scalars on the Cortex boundary mesh

# Vector Function Spaces
V = fenics.VectorFunctionSpace(mesh, "CG", 1)
V_cortexsurface = fenics.VectorFunctionSpace(bmesh_sub_101, "CG", 1) # vectors (e.g. displacement) on the Cortex boundary mesh

# Tensor Function Spaces
#Vtensor = fenics.TensorFunctionSpace(mesh, "DG", 0)
T = fenics.TensorFunctionSpace(mesh,'CG', 1, shape=(gdim, gdim)) # https://fenicsproject.discourse.group/t/outer-product-evaluation/2159; https://fenicsproject.discourse.group/t/how-to-choose-projection-space-for-stress-tensor-post-processing/5568/4

# FEM functions
###############
# Scalar functions of V
H = fenics.Function(S, name="H") 
d2s = fenics.Function(S, name="d2s")
grGrowthZones = fenics.Function(S, name="grGrowthZones")
gm = fenics.Function(S, name="gm") 
mu = fenics.Function(S, name="mu") 
K = fenics.Function(S, name="K") 

dg_TAN = fenics.Function(S, name="dgTAN")
dg_RAD = fenics.Function(S, name="dgRAD") 

# Vector functions of V
u = fenics.Function(V, name="Displacement") # Trial function. Current (unknown) displacement
du = fenics.TrialFunction(V)
v_test = fenics.TestFunction(V) # Test function

BoundaryMesh_Nt = fenics.Function(V, name="BoundaryMesh_Nt")
Mesh_Nt = fenics.Function(V, name="Mesh_Nt")

# Vector functions of Vtensor
Fg_T = fenics.Function(T, name="Fg")    
PK1tot_T = fenics.Function(T, name="PK1tot") 
    
# Compute mappings
##################
# From vertex to DOF_S in the whole mesh --> used to compute distance to surface d2s (for differential function used both in growth rate and mu)
# ----------------------------------------------------------------------------------
vertex2dof_S = mappings.vertex_to_dof_ScalarFunctionSpace(S)

# From vertex to DOFs_V in the whole mesh --> used to compute Mesh_Nt
# -------------------------------------------------------------------
vertex2dofs_V = mappings.vertex_to_dofs_VectorFunctionSpace(V, gdim) # TODO: check if return mapping has coherent shape
vertex2dofs_V101 = mappings.vertex_to_dofs_VectorFunctionSpace(V_cortexsurface, gdim)

# From the surface mesh (cortex envelop) to the whole mesh (B100_2_V_dofmap; vertexB100_2_dofsV_mapping --> used to compute Mesh_Nt)
# --------------------------------------------------------
V101_2_V_dofmap, vertex101_2_dofsV_mapping = mappings.cortexsurface_to_mesh_V(gdim, V, V_cortexsurface, vertex2dofs_V101)
S101_2_S_dofmap, vertex101_2_dofS_mapping = mappings.cortexsurface_to_mesh_S(S, S_cortexsurface)

# From the whole mesh to the cortex surface mesh (to be use for projections onto surface in contact process)
# ----------------------------------------------
vertexWholeMesh_to_projectedVertexCortexBoundaryMesh101_mapping_T0 = mappings.wholemesh_to_cortexsurface_vertexmap(mesh.coordinates(), bmesh_sub_101.coordinates()) # at t=0. (keep reference projection before contact for proximity node to deep tetrahedra vertices)

# Define geometrical and mechanical properties
##############################################
# prerequisites before computing Fg and mu (H, d2s and gm=f(d2s, H) required)
d2s_ = differential_layers.compute_distance_to_cortexsurface(vertex2dof_S, d2s, mesh, bmesh_cortexsurface_bbtree) # init at t=0.0 #d2s_ = differential_layers.compute_distance_to_cortexsurface_2(vertex2dof_S, d2s, mesh, bmesh_cortexsurface_cKDtree) # init at t=0.0
projection.local_project(d2s_, S, d2s)

projection.local_project(H0, S, H) # H.assign( fenics.project(h, S) )  

gm_ = differential_layers.compute_differential_term(S, d2s, H, gm) # init 'gm' at t=0.0
projection.local_project(gm_, S, gm)
"""for dof in S.dofmap().dofs():
    d2s_dof = d2s.vector()[dof]
    gm.vector()[dof] = compute_differential_term(d2s_dof, H.vector()[dof]) """
    
# initializing local stiffness (mu) and bulk modulus (K) using a smooth delineation between cortex and inner layers
mu_ = differential_layers.compute_shear_and_bulk_stiffnesses(gm, muCore, muCortex)
projection.local_project(mu_, S, mu)
#mu_ = fenics.project(mu_, S)
#mu.assign( mu_ )

K_ = differential_layers.compute_shear_and_bulk_stiffnesses(gm, KCore, KCortex)
projection.local_project(K_, S, K)
    
# Measurement entities 
######################
ds = fenics.Measure("ds", domain=mesh, subdomain_data=boundaries) 

# Define adaptative Growth Tensor
#################################
# define growth ratio in tangential and radial directins
projection.local_project(gm * alphaTAN * dt_in_seconds, S, dg_TAN) 
projection.local_project(alphaRAD * dt_in_seconds, S, dg_RAD) 

# compute normals to Cortex surface bmesh
"""BoundaryMesh_Nt.assign( growth.compute_topboundary_normals(mesh, ds, V) )"""
boundary_normals = growth.compute_topboundary_normals(mesh, ds(101), V) 
projection.local_project(boundary_normals, V, BoundaryMesh_Nt)

# project normals to all Cortex nodes (nodes that will have a growth only)
"""Mesh_Nt.assign( growth.compute_mesh_projected_normals(V, mesh.coordinates(), bmesh.coordinates(), vertexB100_2_dofsV_mapping, vertex2dofs_V, BoundaryMesh_Nt) )"""
mesh_normals = growth.compute_mesh_projected_normals(V, mesh.coordinates(), bmesh_sub_101.coordinates(), vertex101_2_dofsV_mapping, vertex2dofs_V, BoundaryMesh_Nt) 
projection.local_project(mesh_normals, V, Mesh_Nt)

# define Growth Tensor
"""helpers.local_project( compute_growth_tensor(Mesh_Nt, dg_TAN, dg_RAD), Vtensor, Fg) # init at t=0.0 (local_project equivalent to .assign())"""
Fg = growth.compute_growth_tensor(Mesh_Nt, dg_TAN, dg_RAD, gdim)
projection.local_project(Fg, T, Fg_T) 

# Constitutive law
##################

# Get elastic deformation gradient (multiplicative decomposition of the total deformation gradient)
Id = fenics.Identity(gdim)

F = fenics.variable( Id + fenics.grad(u) ) # F: deformation gradient --> F = I₃ + ∇u / [F]: 3x3 matrix 

Fg_inv = fenics.variable( fenics.inv(Fg) )
Fe = fenics.variable( F * Fg_inv )# F_t * (F_g)⁻¹ --> Fe: elastic part of the deformation gradient

# Cauchy-Green tensors (elastic part of the deformation only)
Ce = fenics.variable( Fe.T * Fe )
Be = fenics.variable( Fe * Fe.T )

# Invariants 
Je = fenics.variable( fenics.det(Fe) ) 
Tre = fenics.variable( fenics.tr(Be) )

# Neo-Hookean strain energy density function
#We = 0.5 * mu * (Tre * pow(Je, -2/3) - 3) + 0.5 * K * (Je - fenics.ln(Je) - 1) # T.Tallinen
We = 0.5 * mu * (Tre * pow(Je, -2/3) - 3) + 0.5 * K * (Je - 1) * (Je - 1) # X.Wang https://github.com/rousseau/BrainGrowth/ 

# Compute Piola-Kirchhoff I stress
# PK1e = fenics.diff(We, Fe) # --> elastic part. """ PK1e = fenics.variable( Je * Te * fenics.inv(Fe.T) ) # X.Wang https://github.com/rousseau/BrainGrowth/
PK1tot = fenics.diff(We, F) # --> total. Need to translate Pelastic into Ptotal to appsly the equilibrium balance momentum law.

# Residual form
###############
res = fenics.inner(PK1tot, fenics.grad(v_test)) * fenics.Measure("dx")

# Add contact mechanics
#######################
# compute local curvature

# if curvature is high, porental auto-collision --> compute normal gap

# if gn  penalization in the residual form (penalty method)

# Dirichlet BCs
###############
bc_Dirichlet_Left = fenics.DirichletBC(V, fenics.Constant((0., 0.)), boundaries, 102) # no displacement in x,y,z --> fixed zone to avoid additional solution including Rotations & Translations
bc_Dirichlet_Right = fenics.DirichletBC(V, fenics.Constant((0., 0.)), boundaries, 103)
bc_Dirichlet_Bottom = fenics.DirichletBC(V, fenics.Constant((0., 0.)), boundaries, 104)

bcs = [bc_Dirichlet_Left, bc_Dirichlet_Right, bc_Dirichlet_Bottom]

# FEniCS problem
################
jacobian = fenics.derivative(res, u, du) # we want to find u that minimize F(u) = 0 (F(u): total potential energy of the system), where F is the residual form of the PDE => dF(u)/du 
problem = fenics.NonlinearVariationalProblem(res, u, bcs, jacobian) 

####################################################
###################### Solver ######################
####################################################

# FEniCS solver
###############
solver = fenics.NonlinearVariationalSolver(problem) 
# info(solver.parameters, True) # display the list of available parameters and default values
# https://home.simula.no/~hpl/homepage/fenics-tutorial/release-1.0-nonabla/fenics_tutorial_1.0.pdf
#https://link.springer.com/content/pdf/10.1007/978-3-319-52462-7_5.pdf
# https://fenicsproject.org/qa/5894/nonlinearvariationalsolver-tolerance-what-solver-options/ (first used)

# solver parameters
###################
solver.parameters["nonlinear_solver"] = 'newton'
#solver.parameters['newton_solver']['convergence_criterion'] = "incremental" 
solver.parameters['newton_solver']['absolute_tolerance'] = 1E-3 # 1E-7 # 1E-10 for unknown (displacement) in mm
solver.parameters['newton_solver']['relative_tolerance'] = 1E-4 # 1E-8 # 1E-11 for unknown (displacement) in mm
solver.parameters['newton_solver']['maximum_iterations'] = 25 # 50 (25)
solver.parameters['newton_solver']['relaxation_parameter'] = 1.0 # means "full" Newton-Raphson iteration expression: u_k+1 = u_k - res(u_k)/res'(u_k) => u_k+1 = u_k - res(u_k)/jacobian(u_k)

# CHOOSE AND PARAMETRIZE THE LINEAR SOLVER IN EACH NEWTON ITERATION (LINEARIZED PROBLEM) 
solver.parameters['newton_solver']['linear_solver'] = 'mumps' # 'gmres' # linearized problem: AU=B --> Choose between direct method U=A⁻¹B O(N³) (e.g. 'mumps') or iterative/Krylov subspaces method U=A⁻¹B~(b + Ab + A²b + ...) O(num_iter * N²) (e.g. 'gmres' for non-symmetric problem , 'cg') to compute A⁻¹. 
solver.parameters['newton_solver']['preconditioner'] = None # 'sor'

if solver.parameters['newton_solver']['preconditioner'] != None:
    solver.parameters['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-9 #1E-9
    solver.parameters['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-10 #1E-7
    solver.parameters['newton_solver']['krylov_solver']['maximum_iterations'] = 1000 # number of iterations with Krylov subspace method

# Reusing previous unknown u_n as the initial guess to solve the next iteration n+1 
#solver.parameters['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True # https://link.springer.com/content/pdf/10.1007/978-3-319-52462-7_5.pdf --> "Using a nonzero initial guess can be particularly important for timedependent problems or when solving a linear system as part of a nonlinear iteration, since then the previous solution vector U will often be a good initial guess for the solution in the next time step or iteration."
# parameters['krylov_solver']['monitor_convergence'] = True # https://fenicsproject.org/qa/1124/is-there-a-way-to-set-the-inital-guess-in-the-krylov-solver/


########################################################
###################### Simulation ######################
########################################################
    
# Export FEM function at T0_in_GW
#################################
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

# Initialize energies to get
############################
energies = np.zeros((int(Nsteps+1), 4))
E_damp = 0
E_ext = 0

# Loop
######
times = np.linspace(T0_in_seconds, Tmax_in_seconds, int(Nsteps+1))  # in seconds!

for i, dt in enumerate( tqdm( np.diff(times), desc='brain is growing...', leave=True) ): # dt = dt_in_seconds

    # collisions (fcontact_global_V) have to be detected at each step

    fenics.set_log_level(fenics.LogLevel.ERROR) # in order not to print solver info logs 

    t = times[i+1] # in seconds
    t_in_GW = t / 604800 
        
    # Update pre-required entities
    # ----------------------------
    # H
    #print("\nupdating cortical thickness...")
    """
    h.t = t
    #H.assign( fenics.project(h, S) )# Expression -> scalar Function of the mesh
    projection.local_project(cortical_thickness, S, H) # H.assign( fenics.project(cortical_thickness, S) )  
    """

    # d2s
    #print("\nupdating distances to surface...")
    #d2s.assign( differential_layers.compute_distance_to_cortexsurface(vertex2dof_S, d2s, mesh, bmesh_cortexsurface_bbtree) )
    d2s_ = differential_layers.compute_distance_to_cortexsurface(vertex2dof_S, d2s, mesh, bmesh_cortexsurface_bbtree) 
    #d2s_ = differential_layers.compute_distance_to_cortexsurface_2(vertex2dof_S, d2s, mesh, bmesh_cortexsurface_cKDtree) 
    projection.local_project(d2s_, S, d2s)
    
    # gm
    #print("\nupdating differential term function...")
    #gm = differential_layers.compute_differential_term(S, d2s, H, gm)
    gm_ = differential_layers.compute_differential_term(S, d2s, H, gm) 
    projection.local_project(gm_, S, gm)

    # Update differential material stiffness mu 
    # -----------------------------------------
    # mu have to be updated at each timestep (material properties evolution with deformation) (So do previously H, d2s, gm)
    #print("\nupdating local stiffness...")       
    mu_ = differential_layers.compute_shear_and_bulk_stiffnesses(gm, muCore, muCortex)
    projection.local_project(mu_, S, mu)
    
    K_ = differential_layers.compute_shear_and_bulk_stiffnesses(gm, KCore, KCortex)
    projection.local_project(K_, S, K)

    # Update growth tensor coefficients
    # ---------------------------------
    #print("\nupdating growth coefficients: dgTAN & dgRAD...")
    projection.local_project(gm * alphaTAN * dt, S, dg_TAN)
    projection.local_project(alphaRAD * dt, S, dg_RAD) 

    # Update growth tensor orientation (adaptative)
    # ---------------------------------------------
    #print("\nupdating normals to boundary and its projections to the whole mesh nodes...")
    boundary_normals = growth.compute_topboundary_normals(mesh, ds(101), V) 
    projection.local_project(boundary_normals, V, BoundaryMesh_Nt)
    
    mesh_normals = growth.compute_mesh_projected_normals(V, mesh.coordinates(), bmesh_sub_101.coordinates(), vertex101_2_dofsV_mapping, vertex2dofs_V, BoundaryMesh_Nt) 
    projection.local_project(mesh_normals, V, Mesh_Nt)

    # Final growth tensor
    # -------------------
    #print("\nupdating growth tensor...")
    Fg = growth.compute_growth_tensor(Mesh_Nt, dg_TAN, dg_RAD, gdim)
    projection.local_project(Fg, T, Fg_T) # projection of Fg onto Vtensor Function Space    
    

    # Detect and compute penalty forces to include collision correction into the residual form
    ##########################################################################################
    #print("\nupdating contact forces...")
    """
    Ft = contact.correct_collisions(mesh, V, muCortex, K, vertex2dofs_V, vertexB100_2_dofsV_mapping)
    fcontact_global_V.assign( Ft )
    """

    # Solve
    #######       
    solver.solve() 

    # Export displacement & other FEM functions
    ###########################################
    FEniCS_FEM_Functions_file.write(u, t_in_GW)
    
    projection.local_project(PK1tot, T, PK1tot_T) # export Piola-Kirchhoff stress
    FEniCS_FEM_Functions_file.write(PK1tot_T, t_in_GW)
    
    """FEniCS_FEM_Functions_file.write(fcontact_global_V, t_in_GW)"""
    
    FEniCS_FEM_Functions_file.write(d2s, t_in_GW)
    FEniCS_FEM_Functions_file.write(H, t_in_GW)
    FEniCS_FEM_Functions_file.write(gm, t_in_GW)

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
    # E_ext += assemble( Wext(u-u_old) )
    E_tot = E_elas #-E_ext
    
    energies[i+1, :] = np.array([E_elas, E_tot])
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
    bmesh = fenics.BoundaryMesh(mesh, "exterior") # update bmesh 

    bmesh_sub_101 = fenics.SubMesh(bmesh, boundaries_cortexsurface, 101)
    
    bmesh_cortexsurface_bbtree = fenics.BoundingBoxTree()
    bmesh_cortexsurface_bbtree.build(bmesh_sub_101) 
    
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
