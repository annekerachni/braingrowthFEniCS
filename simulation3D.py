import fenics 
import vedo.dolfin

from braingrowth3D.braingrowthFEM.input_mesh import geometry
from braingrowth3D.braingrowthFEM.formulate_problem import problem
from braingrowth3D.braingrowthFEM.solve_problem import solver
from utils.converters import mesh_format_converters_3D, normalisation

# ************ #
#              #
#  Input mesh: #
#              #
# ************ #

# Load .xml input mesh file (directly readable by FEniCS)
# -------------------------------------------------------
meshpath = './braingrowth3D/normalized_data/brain_simp_red.xml'
geometry_type = 'brain' # 'brain', 'ellispoid', 'sphere', ('block') 

# Information on input data:
## ellipsoid:
# './braingrowth3D/normalized_data/ellipsoid/ellipsoid_tetracells.xml' --> 6761 nodes, 35497 tets

## sphere:
# './braingrowth3D/normalized_data/sphere/sphere.xml' --> 13465 nodes, 66557 tets

## dhcp:
# './braingrowth3D/normalized_data/dhcp/dhcpRAS_iso_coarse.xml' --> 1378 nodes, 5955 tets 
# './braingrowth3D/normalized_data/dhcp/dhcpRAS_iso_moderate.xml' --> 4204 nodes, 20204 tets 
# './braingrowth3D/normalized_data/dhcp/dhcpRAS_iso_fine.xml' --> 99461 nodes, 525389 tets

## tallinen:
# './braingrowth3D/normalized_data/tallinen/week23-3M-tets.xml' --> 496994 nodes, 2782075 tets 

## other:
# './braingrowth3D/normalized_data/brain_simp_red.xml' --> 8622 nodes, 42981 tets


# Load .mesh input mesh file and convert it into .xml format (readable by FEniCS)
# -------------------------------------------------------------------------------
""" print("loading mesh...")
loaded_MESHmeshpath = './braingrowth3D/raw_data/brain_simp_red.mesh' 
meshpath = './braingrowth3D/normalized_data/brain_simp_red.xml' 
print("pre-processing loaded mesh...")
MESHmesh = mesh_format_converters_3D.load_MESHmesh(loaded_MESHmeshpath)
coordinates0, n_nodes = mesh_format_converters_3D.get_nodes(MESHmesh) 
tets, n_tets = mesh_format_converters_3D.get_tetrahedrons(MESHmesh, n_nodes)
coordinates_normalized = normalisation.normalise_coord(coordinates0) 
mesh_format_converters_3D.convert_mesh_to_xml(meshpath, coordinates_normalized, tets)
geometry_type = 'brain' 
"""

# Load .vtk input 3D mesh and convert it into .xml format (readable by FEniCS)
# ----------------------------------------------------------------------------
"""
print("loading mesh...")
loaded_mesh_VTKpath = './data/loaded_meshes/ellipsoid/ellipsoid.vtk'
loaded_mesh = loaded_mesh_VTKpath.replace('.vtk', '')
meshpath = mesh_format_converters_3D.vtk_to_xml(loaded_mesh_VTKpath, loaded_mesh)
geometry_type = 'ellipsoid'
"""


# *********************** #
#                         #
#  Simulation parameters: #
#                         #
# *********************** #

# Boundaries & subdomains 
# -----------------------
subdomains_definition_parameters = {'cortical_thickness': 0.22} # original BrainGrowth value: 0.042 --> TODO: gradient-layer model will make possible to decrease H
# if cortical_thickness is too low, risk of discontinous cortex submesh (generally in case of coarse mesh) and applied tangential growth will be missed at some cortex regions:
## ellipsoid: 0.2; ellipsoid_fine: 0.1
## dhcp_coarse: 0.35; dhcp_moderate: 0.35; dhcp_fine: 0.07
## tallinen (week23-3M-tets.xml): 0.07 
## brain_simp_red : 0.22


# Define Dirichlet boundary conditions (default: bcs=[])
# ------------------------------------
dirichlet_bcs_parameters = {'consider_brainsurface_bc_TrueorFalse': False, 
                            'brainsurface_bc': None} # e.g. 'consider_brainsurface_bc_TrueorFalse': True + 'brainsurface_bc' = fenics.Constant(0., -0.1); fenics.Expression(...)


# Define growth tensor: binary, tangential and adaptative
# -------------------------------------------------------
cortex_growth_parameters = {'gr_cortex_TAN': 1., # 'gr': yes/no growth mask for each Point. --> further developments: gr differential and not binary; gr=gr(t); gr_cortex(X) != gr_core(X) 
                            'gr_cortex_RAD':1.,
                            'alpha_cortex_TAN': 1.829, # original BrainGrowth value: 1.829. g_cortex_TAN[Point] = 1 + dg_cortex_TAN[Point] = 1 + gr_cortex_TAN[Point] * alpha_cortex_TAN * dt
                            'alpha_cortex_RAD': 0.} # g_cortex_RAD =  1 + dg_cortex_RAD[Point] = 1 + gr_cortex_RAD[Point] * alpha_cortex_RAD * dt

core_growth_parameters = {'gr_core_TAN': 0., 
                          'gr_core_RAD':0.,
                          'alpha_core_TAN': 0.,  
                          'alpha_core_RAD': 0.}  


# Define brain constitutive model
# -------------------------------
brain_material_parameters = {'type':'elastic',
                             'constitutive_model': 'neo_hookean',
                             'rho': 0.01046, # original BrainGrowth value: 0.01 
                             'damping': 0.5} # original BrainGrowth value: 0.5 

cortex_material_parameters = {'mu': 1., # original BrainGrowth value: 1. 
                              'k': 50.} # original BrainGrowth value: 5.

core_material_parameters = {'mu': 1.167, # original BrainGrowth value: 1.167 
                            'k': 50.} # original BrainGrowth value: 5. 


# External forces
# ---------------
body_forces = fenics.Constant((0., 0., 0.))


# Simulation time parameters
# --------------------------
simulation_time_parameters = {'tmax': 1.,
                              'number_steps': 100} # 50. (often crashing), 100., 1000. increase to get better linearized approximation and to avoid singularities.
dt = fenics.Constant(simulation_time_parameters['tmax']/simulation_time_parameters['number_steps']) # as ALPHA_M and ALPHA_F parameters for temporal discretization were choosen to provide IMPLICIT NUMERICAL SCHEME, 'dt' value is a priori arbitrary (no CFL condition).
print('time step: ~{:5} s.'.format( float(dt) ))


# Numerical scheme parameters
# ---------------------------
# Temporal discretization parameters (generalized-α method)
temporal_discretization_parameters = {'alphaM': 0.2, 'alphaF': 0.4 }
# alphaM = 0.2; alphaF = 0.4 --> Generalized-α method; Unconditional stability (implicit temporal discretization); Energy-dissipative scheme ("numerical damping")
# alphaM = 0.; alphaF = 0. --> Newmark method; Energy-conservative scheme 


# ************* #
#               #
# Main program: #
#               #
# ************* #

# FEniCS based-mesh 
# -----------------
meshObj = geometry.Mesh(meshpath, geometry_type) # Get personalized mesh object from 'mesh' FEniCS object
mesh = meshObj.mesh
vedo.dolfin.plot(mesh, wireframe=False, text='mesh', style='paraview', axes=4).close()

characteristics = meshObj.characteristics # n_nodes, coordinates, n_tets, n_faces_Surface, n_faces_Volume, minx, maxx, miny, maxy, minz, maxz 
cog = meshObj.cog # center of gravity
min_mesh_spacing = meshObj.min_mesh_spacing # meshObj also has following attributes: max_mesh_spacing, average_mesh_spacing 


# Define elastodynamics problem
# -----------------------------
braingrowth_problem = problem.NonLinearDynamicMechanicsProblem(meshObj, 
                                                               subdomains_definition_parameters, 
                                                               temporal_discretization_parameters, dt,
                                                               brain_material_parameters, 
                                                               cortex_material_parameters, 
                                                               core_material_parameters,
                                                               dirichlet_bcs_parameters,
                                                               body_forces,
                                                               results_folderpath='./braingrowth3D/results/')

# Define elastodynamics solver
# ----------------------------
braingrowth_solver = solver.NonLinearDynamicMechanicsSolver(simulation_time_parameters, 
                                                            dt, 
                                                            braingrowth_problem, 
                                                            cortex_growth_parameters, core_growth_parameters,
                                                            results_folderpath='./braingrowth3D/results/',
                                                            results_filename='displacements',
                                                            results_format='.xmdf')

braingrowth_solver.set_solver_parameters(nonlinearsolver='newton', 
                                         linearsolver='gmres', # U=A⁻¹B iterative solving method O(N².Niter). Reduces algorithm complexity. 'gmres" for non-symmetric systems. 'cg' for symmetric, positive problem.
                                         linearsolver_preconditionner='sor') # 'ilu'; 'sor' (default preconditionner for iterative solving: 'ilu')
# See https://fenicsproject.org/pub/tutorial/html/._ftut1017.html to choose solver parameters


# Launch simulation (resolution of elastodynamics variational problem by Finite Element Method)
# ---------------------------------------------------------------------------------------------
braingrowth_solver.launch_simulation()

