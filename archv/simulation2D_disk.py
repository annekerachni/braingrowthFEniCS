import fenics 
import vedo.dolfin
import numpy as np

from utils.mesh_generators import create_xml2Dmesh_withGMSH
from braingrowth2D_disk.braingrowthFEM.input_mesh import geometry
from braingrowth2D_disk.braingrowthFEM.formulate_problem import problem
from braingrowth2D_disk.braingrowthFEM.solve_problem import solver

# ************ #
#              #
#  Input mesh: #
#              #
# ************ #

# Create input gemoetry in the .xml format (readable by FEniCS) 
mesh_folderpath = './braingrowth2D_disk/data/'
geometry_type = 'disk'
geometry_name = 'disk_1' # name that will be provided to the mesh file
mesh_parameters = {'diskcenter': np.array([0., 0., 0.]), 'radius': 1., 'elementsize': 0.03}
brain_representation = 'whole' # 'partial' if mesh does not represent the full brain (e.g. rectangle, halfdisk, quarterdisk); 'whole' otherwise (e.g. in 2D: disk; in 3D: ellipsoid, dhcp, tallinen) --> conditionnes the boundaries that will be computed (e.g. if 'whole: bcs=[])
createdmesh = create_xml2Dmesh_withGMSH.DiskMesh(mesh_folderpath, geometry_name, **mesh_parameters)
meshpath = createdmesh.meshfilepath_xml_2D # .xml mesh file
mesh = fenics.Mesh(meshpath) # FEniCS mesh
vedo.dolfin.plot(mesh, wireframe=False, text='mesh', style='paraview', axes=4).close()


# *********************** #
#                         #
#  Simulation parameters: #
#                         #
# *********************** #

# Boundaries & subdomains 
# -----------------------
subdomains_definition_parameters = {'cortical_thickness': 0.05} 

# Define Dirichlet boundary conditions 
# ------------------------------------
dirichlet_bcs_parameters = {'consider_brainsurface_bc_TrueorFalse': False, 
                            'brainsurface_bc': None} # e.g. 'consider_brainsurface_bc_TrueorFalse': True + 'brainsurface_bc' = fenics.Constant(0., -0.1); fenics.Expression(...) # 'bottom_bc_type': "fixed"; "plan_rolling"

# Define growth tensor: binary, tangential and adaptative
# -------------------------------------------------------
cortex_growth_parameters = {'gr_cortex_TAN': 1., # 'gr': yes/no growth mask for each Point. --> further developments: gr differential and not binary; gr=gr(t); gr_cortex(X) != gr_core(X) 
                            'gr_cortex_RAD': 1.,
                            'alpha_cortex_TAN': 1., # 1. for rectangle mesh / 1.829 in original BrainGrowth // g_cortex_TAN[Point] = 1 + dg_cortex_TAN[Point] = 1 + gr_cortex_TAN[Point] * alpha_cortex_TAN * dt
                            'alpha_cortex_RAD': 0.} # g_cortex_RAD =  1 + dg_cortex_RAD[Point] = 1 + gr_cortex_RAD[Point] * alpha_cortex_RAD * dt

core_growth_parameters = {'gr_core_TAN': 0., 
                          'gr_core_RAD':0.,
                          'alpha_core_TAN': 0.,  
                          'alpha_core_RAD': 0.}  


# Define brain constitutive model
# -------------------------------
brain_material_parameters = {'type':'elastic',
                             'constitutive_model': 'neo_hookean',
                             'rho': 20., # 20. for rectangle mesh / 0.01046 in original BrainGrowth
                             'damping': 10.} # 10. for rectangle mesh / 0.5 in original BrainGrowth

cortex_material_parameters = {'mu': 20., # # 20. for rectangle mesh / 1. in original BrainGrowth
                              'k': 100.} # 100. for rectangle mesh / 5. in original BrainGrowth

core_material_parameters = {'mu': 1., # # 1. for rectangle mesh / 1.167 in original BrainGrowth
                            'k': 100.} # 100. for rectangle mesh / 5. in original BrainGrowth


# External forces
# ---------------
body_forces = fenics.Constant((0., 0.)) # 2D


# Simulation time parameters
# --------------------------
simulation_time_parameters = {'tmax': 5.,
                              'number_steps': 50}
dt = fenics.Constant(simulation_time_parameters['tmax']/simulation_time_parameters['number_steps']) # as ALPHA_M and ALPHA_F parameters for temporal discretization were choosen to provide IMPLICIT NUMERICAL SCHEME, 'dt' value is a priori arbitrary (no CFL condition).
print('time step: ~{:5} s.'.format( float(dt) ))


# Numerical scheme parameters
# ---------------------------
# Temporal discretization parameters (alpha-generalized method)
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
meshObj = geometry.Mesh(meshpath, mesh_parameters, brain_representation, geometry_type) # Get personalized mesh object from 'mesh' FEniCS object

""" 
mesh_characteristics = meshObj.characteristics
centerofgravity_coordinates = meshObj.cog
min_mesh_spacing = meshObj.min_mesh_spacing # max_mesh_spacing, average_mesh_spacing 
""" 

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
                                                               results_folderpath='./braingrowth2D_disk/results/')

# Define elastodynamics solver
# ----------------------------
braingrowth_solver = solver.NonLinearDynamicMechanicsSolver(simulation_time_parameters, 
                                                            dt, 
                                                            braingrowth_problem, 
                                                            cortex_growth_parameters, core_growth_parameters,
                                                            results_folderpath='./braingrowth2D_disk/results/',
                                                            results_filename='displacements',
                                                            results_format='.xmdf')

braingrowth_solver.set_solver_parameters(nonlinearsolver="newton", 
                                         linearsolver="mumps",
                                         linearsolver_preconditionner=None) # U=A⁻¹B direct solving method O(N³) 


# Launch simulation (resolution of elastodynamics variational problem by Finite Element Method)
# ---------------------------------------------------------------------------------------------
braingrowth_solver.launch_simulation()

