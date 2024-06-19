# references: https://fenicsproject.discourse.group/t/compression-of-elastic-sphere/7480/2
# penalty solver: https://fenicsproject.discourse.group/t/is-there-a-contact-model-in-fenics-for-two-imported-physical-volumes/2624/9; https://github.com/evzenkorec/thesis_contact

import fenics
import gmsh
#import matplotlib.pyplot as plt
import meshio
import numpy as np
import math
from tqdm import tqdm
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

from utils.export_functions import export_XML_PVD_XDMF

# input mesh & output folder
############################
output_folder = "./contact_mechanics/results/one_sphere_Eulerian_elastic/"
sphere_mesh_path = "./contact_mechanics/data/sphere_mesh.xml"

# Form compiler options
#######################
fenics.parameters["form_compiler"]["optimize"]     = True
fenics.parameters["form_compiler"]["cpp_optimize"] = True
fenics.parameters["form_compiler"]["quadrature_degree"] = 2 # --> number of node on each edge where to solve the unknown
    
# Input mesh
############
inputmesh_format = sphere_mesh_path.split('.')[-1]

if inputmesh_format == "xml":
    sphere_mesh = fenics.Mesh(sphere_mesh_path)

elif inputmesh_format == "xdmf":
    sphere_mesh = fenics.Mesh()
    with fenics.XDMFFile(sphere_mesh_path) as infile:
        infile.read(sphere_mesh)


# Initial coordinates of the spheremesh
X = fenics.SpatialCoordinate(sphere_mesh) # reference configuration 

# initialize boundaries
boundary_markers = fenics.MeshFunction("size_t", sphere_mesh, sphere_mesh.topology().dim() - 1)  
boundary_markers.set_all(0)
    
# Definition of contact zones
radius_Z = 0.5 * (np.max(sphere_mesh.coordinates()[:,2]) - np.min(sphere_mesh.coordinates()[:,2]))
contact_zone = fenics.CompiledSubDomain('on_boundary && x[2] < penetration + tol', tol=fenics.DOLFIN_EPS, penetration=radius_Z) # Contact search - contact part of the boundary
contact_zone.mark(boundary_markers, 1)

# Dirichlet BC (to fix the sphere to avoid rotations)
fixed_displacement_zone = fenics.CompiledSubDomain('on_boundary && x[2] > fixed_height && x[2] < sphere_height+tol', tol=fenics.DOLFIN_EPS, fixed_height=1.9*radius_Z, sphere_height=2*radius_Z) # Contact search - contact part of the boundary
fixed_displacement_zone.mark(boundary_markers, 2)

# export marked boundaries
path = os.path.join(output_folder, 'boundaries_T0.pvd')
fenics.File(path).write(boundary_markers)

# Residual form integration surface measure
ds = fenics.Measure('ds', domain=sphere_mesh, subdomain_data=boundary_markers)

# Function Spaces
#################
#S = fenics.FunctionSpace(sphere_mesh, "CG", 1)
V = fenics.VectorFunctionSpace(sphere_mesh, "CG", 2) # CG, 2 to have enought continuity to get the approximated result for unknown displacement
#T = fenics.TensorFunctionSpace(sphere_mesh,'CG', 1, shape=(3,3))

# Functions
###########
u = fenics.Function(V, name="Displacement")
#x = X + u # current configuration
du = fenics.TrialFunction(V)
v_test = fenics.TestFunction(V)

# Parameters (SI)
############

# Elastic parameters
# ------------------
#nu = 0.45 #fenics.Constant(0.3)
#K = 112.5 # N.m⁻² = Pa
#mu = 3 * K * (1 - 2 * nu) / (2 * (1 + nu)) # N.mm⁻²
#E = 9*mu*K/(3*K + mu) 
E = 2 * 1e3 # 20 kPa
#mu = 0.3 * 1e3 # N.m⁻²
nu = 0.3
mu_ = E/(2*(1+nu))
lmbda = E*nu/(1+nu)/(1-2*nu)
#K = E * mu / (3 * (3 * mu - E)) # N.m⁻² = Pa

# Contact coefficient
# -------------------
#epsilon = fenics.Constant(1e4) 
#h = sphere_mesh.hmin() 
h = 0.005
#epsilon = E/h # V.A Yastrebov
epsilon = fenics.Constant(1e8)

# Paramereters for motion
# -----------------------
rho = 5 # kg.m⁻³ # 100 withoutcontact
g = 9.81 # m.s-2
b = fenics.Constant((0, 0, -rho*g))

# Hyperelasticity
#################
Id = fenics.Identity(3)

# F: deformation gradient
Fe = fenics.variable( Id + fenics.grad(u) ) # F = I₃ + ∇u / [F]: 3x3 matrix

# Fe: elastic part of the deformation gradient
"""
Fg_inv = fenics.variable( fenics.inv(Fg) )
Fe = fenics.variable( F * Fg_inv )# F_t * (F_g)⁻¹
"""

"""
Ce = fenics.variable( Fe.T * Fe )
Be = fenics.variable( Fe * Fe.T )

Je = fenics.variable( fenics.det(Fe) ) 
Tre = fenics.variable( fenics.tr(Be) ) # Eulerian

Be_deviatoric = fenics.variable( Be * Je**(-2/3))
Tre_deviatoric = fenics.variable( fenics.tr(Be_deviatoric) ) # Eulerian
"""

#We = 0.5 * mu * (Tre * pow(Je, -2/3) - 3) + 0.5 * K * (Je - fenics.ln(Je) - 1) # T.Tallinen
#We = 0.5 * mu * (Tre * pow(Je, -2/3) - 3) + 0.5 * K * (Je - 1) * (Je - 1) # X.Wang https://github.com/rousseau/BrainGrowth/

#sigma = fenics.variable( mu_ * (Be - Tre/3 * Id) * pow(Je, -5/3) + K * (Je - 1) * Id ) # X.Wang https://github.com/rousseau/BrainGrowth/
#sigma = fenics.variable(mu_ / Je * (Be_deviatoric  - Tre_deviatoric/3 * Id) + K * (Je - 1) * Id)
#We = (mu / 2) * (Tre - 3) - mu * fenics.ln(Je) + (lmbda / 2) * (Je-1)**2
sigma = 2 * mu_ * fenics.grad(u) + lmbda * fenics.tr(fenics.grad(u)) * Id

#PK1tot = fenics.diff(We, Fe) 


###
"""
def eps(v):
    return fenics.sym(fenics.grad(v))
def sigma(v):
    return lmbda*fenics.tr(eps(v))*fenics.Identity(3) + 2.0*mu*eps(v)
"""

def normal_gap(u, z_plane): # Definition of gap function (if gap < 0 => penalization)
    x = fenics.SpatialCoordinate(sphere_mesh) # in order to recompute x at each new reference configuration
    return x[2] + u[2] - z_plane # compute each time new virtual configuration at t+1

def mackauley(x):
    return (x + abs(x))/2

# Contact --> One side contact of sphere over fixed rigid plane z=h
#########
z_plane = -h # leave one element size for contact to be well detected at the basis of the sphere
"""
gap = fenics.Function(S, name="Gap")
#gap.assign(fenics.project(mackauley(u[2]- z_plane), V)) # u[2] < z_plane => penalize
gap.assign(fenics.project(mackauley(z_plane - u[2]), S)) # u[2] < z_plane => penalize
"""

# Res Form
##########
n = fenics.Constant((0., 0., 1.))

#res_form = fenics.inner(sigma, fenics.grad(v_test)) * fenics.Measure("dx") - fenics.dot(b, v_test) * fenics.Measure("dx") - epsilon * fenics.dot(mackauley( -normal_gap(u, z_plane) ) * n, v_test) * ds(1) 
#res_form = fenics.inner(2 * mu_ * fenics.grad(u) + lmbda * fenics.tr(fenics.grad(u)) * Id, fenics.grad(v_test)) * fenics.Measure("dx") - fenics.dot(b, v_test) * fenics.Measure("dx") - epsilon * fenics.dot(mackauley( -normal_gap(u, z_plane) ) * n, v_test) * ds(1) 

# REF 
res_form = fenics.inner(2 * mu_ * fenics.grad(u) + lmbda * fenics.tr(fenics.grad(u)) * Id, fenics.grad(v_test)) * fenics.Measure("dx") \
         - fenics.dot(b, v_test) * fenics.Measure("dx") \
         - epsilon * fenics.dot(mackauley( -normal_gap(u, z_plane) ) * n, v_test) * ds(1)  

J = fenics.derivative(res_form, u, du) 

bc_Top_Sphere_X =  fenics.DirichletBC(V.sub(0), fenics.Constant((0)), boundary_markers, 2) # fix sphere nodes at surface boundary 2 in the X direction
bc_Top_Sphere_Y =  fenics.DirichletBC(V.sub(1), fenics.Constant((0)), boundary_markers, 2) # fix sphere nodes at surface boundary 2 in the Y direction
bc_Top_Sphere_Z =  fenics.DirichletBC(V.sub(2), fenics.Constant((0)), boundary_markers, 2)
bcs = [bc_Top_Sphere_X, bc_Top_Sphere_Y, bc_Top_Sphere_Z] # Prescribed displacement at top nodes of the sphere
problem = fenics.NonlinearVariationalProblem(res_form, u, bcs, J) 

# Solver
########
solver = fenics.NonlinearVariationalSolver(problem)

solver.parameters["nonlinear_solver"] = "newton"
solver.parameters['newton_solver']['absolute_tolerance'] = 1E-9
solver.parameters['newton_solver']['relative_tolerance'] = 1E-10 
solver.parameters['newton_solver']['maximum_iterations'] = 20

#solver.parameters["newton_solver"]["linear_solver"] = "gmres"
#solver.parameters["newton_solver"]["preconditioner"] = "sor"


# SNES solver for contact mechanics
"""
snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "gmres",
                                          "maximum_iterations": 20,
                                          "report": True,
                                          "error_on_nonconvergence": False}} # https://bitbucket.org/fenics-project/dolfin/src/946dbd3e268dc20c64778eb5b734941ca5c343e5/python/demo/undocumented/contact-vi-snes/demo_contact-vi-snes.py?at=master#demo_contact-vi-snes.py-1,87:88,99



# Set up the non-linear solver
solver  = fenics.NonlinearVariationalSolver(problem)
solver.parameters.update(snes_solver_parameters)
fenics.info(solver.parameters, True)
"""

"""
solver_options = {"nonlinear_solver": "snes",
                          "snes_solver" : {
                              "linear_solver": 'mumps',
                              "preconditioner": 'amg',
                              "maximum_iterations": 40,
                              "relative_tolerance": 1e-9,
                              "absolute_tolerance": 1e-9,
                              "error_on_nonconvergence": True},
                          }
solver.update_options(solver_options)
"""


solver.solve()

# Export results
################
file_results = fenics.XDMFFile( os.path.join(output_folder, "results_sphere_contact_SIGMA_Contact_Elastic.xdmf") )
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True
file_results.parameters["rewrite_function_mesh"] = True
file_results.write(u, 0.)
#file_results.write(gap, 0.)
#file_results.write(p, 0.) 

"""
T0 = 0.
Tmax = 10.
Nsteps = 10
dt = fenics.Constant((Tmax-T0)/Nsteps)
times = np.linspace(T0, Tmax, Nsteps+1) 

for t, dt in enumerate(times):
    solver.solve()
    file_results.write(u, t)
    fenics.ALE.move(sphere_mesh, u)

    #fenics.plot(u, mode="displacement")
    #plt.show()
"""

