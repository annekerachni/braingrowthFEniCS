# credits: Saad Jbabdi --> https://colab.research.google.com/drive/1MrkHAenYXn7EXrwKcwVyMT6KbcjrXHMm#scrollTo=bqw4dbI_C10-

import numpy as np
import nibabel as nib
from tqdm import tqdm
from fsl.data import gifti, cifti
import matplotlib.pyplot as plt

import slam.io as sio
import slam.curvature as scurv
import slamviz.plot as splt



## helpers 
##########

def read_mesh(surf_file):
    """get vertices and triangles from mesh
    Returns : (vertices, triangles)
    """
    gii = nib.load(surf_file)
    return gii.darrays[0].data, gii.darrays[1].data

# Below uses nilearn for plotting but decided to use plotly directly for flexibility
# def view_surface_data(surf_geom, data):
#   surf_file = f'L.{surf_geom}.surf.gii'
#   from nilearn import plotting
#   engine = 'plotly'
#   view = plotting.view_surf(surf_file, data)
#   return view

# for 3D plotting
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def view_surf_data(surf_geom_path, data, labels=None):

    v , t = read_mesh(surf_geom_path)

    if type(data) is not list:
        data = [data]

    n = len(data)

    fig = make_subplots(rows=1, cols=n,
                        specs=[[{'type': 'surface'}]*n],
                        column_titles=labels)
    
    for i in range(n):
        x = go.Mesh3d(x = v[:,0], y = v[:,1], z = v[:,2],
                      i = t[:,0], j = t[:,1], k = t[:,2],
                      intensity = data[i],
                      showscale=True, colorscale='plasma' )
        
        fig.add_trace(x, row=1, col=i+1)

    camera = {'eye' : dict(x=-1.5, y=0., z=0)}
    prop   = {dim+'axis' : dict(visible=False) for dim in 'xyz'}
    fig.update_scenes(camera=camera, **prop)

    return fig

# Laplace-Beltrami Eigenmodes stuff
from lapy import Solver, TriaMesh

def get_eigenmodes(v, t, num_modes, return_evals=True):

    mesh          = TriaMesh(v, t)
    solver        = Solver(mesh)
    A, B = solver.stiffness, solver.mass # Get B matrix is more respectful of the mesh geometry and rigourously discretize an integration of X and mean_curv onto triangles (J.Lefèvre) . B enables to compute more rigourous decomposition coefficients. 
    evals, emodes = solver.eigs(k=num_modes)     # Eigenmodes

    if return_evals:
        return emodes, evals, A, B
    
    else:
        return emodes, A, B

# SPHERICAL HARMONICS STUFF
def cart2sph(xyz): # xyz: all vertices coordinates
    n   = np.linalg.norm(xyz,axis=1,keepdims=True)
    n[n==0] = 1
    xyz = xyz / n
    th  = np.arccos(xyz[:,2])
    st  = np.sin(th)
    idx = (st==0)
    st[idx] = 1
    ph  = np.arctan2(xyz[:,1]/st,xyz[:,0]/st)
    ph[idx] = 0
    return th, ph

from scipy.special import sph_harm

def form_SHmat(xyz, max_order): # xyz: all vertices coordinates

    # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm.html

    pol, az = cart2sph(xyz) # pol: "Polar (colatitudinal) coordinate; must be in [0, pi]" ; az: "Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
    mat = []

    for l in range(0, max_order+1):
        for m in range(-l, l+1):
            mat.append(sph_harm(m, l, az, pol)) # az: theta ; pol: phi

    return np.asarray(mat).T # --> returns "complex scalar or ndarray the harmonic sampled at theta and phi" Ym,l.





##########################################
### smooth & folded sphere to analyze ####
##########################################

print('\n>> Loading surfaces of initial and folded spheres:') 

# Load surface of the initial sphere and of the folded sphere 
#############################################################
Geometries = {
              'initial_sphere': './data/surfaces/initial_sphere005refinedcoef5.gii',
              'folded_sphere': './data/surfaces/sphere005refinedcoef5_alphaTAN4_alphaRAD0_rho001_gamma05_K100_muCortex20_muCore1_H004_tmax1_nsteps100_solver0_globalcontact_t04.gii'
             }   

Surfaces = {}
for surf_geom, surf_path in Geometries.items():
    print(f'Loading {surf_geom} surface geometry')
    #surf_file       = f'{surf_path}'

    # get surface
    v, t   = read_mesh(surf_path)
    Surfaces[surf_geom] = [v, t] # v: vertices coordinates; t: triangles 3 indices


##################
### mean_curv ####
##################

## get mean_curvature (fonction to analyze)
###########################################
print('\n>> Computing the mean curvature of the folded sphere:')

# LOAD FOLDED MESH
# ----------------
mesh = sio.load_mesh(Geometries['folded_sphere'])

vertices = mesh.vertices
num_vertices = len(vertices)
print('{} vertices'.format(num_vertices))

"""
faces = mesh.faces
num_faces = len(faces)
print('{} faces'.format(num_faces))
"""

# COMPUTE CURVATURE
# -----------------
PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh)
mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])

# Plot of mean curvature on the mesh
visb_sc = splt.visbrain_plot(mesh=mesh,
                             tex=mean_curv,
                             caption='Mean Curvature',
                             cmap='jet')
visb_sc.preview()


##################
###### SH ########
##################

## decomposition of curvature onto spherical harmonics (2nd method)
######################################################

# Compute the spherical harmonics bases onto the initial sphere
# -------------------------------------------------------------
print('\n>> Computing Spherical Harmonics onto the initial sphere')
max_order = 13  # this corresponds to a max number of components (around 196)
SH_basis = form_SHmat(Surfaces['initial_sphere'][0], max_order) # = basis of all complex Yl,m(theta, phi)

# Visualise the spherical harmonics bases onto the initial sphere
# ---------------------------------------------------------------
print('\n>> Visualizing Spherical Harmonics onto the initial sphere')
mode      = 40 # mode to display
fig = view_surf_data(Geometries['initial_sphere'], SH_basis[:, mode].real) # SH_basis[:, mode].imag])
fig.show()

# Project the curvature of the folded sphere against the SH basis
# ---------------------------------------------------------------
print('\n>> Projecting the mean curv onto the SH basis:')
n = 1 # length of mean_curv array (one array containing one mean_curv value per node)
decomposition_coefs_SH = []
recomposed_signal_allmodes_SH = [] # all_modes

for i in tqdm(range(1, SH_basis.shape[1])):
    
    # select eigenmode i
    if i == 1:
        Xi = SH_basis[:, :1]  
    else:
        Xi = SH_basis[:, (i-1):i] # len = num_vertices --> X corresponds to eigenvectors? associated to the given eigenmode i, at each node.
    
    # compute coefficient of mean curvature decomposition onto the eigenmode i
    decomposition_coef_modei = np.dot(np.linalg.pinv(Xi), mean_curv) # --> coefficient a1, a2... of the decomposition of mean_curv onto the SH basis.
    decomposition_coefs_SH.append(decomposition_coef_modei)

    recomposed_signal_modei = np.dot(Xi, decomposition_coef_modei).real # signal recomposition onto mode i of SH basis (-->signal recomposition component "i")
    recomposed_signal_allmodes_SH.append(recomposed_signal_modei)

decomposition_coefs_SH = np.asarray(decomposition_coefs_SH)
recomposed_signal_allmodes_SH = np.asarray(recomposed_signal_allmodes_SH) # one mode --> one real value (of recomposed "mean_curv) per node 

# plot projection
# --------------
modes_to_display = [1, 20, 50, 100, 190]

fig2 = view_surf_data(Geometries['initial_sphere'], [recomposed_signal_allmodes_SH[mode] for mode in modes_to_display], [f'mode{mode}' for mode in modes_to_display]) # plot coefficients from the projection of mean_curv onto SH, for a given mode to choose (among N_modes)
fig2.show()

fig3 = view_surf_data(Geometries['folded_sphere'], [recomposed_signal_allmodes_SH[mode] for mode in modes_to_display], [f'mode{mode}' for mode in modes_to_display]) # plot coefficients from the projection of mean_curv onto SH, for a given mode to choose (among N_modes)
fig3.show()

# plot one single mode
# --------------------
mode = 40
fig4 = view_surf_data(Geometries['initial_sphere'], [recomposed_signal_allmodes_SH[mode]], [f'mode{mode}'])
fig4.show()

# Power spectrum 
# --------------
# TODO: find how to get B and evals with SH + power spectrum of complex coefficients 