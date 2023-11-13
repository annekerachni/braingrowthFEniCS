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
###### LBO #######
##################

## decomposition of curvature onto Laplace-Beltrami eigenmodes (1st method)
##############################################################

# Compute eigen modes of the initial sphere (alternative to spherical harmonics) 
# ------------------------------------------------------------------------------
print("\n>> Processing initial_sphere eigen modes with lapy (Laplace-Beltrami.) --> cf. Saad Jbabdi + Spangy")    
num_modes = 1500
emodes, evals, A, B = get_eigenmodes(Surfaces['initial_sphere'][0], Surfaces['initial_sphere'][1], num_modes) # emodes is REAL vs. to SH that are complex

# Visualise the eigenmodes basis onto the initial sphere
# ------------------------------------------------------
"""
print('\n>> Visualizing eigen modes onto the initial sphere')
mode      = int(num_modes/2) # mode to display
fig0 = view_surf_data(Geometries['initial_sphere'], emodes[:, mode], [f'eigenmode n°{mode} of the initial sphere'])  
fig0.show()
"""

# Project the curvature of the folded sphere against the SH basis
# ---------------------------------------------------------------
print('\n>> Projecting the mean curv onto the LB eigen modes basis:')
n = 1 # length of mean_curv array (one array containing one mean_curv value per node)
decomposition_coefs_LB = []
recomposed_signal_allmodes_LB = [] # all_modes

for i in tqdm(range(1, emodes.shape[1])): # eigen mode 0 is left
    
    # select eigenmode i

    if i == 1:
        Xi = emodes[:, :1]  
    else:
        Xi = emodes[:, (i-1):i] 
    
    # compute coefficient of mean curvature decomposition onto the eigenmode i
    decomposition_coef_modei = mean_curv.dot(B.transpose().dot(Xi)) # cf. Spangy
    decomposition_coefs_LB.append(decomposition_coef_modei)

    recomposed_signal_modei = np.dot(Xi, decomposition_coef_modei)
    recomposed_signal_allmodes_LB.append(recomposed_signal_modei)

decomposition_coefs_LB = np.asarray(decomposition_coefs_LB)
recomposed_signal_allmodes_LB = np.asarray(recomposed_signal_allmodes_LB) 

# plot projection
# --------------
"""
modes_to_display = [[1, 2, 3, 4], [10, 50, 100, 190]]

for modes_sery in modes_to_display:

    fig2 = view_surf_data(Geometries['initial_sphere'], [recomposed_signal_allmodes_LB[mode] for mode in modes_sery], [f'recomposed mean_curv signal at mode{mode}' for mode in modes_sery]) # plot coefficients from the projection of mean_curv onto SH, for a given mode to choose (among N_modes)
    fig2.show()

    fig3 = view_surf_data(Geometries['folded_sphere'], [recomposed_signal_allmodes_LB[mode] for mode in modes_sery], [f'mean_curv signal at mode{mode}' for mode in modes_sery]) # plot coefficients from the projection of mean_curv onto SH, for a given mode to choose (among N_modes)
    fig3.show()
"""

# plot one single mode
# --------------------
"""
mode = 40
fig4 = view_surf_data(Geometries['initial_sphere'], [recomposed_signal_allmodes_LB[mode]], [f'recomposed mean_curv signal at mode{mode}'])
fig4.show()
"""

# Power spectrum (cf. Spangy)
# --------------
power_spectrum_LB = decomposition_coefs_LB[:]**2 # coef**2

power_spectrum_LB_float = []
for i in range(len(power_spectrum_LB)):
    power_spectrum_LB_float.append( power_spectrum_LB[i][0] )
power_spectrum_LB_float = np.array(power_spectrum_LB_float)

emodes_numbers = np.linspace(1, num_modes-1, num_modes-1).astype(int)

# Power Spectrum vs. frequency (2pi/lambda^(0.5), with lambda eigen value of LBO decomposition)
fig = plt.figure()
plt.scatter(np.sqrt(evals[1:]) / (2 * np.pi), power_spectrum_LB_float[:], marker='+', s=10, linewidths=1.5) 
plt.xlabel('Frequency (m^{-1})')
plt.ylabel('Power Spectrum (coefs**2)')
plt.show()

# Power Spectrum vs. wavelength  (lambda^(0.5)/2pi with lambda eigen value of LBO decomposition)
fig = plt.figure()
plt.scatter((2 * np.pi) / np.sqrt(evals[1:]), power_spectrum_LB_float[:], marker='+', s=10, linewidths=1.5) 
plt.xlabel('Wavelength (m)')
plt.ylabel('Power Spectrum (coefs**2)')
plt.show()

# Log Power Spectrum vs. frequency
fig = plt.figure()
plt.scatter(np.sqrt(evals[1:]) / (2 * np.pi), np.log(power_spectrum_LB_float)[:], marker='+', s=10, linewidths=1.5) 
plt.xlabel('Frequency (m^{-1})')
plt.ylabel('Log Power Spectrum ( log(coefs**2) )')
plt.show()

# Log Power Spectrum vs. wavelength  (lambda^(0.5)/2pi with lambda eigen value of LBO decomposition)
fig = plt.figure()
plt.scatter((2 * np.pi) / np.sqrt(evals[1:]), np.log(power_spectrum_LB_float)[:], marker='+', s=10, linewidths=1.5) 
plt.xlabel('Wavelength (m)')
plt.ylabel('Log Power Spectrum ( log(coefs**2) )')
plt.show()

# Power Spectrum vs. emodes
"""
fig = plt.figure()
plt.bar(emodes_numbers, power_spectrum_LB_float)
plt.xlabel("Eigen modes")
plt.ylabel("Power Spectrum (coefs**2)")
plt.show()
"""

# Frequencies vs. LBO eigen modes (step function)
fig = plt.figure()
plt.scatter(emodes_numbers[:], np.sqrt(evals[1:]) / (2 * np.pi), marker='+', s=10, linewidths=1.5)
plt.xlabel('Eigen modes') 
plt.ylabel('Frequency (m^{-1})')
plt.show()

#Wavelengths vs. LBO eigen modes (step function)
fig = plt.figure()
plt.scatter(emodes_numbers[:], (2 * np.pi)/np.sqrt(evals[1:]), marker='+', s=10, linewidths=1.5)
plt.xlabel('Eigen modes') 
plt.ylabel('Wavelengths (m)')
plt.show()


## recomposed signal analysis: is the number of LBO modes 'num_modes' sufficient to represent the signal?
#######################################################################################################

# Projection of mean_curv raw signal onto the sphere and of recomposed signal from LBO decomposition modes
print('\n>> Recompose the mean curv with LB0 eigen modes (200) and see if it does well represent the initial function:')
decomposition_coef_modes_LB = mean_curv.dot(B.transpose().dot(emodes)) # cf. Spangy
decomposition_coef_modes_LB = np.asarray(decomposition_coef_modes_LB)

recomposed_signal_LB = np.dot(emodes, decomposition_coef_modes_LB)
recomposed_signal_LB = np.asarray(recomposed_signal_LB) 

"""
fig_recomposed_meancurv = view_surf_data(Geometries['initial_sphere'], [recomposed_signal_LB], [f'recomposed mean_curv signal (onto LBO modes)']) # plot coefficients from the projection of mean_curv onto SH, for a given mode to choose (among N_modes)
fig_recomposed_meancurv.show()
"""

fig_comparative = view_surf_data(Geometries['initial_sphere'], [recomposed_signal_LB, mean_curv], ['recomposed mean_curv' + f' ({num_modes} LBO modes)', 'mean_curv'])
fig_comparative.show()

""" 
# does not work: np.corrcoef works for n_nodes x n_modes arrays comparison
# Correlation matrix between mean_curv raw signal and recomposed signal from LBO decomposition modes (cf. Saad Jbabdi)
n   = mean_curv.shape[1]
num_modes = emodes.shape[1]
c_SH   = []
n_basis = []
for i in tqdm(range(1, emodes.shape[1])):
    # select subset of the eigenmodes
    X = emodes[:,:i]
    # solve glm with pseudo-inverse (sklearn does not like complex numbers)
    beta = mean_curv.dot(B.transpose().dot(X))
    meancurv_predicted = np.dot(X, beta)
    C = np.corrcoef(meancurv_predicted.T, mean_curv.T)[:num_modes,num_modes:]
    #c_SH.append( np.diag(C) )
    #n_basis.append(X.shape[1])

#c_SH = np.asarray(c_SH)

plt.imshow(C, vmin=-.2, vmax=.2)
plt.colorbar()
plt.show()
"""

# Regression against the LBO eigenmode basis
# ------------------------------------------
from sklearn.linear_model import LinearRegression

glm = LinearRegression()
n   = 1 # shape[1] of mean_curv 
c   = [] # will contain the correlation between true mean_curv and its prediction (with LBO decomposition)

for i in tqdm(range(1, num_modes)):
    # select subset of the eigenmodes
    X = emodes[:,:i]
    glm.fit(X, mean_curv)
    meancurv_predicted = glm.predict(X)
    C = np.corrcoef(meancurv_predicted.T, mean_curv.T)[:n,n:]
    c.append( np.diag(C) )
c = np.asarray(c)

plt.figure()
plt.plot(np.arange(1, num_modes), np.mean(c, axis=1), label='mean curvature modes')
plt.legend()
plt.xlabel('Number of basis functions')
plt.ylabel('Avg corr (true mean_curv vs prediction)')
plt.show()
