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
def analyze_mesh_folding_pattern(Geometries, num_modes, visualization_mode):
    """
    Analyse the decompositon of the folded sphere curvature using the Laplace Beltrami operator.
    Deduce the characteristical wavelegnths associated to that decomposition, and especially the mean wavelength.
    """

    print('\n>> Loading surfaces of initial and folded spheres:') 

    # Load surface of the initial sphere and of the folded sphere 
    #############################################################
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

    if visualization_mode == True:
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
    
    """
    FIRST VERSION
    
    n = 1 # length of mean_curv array (one array containing one mean_curv value per node)
    decomposition_coefs_LB = []
    recomposed_signal_allmodes_LB = [] # all_modes

    for i in tqdm(range(1, emodes.shape[1])): # eigen mode 0 is left
        
        # select eigenmode i
        elif i == 1:
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
    """
    
    n = 1 # length of mean_curv array (one array containing one mean_curv value per node)
    decomposition_coefs_LB = []
    recomposed_signal_allmodes_LB = [] # all_modes

    for i in tqdm(range(0, emodes.shape[1])): 
        
        # select eigenmode i
        Xi = emodes[:, i]  
        
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

    """
    power_spectrum_LB_float = []
    for i in range(len(power_spectrum_LB)):
        power_spectrum_LB_float.append( power_spectrum_LB[i][0] )
    power_spectrum_LB_float = np.array(power_spectrum_LB_float)
    """

    emodes_numbers = np.linspace(1, num_modes, num_modes).astype(int)

    if visualization_mode == True:
        # Power Spectrum vs. frequency (2pi/lambda^(0.5), with lambda eigen value of LBO decomposition)
        """
        plt.figure()
        plt.scatter(np.sqrt(evals[:]) / (2 * np.pi), power_spectrum_LB[:], marker='+', s=10, linewidths=1.5) 
        plt.xlabel('Frequency (m^{-1})')
        plt.ylabel('Power Spectrum (coefs**2)')
        plt.show()
        """

        # Power Spectrum vs. wavelength  (lambda^(0.5)/pi with lambda eigen value of LBO decomposition)
        plt.figure()
        plt.scatter(np.pi/np.sqrt(evals[1:]), power_spectrum_LB[1:], marker='+', s=10, linewidths=1.5) 
        # N.B. np.pi/np.sqrt(evals[1:]) & power_spectrum_LB[1:] to remove the 1st mode from the analysis. 
        # N.B. Indeed, this 1st mode (evals[0] & emodes[0]), the coefficient decomposition_coefs_LB[0] resulting from the projection of curvature onto emodes is constant and associated to the mean of the curvature onto the sphere surface. Corresponding to the "inifiny" wavelength.
        plt.xlabel('Wavelength (m)')
        plt.ylabel('Power Spectrum (coefs**2)')
        plt.show()

        # Log Power Spectrum vs. frequency
        """
        plt.figure()
        plt.scatter(np.sqrt(evals[:]) / (2 * np.pi), np.log(power_spectrum_LB)[:], marker='+', s=10, linewidths=1.5) 
        plt.xlabel('Frequency (m^{-1})')
        plt.ylabel('Log Power Spectrum ( log(coefs**2) )')
        plt.show()
        """

        # Log Power Spectrum vs. wavelength  (lambda^(0.5)/pi with lambda eigen value of LBO decomposition)
        plt.figure()
        plt.scatter(np.pi/np.sqrt(evals[1:]), np.log(power_spectrum_LB)[1:], marker='+', s=10, linewidths=1.5) 
        plt.xlabel('Wavelength (m)')
        plt.ylabel('Log Power Spectrum ( log(coefs**2) )')
        plt.show()
        

        # Power Spectrum vs. emodes
        """
        plt.figure()
        plt.bar(emodes_numbers, power_spectrum_LB_float)
        plt.xlabel("Eigen modes")
        plt.ylabel("Power Spectrum (coefs**2)")
        plt.show()
        """

        # Frequencies vs. LBO eigen modes (step function)
        """
        plt.figure()
        plt.scatter(emodes_numbers[:], np.sqrt(evals[:])/(2 * np.pi), marker='+', s=10, linewidths=1.5)
        plt.xlabel('Eigen modes') 
        plt.ylabel('Frequency (m^{-1})')
        plt.show()
        """

        #Wavelengths vs. LBO eigen modes (step function)
        """
        plt.figure()
        plt.scatter(emodes_numbers[1:], np.pi/np.sqrt(evals[1:]), marker='+', s=10, linewidths=1.5)
        plt.xlabel('Eigen modes') 
        plt.ylabel('Wavelengths (m)')
        plt.show()
        """


    ## recomposed signal analysis: is the number of LBO modes 'num_modes' sufficient to represent the signal?
    #######################################################################################################

    if visualization_mode == True:
        # Projection of mean_curv raw signal onto the sphere and of recomposed signal from LBO decomposition modes
        """ print('\n>> Recompose the mean curv with LB0 eigen modes ({}) and see if it does well represent the initial function:'.format(num_modes)) """
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

    # Regression against the LBO eigenmode basis (--> if curvature is well approximated by the number of LBO modes choosen. If not, maybe increase the number.)
    # ------------------------------------------
    """
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
    """

    return np.pi/np.sqrt(evals[:]), power_spectrum_LB[:] # wavelength: see formula 6 in [J.Lefèvre et al. 2012] https://www.researchgate.net/profile/Olivier-Coulon/publication/254044129_Fast_surface-based_measurements_using_first_eigenfunction_of_the_Laplace-Beltrami_Operator_Interest_for_sulcal_description/links/00463529ce8eeb4cdf000000/Fast-surface-based-measurements-using-first-eigenfunction-of-the-Laplace-Beltrami-Operator-Interest-for-sulcal-description.pdf


if __name__ == '__main__':
    
    import argparse
    import os
    
    import sys
    sys.path.append(".")
    
    from utils.converters import convert_stl_to_gii

    """
    # input file (result from the braingrowthFEnICS simulation)
    folded_mesh_vtk = './simulations/25sept/sphere_Ptot_alpha4_nsteps100_newtonabs3rel2relax1_gmres_sor/sphere_Ptot_alpha4_nsteps100_newtonabs3rel2relax1_gmres_sor_iteration76_100_volume.vtk'

    #convert .vtk folded mesh into .stl format
    from utils.converters import convert_meshformats_to_stl
    folded_mesh_stl = folded_mesh_vtk.split('.vtk')[0] + ".stl"
    convert_meshformats_to_stl.vtk_to_stl(folded_mesh_vtk, folded_mesh_stl)
    """
    # --> FIND A WAY TO GET A STL WITH NO FACES INVERTED!

    """
    # Convert the folded mesh at time of interest from .xdmf into .stl:
    # -----------------------------------------------------------------
    # 1. Paraview: 
    # Import folded mesh (.xdmf) and place at time of interest (e.g. end of the simulation)
    # Filter / "Extract Surface"
    # Save Data into .stl

    # 2. Meshlab: revert potential inverted triangle elements
    # Filer / "Reorient all faces coherently" 
    # Eventually Filter / "Invert Faces Orientation"
    # "Export Mesh As..." stl. (Untick "Binary encoding")
    
    """
    
    parser = argparse.ArgumentParser(description='Compute principal wavelength of the folded sphere mesh (by braingrowthFEniCS) using Laplace-Beltrami operator decomposition')

    #parser.add_argument('-fs', '--foldedspheremeshpath', help='Path to the folded mesh geometry (.stl) on which to compute curvature', type=str, required=False, 
    #default='./simulations/series_of_nsteps/nsteps100/sphere_Ptot_alpha3_nsteps2000_newtonabs8rel7relax1_gmres_sor_time0_4.stl') 
    
    parser.add_argument('-is', '--initialspheresurface', help='Path to the initial smooth mesh geometry (.stl) corresponding to mesh at t=0, on which to compute LBO eigenmodes basis', type=str, required=False, 
    default='./data/surface_meshes/initial_sphere005refinedcoef5_V2.stl') 
    
    parser.add_argument('-s', '--step', help='step number of the folded sphere mesh', type=int, required=False, 
    default=100) 
    
    #parser.add_argument('-t', '--numericaltime', help='numerical time (between 0. and 1.) of the folded sphere mesh', type=float, required=False, 
    #default=1.0) 
    
    parser.add_argument('-t', '--numericaltimes', help='numerical time (between 0. and 1.)', type=int, nargs='+', required=False, 
    default=[1.0])
    
    parser.add_argument('-nm', '--numberofeigenmodes', help='Number of eigen modes on which to decompose the curvature of the folded sphere mesh', type=int, required=False, 
    default=1500) 
    
    #parser.add_argument('-o', '--outputfolderpath', help='Path to the output folder where to save .txt file on wavelengths and power spectrum', type=str, required=False, 
    #default='./simulations/series_of_nsteps/nsteps2000/analytics/')  
    
    #parser.add_argument('-tf', '--txtfilename', help='Name of the text file to save', type=str, required=False, 
    #default='foldedsphere_curvature_LBOanalysis_nsteps2000_time0_4.txt') 
    
    """ 
    parser.add_argument('-it', '--step', help='step reached', type=int, required=False, 
                        default=1700)
    
    parser.add_argument('-n', '--nsteps', help='nsteps goal', type=int, required=False, 
                        default=2000)
    """
    parser.add_argument('-v', '--visualization', help='Visualization during simulation', type=bool, required=False, default=False)
    #parser.add_argument('-v', '--visualization', help='Visualization during simulation', action='store_true')

    args = parser.parse_args()
    ##
    
    
    # Number of modes used to decompose the folded mesh curvature on
    # --------------------------------------------------------------
    num_modes = args.numberofeigenmodes

    # initial sphere mesh surface .gii
    # ----------------------------------
    initial_mesh_stl = args.initialspheresurface

    triangle_mesh_0, faces_0, vertices_0 = convert_stl_to_gii.read_stl_mesh(initial_mesh_stl)

    initial_mesh_gii = initial_mesh_stl.split('.stl')[0] + ".gii"
    convert_stl_to_gii.write_gii_mesh(vertices_0, faces_0, initial_mesh_gii)
    
    # reference smooth mesh path and folded mesh path (after braingrowthFEniCS simulation)
    # ------------------------------------------------------------------------------------
    Geometries = {}
    Geometries['initial_sphere'] = initial_mesh_gii


    for numericaltime in args.numericaltimes:
        # convert folded mesh from .stl to .gii format 
        # --------------------------------------------
        folded_mesh_stl = './simulations/references_Fg0/sphere_Ptot_alphaTAN3_nsteps100_newtonabs3rel2relax1_gmres_sor/sphere_Ptot_alpha3_nsteps{}_newtonabs3rel2relax1_gmres_sor_time{}.stl'.format(args.step, str(numericaltime).split('.')[0] + "_" + str(numericaltime).split('.')[-1])
        
        triangle_mesh, faces, vertices = convert_stl_to_gii.read_stl_mesh(folded_mesh_stl)

        folded_mesh_gii = folded_mesh_stl.split('.stl')[0] + ".gii"
        convert_stl_to_gii.write_gii_mesh(vertices, faces, folded_mesh_gii)

        # reference smooth mesh path and folded mesh path (after braingrowthFEniCS simulation)
        # ------------------------------------------------------------------------------------
        Geometries['folded_sphere'] = folded_mesh_gii

        # compute LBO decomposition of the folded mesh curvature (at Tf)
        # --------------------------------------------------------------
        wavelenghts_array, power_spectrum_array = analyze_mesh_folding_pattern(Geometries, num_modes, args.visualization) # len(wavelenghts_array) = len(power_spectrum_array) = 1499, eigen mode 0 is removed.
        
        # extract principal wavelength of the foled mesh
        # ----------------------------------------------
        wavelenghts_array_without_mode0 = wavelenghts_array[1:]
        power_spectrum_array_without_mode0 = power_spectrum_array[1:]
        principal_wavelength = [ wavelenghts_array_without_mode0[i] for i in range(len(power_spectrum_array_without_mode0)) if power_spectrum_array_without_mode0[i] == np.max(power_spectrum_array_without_mode0) ]
        
        # export .txt data
        # ----------------        
        output_folder_path = './simulations/references_Fg0/sphere_Ptot_alphaTAN3_nsteps100_newtonabs3rel2relax1_gmres_sor/analytics/'
        filetext_name = 'foldedsphere_abs3rel2_nsteps{}_curvature_LBOanalysis_time{}.txt'.format(args.step, str(numericaltime).split('.')[0] + "_" + str(numericaltime).split('.')[-1])
        filetext_path = os.path.join(output_folder_path, filetext_name)
        
        try:
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
        except OSError:
            print ('Error: Creating directory. ' + output_folder_path) 
            
        filetxt = open(filetext_path, "w")
        
        filetxt.write('LBO Analysis of the curvature of the folded sphere mesh\n')
        filetxt.write('*******************************************************\n')
        
        filetxt.write('>> numerical time: {}\n'.format(numericaltime))
        #filetxt.write('>> current step: {}\n'.format(args.step))
        #filetxt.write('>> nsteps to reach: {}\n'.format(args.nsteps))
        filetxt.write('\n')

        filetxt.write('>> num_modes :{}\n'.format(num_modes))
        filetxt.write('\n')
        
        #filetxt.write('>> wavelength (1499) : {} \n'.format(wavelenghts_array))
        filetxt.write('>> wavelength mode1 = {} mm \n'.format(wavelenghts_array_without_mode0[0]))
        filetxt.write('>> wavelength mode2 = {} mm \n'.format(wavelenghts_array_without_mode0[1]))
        filetxt.write('>> wavelength mode2 = {} mm \n'.format(wavelenghts_array_without_mode0[2]))
        filetxt.write('>> wavelength mode3 = {} mm \n'.format(wavelenghts_array_without_mode0[3]))
        filetxt.write('>> wavelength last mode = {} mm \n'.format(wavelenghts_array_without_mode0[-1]))
        filetxt.write('\n')
        
        #filetxt.write('>> power spectrum values (1499) : {} \n'.format(power_spectrum_array))
        filetxt.write('>> power spectrum value mode1 = {} mm \n'.format(power_spectrum_array_without_mode0[0]))
        filetxt.write('>> power spectrum value mode2 = {} mm \n'.format(power_spectrum_array_without_mode0[1]))
        filetxt.write('>> power spectrum value mode3 = {} mm \n'.format(power_spectrum_array_without_mode0[2]))
        filetxt.write('>> power spectrum value mode4 = {} mm \n'.format(power_spectrum_array_without_mode0[3]))
        filetxt.write('>> power spectrum value last mode = {} mm \n'.format(power_spectrum_array_without_mode0[-1]))
        filetxt.write('\n')
        
        filetxt.write('>> max power spectrum value = {} (-) \n'.format(np.max(power_spectrum_array_without_mode0)))
        filetxt.write('>> principal (associated) wavelength = {} mm \n'.format(principal_wavelength[0]))
        filetxt.write('\n')
        
        filetxt.close()

        print("\n>> folded sphere (nsteps{}, numericaltime {}) curvature LBO analysis .txt file was written.".format(args.step, numericaltime))  

