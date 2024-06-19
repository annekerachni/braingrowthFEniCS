import numpy as np
import matplotlib.pyplot as plt

import slam.io as sio
import slam.curvature as scurv

import spangy as spgy
import visualization_spangy as spgyviz

import sys, os
sys.path.append(os.path.dirname(sys.path[0])) # os.path.dirname(sys.path[0]): 'metrics' / sys.path.append(sys.path[0]): 'metrics/slam_Spangy' 
import slam_plot as splt

import math

# VISUALIZATION MODE
# ------------------
visualization = True

# LOAD MESH
# ---------
# mesh = sio.load_mesh("./data/example_mesh.gii")
mesh = sio.load_mesh('./data/dhcp_fetal_atlas/WM_Right/fetal.week36.right.wm.surf.gii' ) 

# theoritical expression of N (number of eigen modes) with Weyl conjecture --> https://fr.wikipedia.org/wiki/G%C3%A9om%C3%A9trie_spectrale
cortical_area = mesh.area
WL_resolution = 7 # resolution_WL = 7 mm (= 447mm/(lambda_B1)^6) --> Spangy, Germanaud et al. 2012, Figure 9 a)
lambda_n = 4 * math.pi**2 / (WL_resolution)**2 # --> Spangy, Germanaud et al. 2012, Figure 3 a)
N_theory_Weyl = cortical_area*lambda_n/(4*math.pi)

N = 1500 #1500 # number of eigenpairs to compute. Should be < to the number of mesh vertices
# Compute eigenpairs and mass matrix
eigVal, eigVects, lap_b = spgy.eigenpairs(mesh, N)

# CURVATURE
# ---------
PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = \
    scurv.curvatures_and_derivatives(mesh)
mean_curv = 0.5 * (PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])

# Plot of mean curvature on the mesh
if visualization == True:
      visb_sc = splt.visbrain_plot(
      mesh=mesh,
      tex=mean_curv,
      caption='Mean Curvature',
      cmap='jet')
      visb_sc.preview()

# Export of mean curvature to .png
curvature_scene_viewer = splt.pyglet_plot(mesh, values=mean_curv, color_map='jet',
                                          plot_colormap=True, caption='Mean Curvature',
                                          alpha_transp=255, background_color=None,
                                          default_color=[100, 100, 100, 200], cmap_bounds=None)
splt.save_image(curvature_scene_viewer[0], './metrics/slam_Spangy/dHCP_fetal_RESULTS/36GW/dHCP_pial_Right_36GW_MeanCurvature.png')
plt.close()


# WHOLE BRAIN SPECTRUM
# --------------------
grouped_spectrum, group_indices, coefficients, nlevels = spgy.spectrum_reindexed(mean_curv, lap_b,
                                                              eigVects, eigVal)

mean_curv_sulci = np.zeros((mean_curv.shape))
mean_curv_sulci[mean_curv <= 0] = mean_curv[mean_curv <= 0]
grouped_spectrum_sulci, group_indices_sulci, coefficients_sulci, nlevels_sulci = spgy.spectrum_reindexed(mean_curv_sulci, lap_b,
                                                              eigVects, eigVal)

mean_curv_gyri = np.zeros((mean_curv.shape))
mean_curv_gyri[mean_curv > 0] = mean_curv[mean_curv > 0]
grouped_spectrum_gyri, group_indices_gyri, coefficients_gyri, nlevels_gyri = spgy.spectrum_reindexed(mean_curv_gyri, lap_b,
                                                              eigVects, eigVal)

# a. Whole brain parameters
mL_in_MM3 = 1000
CM2_in_MM2 = 100
volume = mesh.volume
surface_area = mesh.area
afp = np.sum(grouped_spectrum[1:])

if visualization == True:
      print('**\n a. Whole brain parameters **')
      print('Volume = %d mL, Area = %d cm², Analyze Folding Power = %f,' %
            (np.floor(volume / mL_in_MM3), np.floor(surface_area / CM2_in_MM2), afp))

      # b. Band number of parcels
      print('**\n b. Band number of parcels **')
      print('B4 = %f, B5 = %f, B6 = %f' % (0, 0, 0))

      # c. Band power
      print('**\n c. Band power **')
      print('B4 = %f, B5 = %f, B6 = %f' %
            (grouped_spectrum[4], grouped_spectrum[5],
            grouped_spectrum[6]))

      # d. Band relative power
      print('**\n d. Band relative power **')
      print('B4 = %0.5f, B5 = %0.5f , B6 = %0.5f' %
            (grouped_spectrum[4] / afp, grouped_spectrum[5] / afp,
            grouped_spectrum[6] / afp))

# Export global power spectrum
# ----------------------------
# Plot coefficients and bands for all mean curvature signal
""" spgyviz.plot_global_coefficients_and_bands(eigVal, grouped_spectrum, group_indices, coefficients) """
fig, (ax1, ax2) = plt.subplots(1, 2)
# Plot the sqrt of the eigVals divided by 2*np.pi against the coefficients in
# the first subplot

ax1.scatter(np.sqrt(eigVal/2*np.pi), coefficients, marker='+', s=10, linewidths=0.5) 

#ax1.plot(np.sqrt(eigVal[1:]) / (2 * np.pi), coefficients[1:]) # remove B0 coefficients
#ax1.scatter(np.sqrt(eigVal[1:]/2*np.pi), coefficients[1:], marker='+', s=10, linewidths=0.5) # remove B0 coefficients
ax1.set_xlabel('Frequency (m⁻¹)')
ax1.set_ylabel('Coefficients')

# Barplot for the nlevels and grouped spectrum in the second subplot
# Barplot in ax2 between nlevels and grouped spectrum
print(grouped_spectrum)
ax2.bar(np.arange(0, nlevels), grouped_spectrum) 
#ax2.bar(np.arange(1, nlevels), grouped_spectrum[1:]) # remove B0
ax2.set_xlabel('Spangy Frequency Bands')
ax2.set_ylabel('Power Spectrum')

plt.savefig('./metrics/slam_Spangy/dHCP_fetal_RESULTS/36GW/dHCP_pial_Right_36GW_PowerSpectrum.png', bbox_inches='tight')

# Export gyri & sulci resp. power spectrum
# ----------------------------------------
# colormap for sulci and gyri
colors, colormap_gyri, colormap_sulci = spgyviz.colormap_sulci_and_gyri(group_indices)

# Plot coefficients and bands for Sulci
plt = spgyviz.plot_global_SulciORGyri_coefficients_and_bands('mean_curv<=0', eigVal, nlevels_sulci, grouped_spectrum_sulci, group_indices_sulci, coefficients_sulci, colormap_sulci)
plt.savefig('./metrics/slam_Spangy/dHCP_fetal_RESULTS/36GW/dHCP_pial_Right_36GW_PowerSpectrum_Sulci.png', bbox_inches='tight')

# Plot coefficients and bands for Gyri

plt = spgyviz.plot_global_SulciORGyri_coefficients_and_bands('mean_curv>0',eigVal, nlevels_gyri, grouped_spectrum_gyri, group_indices_gyri, coefficients_gyri, colormap_gyri)
plt.savefig('./metrics/slam_Spangy/dHCP_fetal_RESULTS/36GW/dHCP_pial_Right_36GW_PowerSpectrum_Gyri.png', bbox_inches='tight')

# LOCAL SPECTRAL BANDS
# --------------------
loc_dom_band, frecomposed = spgy.local_dominance_map(coefficients, mean_curv,
                                                     nlevels, group_indices,
                                                     eigVects)

if visualization == True:
      # Plot of spectral dominant bands on the mesh
      visb_sc = splt.visbrain_plot(mesh=mesh, tex=loc_dom_band,
                              caption='Local Dominant Band', cmap='jet')
      visb_sc.preview()

      #spgyviz.plot_local_dominant_bands(mesh, nlevels, loc_dom_band) 
      """spgyviz.plot_local_dominant_bands(mesh, nlevels, loc_dom_band, colors)"""


# Export of local dominant band projected onto mesh to .png
# ---------------------------------------------------------
LocalDominantBands_scene_viewer = splt.pyglet_plot(mesh, values=loc_dom_band, color_map='jet',
                                                   plot_colormap=True, caption='Local Dominant Band',
                                                   alpha_transp=255, background_color=None,
                                                   default_color=[100, 100, 100, 200], cmap_bounds=None)
splt.save_image(LocalDominantBands_scene_viewer[0], './metrics/slam_Spangy/dHCP_fetal_RESULTS/36GW/dHCP_pial_Right_36GW_LocalDominantBands.png')

# Export Spangy analysis (.txt)
# -----------------------------
output_path = './metrics/slam_Spangy/dHCP_fetal_RESULTS/36GW/dHCP_pial_Right_36GW_SPANGY_ANALYSIS.txt'
    
filetxt = open(output_path, "w")

filetxt.write('###################\n')
filetxt.write('# Spangy analysis #\n')
filetxt.write('###################\n')

# Mesh info
filetxt.write('** Mesh infos **\n')
filetxt.write('Volume = %d mL, Cortex Area = %d cm², Analyze Folding Power = %f,' %
      (np.floor(volume / mL_in_MM3), np.floor(surface_area / CM2_in_MM2), afp))
filetxt.write('\n')

# Theoritical number of eigen modes N_Weyl
filetxt.write('** Theoritical number of eigen modes considering Weyl theory **\n')
filetxt.write('N_Weyl: {}\n'.format(N_theory_Weyl))
filetxt.write('\n')

# Power spectrum of "folding bands" B4, B5, B6
filetxt.write('** Band power **\n')
filetxt.write('B4 = %f, B5 = %f, B6 = %f' %
              (grouped_spectrum[4], grouped_spectrum[5], grouped_spectrum[6]))
filetxt.write('\n')

filetxt.write('** Band relative power **\n')
filetxt.write('B4 = %f, B5 = %f, B6 = %f' %
              (grouped_spectrum[4] / afp, grouped_spectrum[5] / afp, grouped_spectrum[6] / afp))
filetxt.write('\n')

filetxt.close()

# Save eigenVals and eigen modes repartition in the Bands (.csv)
# --------------------------------------------------------------
np.savetxt('./metrics/slam_Spangy/dHCP_fetal_RESULTS/36GW/dHCP_pial_Right_36GW_SPANGY_ANALYSIS_group_indices.csv', group_indices, delimiter=",")
np.savetxt('./metrics/slam_Spangy/dHCP_fetal_RESULTS/36GW/dHCP_pial_Right_36GW_SPANGY_ANALYSIS_eigenVals.csv', eigVal, delimiter=",")

