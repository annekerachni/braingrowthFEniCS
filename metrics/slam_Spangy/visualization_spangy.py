import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pyvista as pv
#from numba import jit, prange


def plot_global_coefficients_and_bands_Allsignal(eigVal, grouped_spectrum, group_indices, coefficients):
    
    """Plot Plot Coefficients of the neurofunctional signal projection onto the Laplace-Beltrami eigenpairs basis & Histogram of the eigenpairs compacted bands"""
    ######## COLORMAP

    """
    nlevels = len(group_indices)
    #colors = np.zeros((nlevels, 4))
    highfrequency_band_number = int(nlevels/2) # e.g. for 7 bands basis --> 13 total gyri, sulci and low frequency bands --> 3 high frequency bands are kept
    #colors = np.zeros((2*highfrequency_band_number+1, 4))
    colors = np.zeros((2*highfrequency_band_number+1, 4)) # 4 colors for 7 bands (black + B4 + B5 + B6)
    color_variations_coef = np.linspace(0, 1, highfrequency_band_number)

    # low frequency bands  
    colors[0, :] = np.array([0, 0, 0, 1]) # to display in grey: np.array([0.75, 0.75, 0.75, 1])

    # sulci bands (number = int(nlevels/2) )  
    for i in prange(highfrequency_band_number):
        colors[i+1, :] = np.array([0, color_variations_coef[i], 1, 1]) 

    # gyri bands (number = int(nlevels/2) ) 
    for i in prange(highfrequency_band_number):
        colors[i+1+highfrequency_band_number, :] = np.array([1, color_variations_coef[i], 0, 1])

    # allocate dedicated color to each bar of the histogram
    colormap_sulci = [] # len = 7
    colormap_gyri = [] # len = 7

    # Sulci colormap
    for i in np.arange(0, nlevels-highfrequency_band_number):
        colormap_sulci.append(colors[0, :]) # B0, B-1, B-2, B-3
    for i in np.arange(0, highfrequency_band_number):
        colormap_sulci.append(colors[i+1, :]) # B-4, B-5, B-6

    # Gyri colormap
    for i in np.arange(0, nlevels-highfrequency_band_number):
        colormap_gyri.append(colors[0, :]) # B0, B1, B2, B3
    for i in np.arange(0, highfrequency_band_number):
        colormap_gyri.append(colors[i+1+highfrequency_band_number, :]) # B4, B5, B6
    
    ######## PLOTS WHOLE BRAIN SPECTRUM

    # Plot f2analyse projection (neurofunction spectrum) and compacted spectrum

    fig, (ax1, ax2) = plt.subplots(1, 2)
    coefficients_colors = [] # len = N (e.g. 1500 eigenpairs)

    ############## 1.NEUROFUNCTIONAL SIGNAL (e.g. mean curvature) PROJECTION COEFFICIENTS (number of coefficients = number of eigenpairs N)

    # Plot the sqrt of the eigVals divided by 2*np.pi against the coefficients in the first subplot
    band_last_eigenvalue = []
    for i in prange(len(group_indices)):
        band_last_eigenvalue.append(group_indices[i][1]) # i, first eigen value of the band Bi

    # coefficients belonging to band B0
    coefficients_colors.append(colormap_gyri[0])

    # coefficients belonging to other bands i.e. B1..B6 & B-1..B-6
    for i in np.arange(1, len(coefficients)): # len(coefficients) = N
        # trouver la bande correspondante à l'indice de l'eigenpair 
        j = 0
        while i > band_last_eigenvalue[j]:
            j = j+1
            if i < band_last_eigenvalue[j]:
                break
            #return j # number of the band in which the eigenvalue is

        # si coefficient[i] < 0 --> associer la couleur correspondant à la bande & sulcus
        if coefficients[i] <= 0:
            coefficients_colors.append(colormap_sulci[j])

        # si coefficient[i] > 0 --> associer la couleur correspondant à la bande & gyrus
        else:
            coefficients_colors.append(colormap_gyri[j]) 

    #ax1.plot(np.sqrt(eigVal/2*np.pi), coefficients, color=coefficients_colors)
    ax1.scatter(np.sqrt(eigVal/2*np.pi), coefficients, marker='+', s=10, linewidths=0.5, color=coefficients_colors)
    ax1.set_xlabel('Frequency (m⁻¹)')
    ax1.set_ylabel('Mean Curvature projection coefficients')

    ############## 2.COMPACTED FREQUENCY-BANDS

    #Barplot for the nlevels and grouped spectrum in the second subplot
    ax2.bar(np.arange(0, nlevels), grouped_spectrum.squeeze(), color=colormap_gyri)
    ax2.set_xlabel('Spangy frequency bands')
    ax2.set_ylabel('Power spectrum')

    plt.show()
    """

    return

def colormap_sulci_and_gyri(group_indices):

    nlevels = len(group_indices)
    #colors = np.zeros((nlevels, 4))
    highfrequency_band_number = int(nlevels/2) # e.g. for 7 bands basis --> 13 total gyri, sulci and low frequency bands --> 3 high frequency bands are kept
    #colors = np.zeros((2*highfrequency_band_number+1, 4))
    colors = np.zeros((2*highfrequency_band_number+1, 4)) # 4 colors for 7 bands (black + B4 + B5 + B6)
    color_variations_coef = np.linspace(0, 1, highfrequency_band_number)

    # low frequency bands  
    colors[0, :] = np.array([0, 0, 0, 1]) #--> black. / To display in grey: np.array([0.75, 0.75, 0.75, 1])

    # sulci bands (number = int(nlevels/2) )  
    for i in range(highfrequency_band_number):
        colors[i+1, :] = np.array([0, color_variations_coef[i], 1, 1]) 

    # gyri bands (number = int(nlevels/2) ) 
    for i in range(highfrequency_band_number):
        colors[i+1+highfrequency_band_number, :] = np.array([1, color_variations_coef[i], 0, 1])

    """ # gyri bands (number = int(nlevels/2) ) 
    for i in range(highfrequency_band_number):
        colors[i+1, :] = np.array([1, color_variations_coef[i], 0, 1])  """

    # allocate dedicated color to each bar of the histogram
    colormap_sulci = [] # len = 7
    colormap_gyri = [] # len = 7

    # Sulci colormap
    for i in np.arange(0, nlevels-highfrequency_band_number):
        colormap_sulci.append(colors[0, :]) # B0, B-1, B-2, B-3 --> black
    for i in np.arange(0, highfrequency_band_number):
        colormap_sulci.append(colors[i+1, :]) # B-4, B-5, B-6

    # Gyri colormap
    for i in np.arange(0, nlevels-highfrequency_band_number):
        colormap_gyri.append(colors[0, :]) # B0, B1, B2, B3 --> black
    for i in np.arange(0, highfrequency_band_number):
        colormap_gyri.append(colors[i+1+highfrequency_band_number, :]) # B4, B5, B6

    return colors, colormap_gyri, colormap_sulci


#@jit(parallel=True, forceobj=True)
def plot_global_SulciORGyri_coefficients_and_bands(subsignal, eigVal, nlevels, grouped_spectrum, group_indices, coefficients, colormap):
    """Plot Plot Coefficients of the neurofunctional signal projection onto the Laplace-Beltrami eigenpairs basis & Histogram of the eigenpairs compacted bands"""
    
    ######## PLOTS WHOLE BRAIN SPECTRUM

    # Plot f2analyse projection (neurofunction spectrum) and compacted spectrum

    fig, (ax1, ax2) = plt.subplots(1, 2)
    coefficients_colors = [] # len = N (e.g. 1500 eigenpairs)

    ############## 1.NEUROFUNCTIONAL SIGNAL (e.g. mean curvature) PROJECTION COEFFICIENTS (number of coefficients = number of eigenpairs N)

    # Plot the sqrt of the eigVals divided by 2*np.pi against the coefficients in the first subplot
    band_last_eigenvalue = []
    for i in range(len(group_indices)):
        band_last_eigenvalue.append(group_indices[i][1]) # i, first eigen value of the band Bi

    # coefficients belonging to band B0
    coefficients_colors.append(colormap[0])

    # coefficients belonging to other bands i.e. B1..B6 & B-1..B-6
    for i in np.arange(1, len(coefficients)): # len(coefficients) = N
        # trouver la bande correspondante à l'indice de l'eigenpair 
        """ if i == 0: # first eigenpair ~0
            j = 0 # j: band_number
            coefficients_colors.append(colormap_gyri[j]) """
        j = 0
        while i > band_last_eigenvalue[j]:
            j = j+1
            if i < band_last_eigenvalue[j]:
                break

        coefficients_colors.append(colormap[j])

    #ax1.plot(np.sqrt(eigVal/2*np.pi), coefficients, color=coefficients_colors)
    ax1.scatter(np.sqrt(eigVal/2*np.pi), coefficients, marker='+', s=10, linewidths=0.5, color=coefficients_colors)
    #ax1.scatter(np.sqrt(eigVal[1:]/2*np.pi), coefficients[1:], marker='+', s=10, linewidths=0.5, color=coefficients_colors[1:]) # remove B0 coefficient
    ax1.set_xlabel('Frequency (m⁻¹)')
    ax1.set_ylabel('Global coefficients from decomposition of ' + str(subsignal))

    ############## 2.COMPACTED FREQUENCY-BANDS

    #Barplot for the nlevels and grouped spectrum in the second subplot
    """ print(grouped_spectrum) """
    ax2.bar(np.arange(0, nlevels), grouped_spectrum.squeeze(), color=colormap)
    #ax2.bar(np.arange(1, nlevels), grouped_spectrum[1:].squeeze(), color=colormap[1:]) # remove B0
    ax2.set_xlabel('Spangy frequency bands')
    ax2.set_ylabel('Global power spectrum of '+ str(subsignal))

    #plt.show()

    return plt


def plot_local_dominant_bands(mesh3D, nlevels, loc_dom_band, colors):

    # Generalized function to plot local dominant bands (adapting to automatically computed nlevels)

    number_of_displayed_bands = len(np.unique(loc_dom_band)) # 12

    #colors = np.zeros((nlevels, 4))
    highfrequency_band_number = int(nlevels/2) # e.g. for 7 bands basis --> 13 total gyri, sulci and low frequency bands --> 3 high frequency bands are kept
    """ colors = np.zeros((2*highfrequency_band_number+1, 4))
    color_variations_coef = np.linspace(0, 1, highfrequency_band_number)

    # low frequency bands  
    colors[0, :] = np.array([0, 0, 0, 1]) # to display in grey: np.array([0.75, 0.75, 0.75, 1])

    # sulci bands (number = int(nlevels/2) )  
    for i in range(highfrequency_band_number):
        colors[i+1, :] = np.array([0, color_variations_coef[i], 1, 1]) 

    # gyri bands (number = int(nlevels/2) ) 
    for i in range(highfrequency_band_number):
        colors[i+1+highfrequency_band_number, :] = np.array([1, color_variations_coef[i], 0, 1])  """


    band_values = np.linspace(np.min(loc_dom_band), np.max(loc_dom_band), number_of_displayed_bands+1)  # 13 = 6 positive bands * 2 + Band 0 --> to generalize
    band_colors = np.empty((number_of_displayed_bands+1, 4))
    limit = np.max(loc_dom_band) - highfrequency_band_number + 1 # limit band number under which band is displayed in black

    band_colors[band_values < limit] = colors[0, :] # low frequency bands

    # associate one color per high frequency sulcus band 
    for i in range(highfrequency_band_number):
        band_colors[band_values == -(limit+i)] = colors[i+1, :]  

    # associate one color per high frequency gyrus band 
    for i in range(highfrequency_band_number):
        band_colors[band_values == (limit+i)] = colors[i+1+highfrequency_band_number, :]  

    localbands_colormap = ListedColormap(band_colors)
    #loc_dom_band = loc_dom_band.astype(int)

    p = pv.Plotter()
    p.add_mesh(mesh3D, scalars=loc_dom_band, show_edges=False, cmap=localbands_colormap, show_scalar_bar=False)
    p.add_text("Local Dominant Bands", font_size=14)
    p.add_scalar_bar('Band n°', fmt="%.0f") 
    p.show()

    return

