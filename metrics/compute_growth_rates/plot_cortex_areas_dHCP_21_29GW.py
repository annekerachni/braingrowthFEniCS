import sys, os
sys.path.append(sys.path[0]) 
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0]))) # braingrowthFEniCS

import json
import matplotlib.pyplot as plt
import numpy as np

# ouput folder to save plot figure
##################################
output_folder_path = './metrics/compute_growth_rates/'

# read json file
################ 
cortex_areas_dHCP_surface_21_29GW_path = './metrics/compute_growth_rates/cortex_areas_dHCPsurface_21_29GW.json' 
cortex_areas_dHCP_volume_21_29GW_path = './metrics/compute_growth_rates/cortex_areas_dHCPvolume_21_29GW.json' 

with open(cortex_areas_dHCP_surface_21_29GW_path, 'r') as cortex_areas_json_file:  
    cortex_areas_json_str = cortex_areas_json_file.read()
    cortex_areas_values_dict = json.loads(cortex_areas_json_str) # in m2

    with open(cortex_areas_dHCP_volume_21_29GW_path, 'r') as cortex_areas_dHCPvolume_json_file:  
        cortex_areas_dHCPvolume_json_str = cortex_areas_dHCPvolume_json_file.read()
        cortex_areas_dHCPvolume_values_dict = json.loads(cortex_areas_dHCPvolume_json_str) # in m2
    
    #area_21GW = cortex_areas_values_dict['21']
    #area_22GW = cortex_areas_values_dict['22']
    
    # save .svg
    ###########
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # dHCP atlas surface meshes
    tGW_dHCPsurface = np.array(list(cortex_areas_values_dict.keys()), dtype=float)
    cortex_areas_dHCPsurface = np.array(list(cortex_areas_values_dict.values()), dtype=float) * 1e04 # * 1e04: cortex area  from m2 into cm2
    ax.scatter(tGW_dHCPsurface, cortex_areas_dHCPsurface, s=50, c='royalblue', marker='.', label='dHCP surface atlas') # s:marker size - linewidth=3 - edgecolors='none'
    ax.plot(tGW_dHCPsurface, cortex_areas_dHCPsurface, '-', color='royalblue', linewidth=0.5, alpha=0.5)

    # dHCP atlas volume meshes
    tGW_dHCPvolume = np.array(list(cortex_areas_dHCPvolume_values_dict.keys()), dtype=float)
    cortex_areas_dHCPvolume = np.array(list(cortex_areas_dHCPvolume_values_dict.values()), dtype=float) * 1e04 # * 1e04: cortex area  from m2 into cm2
    ax.scatter(tGW_dHCPvolume, cortex_areas_dHCPvolume, s=50, c='deeppink', marker='.', label='dHCP volume atlas') # s:marker size - linewidth=3 - edgecolors='none'
    ax.plot(tGW_dHCPvolume, cortex_areas_dHCPvolume, '-', color='deeppink', linewidth=0.5, alpha=0.5)

    # results from xx.xx 2016
    """
    tGW_to_volumes_of_tallinen2016 = {"22":60, "29":150, "34":240}
    GI_TTallinen2016 = np.array(list(tGW_to_volumes_of_tallinen2016.keys()), dtype=float)
    volumes_TTallinen2016 = np.array(list(tGW_to_volumes_of_tallinen2016.values()), dtype=float)
    ax.scatter(volumes_TTallinen2016, GI_TTallinen2016, s=20, c='purple', marker='x', label='T.Tallinen et al. 2016') # s:marker size - linewidth=3 - edgecolors='none'
    ax.plot(volumes_TTallinen2016, GI_TTallinen2016, '-', color='purple', linewidth=0.5, alpha=0.5)
    """

    # simulation results meshes
    """
    cortical_areas_V2 = []
    for ar in cortical_areas:
        ar = ar * 10
        cortical_areas_V2.append(vol)
        
    ax.scatter(times_dHCPsurfaceatlas, cortical_areas_V2, s=50, c='blue', marker='+', label='simulation results') 
    ax.plot(times_dHCPsurfaceatlas, cortical_areas_V2, '-', color='blue', linewidth=0.5, alpha=0.5)
    """

    ax.set_xlim(21, 29.5)
    ax.set_ylim(50, 280) # ax.set_ylim(100, 250) for dHCP meshes from 21 to 28GW; ax.set_ylim(100, 510) from 21 to 36GW

    #ax.set_title('dHCP surface atlas meshes (21 to 29GW)')
    ax.legend()

    ax.set_xlabel("Gestational time (GW)") 
    ax.set_ylabel("Brain cortex area (cm2)")

    plt.savefig(output_folder_path + "tGW_to_cortex_area_dHCPsurface_vs_dHCPvolume.svg") 

    #plt.show()