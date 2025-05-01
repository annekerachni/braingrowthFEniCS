import sys, os
sys.path.append(sys.path[0]) 
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0]))) # braingrowthFEniCS

import json
import math

# OPTION
########
which_quantity_to_compare = "cortex_areas" # "cortex_areas"; "wholebrain_volumes"; "cortex_volumes"
        
# dt
####
tGW1 = 28 # GW
tGW2 = 29 # GW
dt = (tGW2 - tGW1) * 604800 # in seconds (1GW=168h=604800s)

# CORTEX AREAS tGW1, tGW2 
#######################
if which_quantity_to_compare == "cortex_areas":
    print('\n')

    json_path = './metrics/compute_growth_rates/cortex_areas_dHCPvolume_21_29GW.json' # cortex_areas_dHCPsurface_21_29GW; cortex_areas_dHCPvolume_21_28_36GW
    
    with open(json_path, 'r') as cortex_areas_json_file:  
        cortex_areas_json_str = cortex_areas_json_file.read()
        cortex_areas_values_dict = json.loads(cortex_areas_json_str)
        area_tGW1 = cortex_areas_values_dict[str(tGW1)]
        area_tGW2 = cortex_areas_values_dict[str(tGW2)]
        
        ratio_A_tGW1_tGW2 = area_tGW2 / area_tGW1
        alphaTAN_area_tGW1_tGW2 = (math.sqrt(ratio_A_tGW1_tGW2) - 1)/ dt # [(m).s⁻¹] # Jg = area_tGW2 / area_tGW1 = theta² = (1 + alphaTAN * (gr * gm) * dt)²
        
        print('\n')
        print("alphaTAN_cortex_areas_{}_{} = {} s⁻¹".format(str(tGW1), str(tGW2), alphaTAN_area_tGW1_tGW2))

        print('\n')


# WHOLE BRAIN MESH VOLUMES tGW1, tGW2 
###################################
if which_quantity_to_compare == "wholebrain_volumes":
    print('\n')

    json_path = './metrics/compute_growth_rates/brain_volumes_21_29GW.json'

    with open(json_path, 'r') as volumes_json_file:  
        volumes_json_str = volumes_json_file.read()
        volumes_values_dict = json.loads(volumes_json_str)
        volume_tGW1 = volumes_values_dict[str(tGW1)]
        volume_tGW2 = volumes_values_dict[str(tGW2)]
        
        #dV_tGW1_tGW2 = volume_tGW2 - volume_tGW1
        #alphaTAN_volume_tGW1_tGW2 = dV_tGW1_tGW2 / dt # [(m³).s⁻¹]
        ratio_V_tGW1_tGW2 = volume_tGW2 / volume_tGW1
        alphaTAN_volume_tGW1_tGW2 = (ratio_V_tGW1_tGW2**(1/3) - 1)/ dt # [s⁻¹] # Jg = volume_tGW2 / volume_tGW1 = theta³ = (1 + alphaTAN * (gr * gm) * dt)³
        
        print("alphaTAN_brain_volumes_{}_{} = {} s⁻¹".format(str(tGW1), str(tGW2), alphaTAN_volume_tGW1_tGW2))

# CORTEX MESH VOLUMES tGW1, tGW2 
##############################
if which_quantity_to_compare == "cortex_volumes":
    print('\n')
    
    json_path = './metrics/compute_growth_rates/brain_cortex_volumes_21_29GW.json'

    with open(json_path, 'r') as cortex_volumes_json_file:  
        cortex_volumes_json_str = cortex_volumes_json_file.read()
        cortex_volumes_values_dict = json.loads(cortex_volumes_json_str)
        cortex_volume_tGW1 = cortex_volumes_values_dict['cortex mesh volume attGW1']
        cortex_volume_tGW2 = cortex_volumes_values_dict['cortex mesh volume attGW2']
        
        """
        # dV
        dV_cortex_tGW1_tGW2 = cortex_volume_tGW2 - cortex_volume_tGW1
        alphaTAN_cortex_volume_tGW1_tGW2 = dV_cortex_tGW1_tGW2 / dt # [(m³).s⁻¹]
        
        print("alphaTAN_brain_cortex_volumes_{}_{} = {} (m³).s⁻¹".format(str(tGW1), str(tGW2), alphaTAN_cortex_volume_tGW1_tGW2))
        
        print("\n")
        # d[V^(1/3)]
        dV_1_3_cortex_tGW1_tGW2 = (cortex_volume_tGW2)**(1/3) - (cortex_volume_tGW1)**(1/3)
        alphaTAN_cortex_volume_1_3_tGW1_tGW2 = dV_1_3_cortex_tGW1_tGW2 / dt # [(m).s⁻¹]
        
        print("alphaTAN_brain_cortex_volumes_{}_{} = {} (m).s⁻¹".format(str(tGW1), str(tGW2), alphaTAN_cortex_volume_1_3_tGW1_tGW2))
        """

        ratio_Vcortex_tGW1_tGW2 = cortex_volume_tGW2 / cortex_volume_tGW1
        alphaTAN_cortex_volume_tGW1_tGW2 = (ratio_Vcortex_tGW1_tGW2**(1/3) - 1)/ dt # [s⁻¹] # Jg = volume_tGW2 / volume_tGW1 = theta³ = (1 + alphaTAN * (gr * gm) * dt)³
        
        print("alphaTAN_brain_cortex_volumes_{}_{} = {} s⁻¹".format(str(tGW1), str(tGW2), alphaTAN_cortex_volume_tGW1_tGW2))

    # Compute mean cortical thickness H0
    ####################################
        
        H0_mean_tGW1 = cortex_volume_tGW1 / area_tGW1
        H0_mean_tGW2 = cortex_volume_tGW2 / area_tGW2