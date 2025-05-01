import json
import os, sys

sys.path.append(sys.path[0]) # alphaTAN_homogeneous_from_MRI
sys.path.append(os.path.dirname(os.path.dirname((os.path.dirname(sys.path[0]))))) # braingrowthFEniCS

# OPTION
########
which_quantity_to_compare = "cortex_volumes" # "cortex_areas"; "wholebrain_volumes"; "cortex_volumes"
        
# dt
####
dt_21_28 = (28 - 21) * 604800 # 1GW=168h=604800s
dt_21_36 = (36 - 21) * 604800 

# CORTEX AREAS 21, 28 & 36GW 
############################
if which_quantity_to_compare == "cortex_areas":
    print('\n')
    
    json_path = './metrics/volumes_21_28_36GW/cortex_areas_21_28_36GW.json' 
    
    with open(json_path, 'r') as cortex_areas_json_file:  
        cortex_areas_json_str = cortex_areas_json_file.read()
        cortex_areas_values_dict = json.loads(cortex_areas_json_str)
        area_21GW = cortex_areas_values_dict['21']
        area_28GW = cortex_areas_values_dict['28']
        area_36GW = cortex_areas_values_dict['36']
        
        dA_21_28 = area_28GW - area_21GW
        alphaTAN_area_21_28 = dA_21_28 / dt_21_28 # [(m²).s⁻¹]
        
        dA_21_36 = area_36GW - area_21GW   
        alphaTAN_area_21_36 = dA_21_36 / dt_21_36 # [(m²).s⁻¹] 
        
        print('\n')
        print("alphaTAN_area_21_28 = {} (m²).s⁻¹ \nalphaTAN_area_21_36 = {} (m²).s⁻¹".format(alphaTAN_area_21_28, alphaTAN_area_21_36))

        print('\n')

# WHOLE BRAIN MESH VOLUMES 21, 28 & 36GW 
########################################
if which_quantity_to_compare == "wholebrain_volumes":
    print('\n')

    json_path = './metrics/volumes_21_28_36GW/volume_21_28_36GW.json'

    with open(json_path, 'r') as volumes_json_file:  
        volumes_json_str = volumes_json_file.read()
        volumes_values_dict = json.loads(volumes_json_str)
        volume_21GW = volumes_values_dict['21']
        volume_28GW = volumes_values_dict['28']
        volume_36GW = volumes_values_dict['36']
        
        dV_21_28 = volume_28GW - volume_21GW
        alphaTAN_volume_21_28 = dV_21_28 / dt_21_28 # [(m³).s⁻¹]
        
        dV_21_36 = volume_36GW - volume_21GW
        alphaTAN_volume_21_36 = dV_21_36 / dt_21_36 # [(m³).s⁻¹]
        
        print("alphaTAN_volume_21_28 = {} (m³).s⁻¹ \nalphaTAN_volume_21_36 = {} (m³).s⁻¹".format(alphaTAN_volume_21_28, alphaTAN_volume_21_36))

# CORTEX MESH VOLUMES 21, 28 & 36GW 
####################################
if which_quantity_to_compare == "cortex_volumes":
    print('\n')
    
    json_path = './metrics/volumes_21_28_36GW/volume_wholebrain_cortex_21_28_36GW.json'

    with open(json_path, 'r') as cortex_volumes_json_file:  
        cortex_volumes_json_str = cortex_volumes_json_file.read()
        cortex_volumes_values_dict = json.loads(cortex_volumes_json_str)
        cortex_volume_21GW = cortex_volumes_values_dict['cortex mesh volume at21GW']
        cortex_volume_28GW = cortex_volumes_values_dict['cortex mesh volume at28GW']
        cortex_volume_36GW = cortex_volumes_values_dict['cortex mesh volume at36GW']
        
        # dV
        dV_cortex_21_28 = cortex_volume_28GW - cortex_volume_21GW
        alphaTAN_cortex_volume_21_28 = dV_cortex_21_28 / dt_21_28 # [(m³).s⁻¹]
        
        dV_cortex_21_36 = cortex_volume_36GW - cortex_volume_21GW
        alphaTAN_cortex_volume_21_36 = dV_cortex_21_36 / dt_21_36 # [(m³).s⁻¹]
        
        print("alphaTAN_cortex_volume_21_28 = {} (m³).s⁻¹ \nalphaTAN_cortex_volume_21_36 = {} (m³).s⁻¹".format(alphaTAN_cortex_volume_21_28, alphaTAN_cortex_volume_21_36))
        
        print("\n")
        # d[V^(1/3)]
        dV_1_3_cortex_21_28 = (cortex_volume_28GW)**(1/3) - (cortex_volume_21GW)**(1/3)
        alphaTAN_cortex_volume_1_3_21_28 = dV_1_3_cortex_21_28 / dt_21_28 # [(m).s⁻¹]
        
        dV_1_3_cortex_21_36 = (cortex_volume_36GW)**(1/3) - (cortex_volume_21GW)**(1/3)
        alphaTAN_cortex_volume_1_3_21_36 = dV_1_3_cortex_21_36 / dt_21_36 # [(m).s⁻¹]
        
        print("alphaTAN_cortex_volume1/3_21_28 = {} (m).s⁻¹ \nalphaTAN_cortex_volume1/3_21_36 = {} (m).s⁻¹".format(alphaTAN_cortex_volume_1_3_21_28, alphaTAN_cortex_volume_1_3_21_36))
        

    # Compute mean cortical thickness H0
    ####################################
        
        H0_mean_21GW = cortex_volume_21GW / area_21GW
        H0_mean_28GW = cortex_volume_28GW / area_28GW
        H0_mean_36GW = cortex_volume_36GW / area_36GW   