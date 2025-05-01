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
json_path = './metrics/volumes_21_28_36GW/volume_21_28_36GW.json'

whole_brain_volumes = []

with open(json_path, 'r') as volumes_json_file:  
    volumes_json_str = volumes_json_file.read()
    volumes_values_dict = json.loads(volumes_json_str)
    volume_21GW = volumes_values_dict['21']
    volume_28GW = volumes_values_dict['28']
    volume_36GW = volumes_values_dict['36']
    
    whole_brain_volumes.append(volume_21GW)
    whole_brain_volumes.append(volume_28GW)
    whole_brain_volumes.append(volume_36GW)
    
# CORTEX MESH VOLUMES 21, 28 & 36GW 
###################################  
json_path = './metrics/volumes_21_28_36GW/volume_wholebrain_cortex_21_28_36GW.json'

cortex_volumes =  []

with open(json_path, 'r') as cortex_volumes_json_file:  
    cortex_volumes_json_str = cortex_volumes_json_file.read()
    cortex_volumes_values_dict = json.loads(cortex_volumes_json_str)
    cortex_volume_21GW = cortex_volumes_values_dict['cortex mesh volume at21GW']
    cortex_volume_28GW = cortex_volumes_values_dict['cortex mesh volume at28GW']
    cortex_volume_36GW = cortex_volumes_values_dict['cortex mesh volume at36GW']
    
    cortex_volumes.append(cortex_volume_21GW)
    cortex_volumes.append(cortex_volume_28GW)
    cortex_volumes.append(cortex_volume_36GW)
    
    # Compute mean cortical thickness H0
    ####################################
    H0_mean_21GW = cortex_volume_21GW / area_21GW
    H0_mean_28GW = cortex_volume_28GW / area_28GW
    H0_mean_36GW = cortex_volume_36GW / area_36GW   
        
# CORE MESH VOLUMES 21, 28 & 36GW 
#################################
Core_volume_21GW = whole_brain_volumes[0] - cortex_volumes[0]
Core_volume_28GW = whole_brain_volumes[1] - cortex_volumes[1]
Core_volume_36GW = whole_brain_volumes[2] - cortex_volumes[2]

print("Core_volume_21GW = {} m3".format(Core_volume_21GW))
print("Core_volume_28GW = {} m3".format(Core_volume_28GW))
print("Core_volume_36GW = {} m3".format(Core_volume_36GW))

dV_21_28_Core = Core_volume_28GW - Core_volume_21GW
dV_21_36_Core = Core_volume_36GW - Core_volume_21GW

print("dV_21_28_Core = {} m3".format(dV_21_28_Core))
print("dV_21_36_Core = {} m3".format(dV_21_36_Core))

# d[V^(1/3)]
dV_1_3_Core_21_28 = (Core_volume_28GW)**(1/3) - (Core_volume_21GW)**(1/3)
alphaTAN_Core_volume_1_3_21_28 = dV_1_3_Core_21_28 / dt_21_28 # [(m).s⁻¹]

dV_1_3_Core_21_36 = (Core_volume_36GW)**(1/3) - (Core_volume_21GW)**(1/3)
alphaTAN_Core_volume_1_3_21_36 = dV_1_3_Core_21_36 / dt_21_36 # [(m).s⁻¹]

print("\nalphaTAN_Core_volume1/3_21_28 = {} (m).s⁻¹ \nalphaTAN_Core_volume1/3_21_36 = {} (m).s⁻¹".format(alphaTAN_Core_volume_1_3_21_28, alphaTAN_Core_volume_1_3_21_36))    

# V ~ 4/3 * Pi * R³
import math
radius_21GW = (3 / (4 * math.pi) * Core_volume_21GW)**(1/3)
radius_28GW = (3 / (4 * math.pi) * Core_volume_28GW)**(1/3)
radius_36GW = (3 / (4 * math.pi) * Core_volume_36GW)**(1/3)

dR_Core_21_28 = radius_28GW - radius_21GW
dR_Core_21_36 = radius_36GW - radius_21GW

alphaTAN_Core_radius_21_28 = dR_Core_21_28 / dt_21_28
alphaTAN_Core_radius_21_36 = dR_Core_21_36 / dt_21_36

print("\nalphaTAN_Core_radius_21_28 = {} (m).s⁻¹ \nalphaTAN_radius_21_36 = {} (m).s⁻¹".format(alphaTAN_Core_radius_21_28, alphaTAN_Core_radius_21_36))    
