import sys, os
sys.path.append(sys.path[0]) 
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0]))) # braingrowthFEniCS

import fenics
from FEM_biomechanical_model import preprocessing
from metrics.geometrical import brain_external_area, brain_volume
from utils.converters import convert_meshformats
import json

# COMPUTE CORTEX AREA FOR ALL dHCP SURFACE MESHES 21 to 29GW 
############################################################

# dHCP atlas surface joint meshes
inputmeshes_dict = {21: "./data/dHCP_surface_21_to_29GW/joint_STL/fetal_week21_pial_surf.stl",
                    22: "./data/dHCP_surface_21_to_29GW/joint_STL/fetal_week22_pial_surf.stl",
                    23: "./data/dHCP_surface_21_to_29GW/joint_STL/fetal_week23_pial_surf.stl",
                    24: "./data/dHCP_surface_21_to_29GW/joint_STL/fetal_week24_pial_surf.stl",
                    25: "./data/dHCP_surface_21_to_29GW/joint_STL/fetal_week25_pial_surf.stl",
                    26: "./data/dHCP_surface_21_to_29GW/joint_STL/fetal_week26_pial_surf.stl",
                    27: "./data/dHCP_surface_21_to_29GW/joint_STL/fetal_week27_pial_surf.stl",
                    28: "./data/dHCP_surface_21_to_29GW/joint_STL/fetal_week28_pial_surf.stl",
                    29: "./data/dHCP_surface_21_to_29GW/joint_STL/fetal_week29_pial_surf.stl"
                    }

# output
cortical_areas = {}

# compute cortical area for each tGW
for tGW, inputmesh_path in inputmeshes_dict.items():

    # convert .stl into .xdmf
    inputmesh_name = inputmesh_path.split('.stl')[0]
    convert_meshformats.stl_to_xml_xdmf_2D(inputmesh_path, inputmesh_name + ".xdmf")

    # read .xdmf mesh
    mesh = fenics.Mesh()
    with fenics.XDMFFile(inputmesh_name + ".xdmf") as infile:
        infile.read(mesh)
            
    # convert mesh from mm into m (SI)
    mesh = preprocessing.converting_mesh_from_millimeters_into_meters(mesh)
            
    # bmesh
    #bmesh = fenics.BoundaryMesh(mesh, "exterior")
    
    # compute cortical surface
    area = brain_external_area.compute_mesh_external_surface(mesh)
    
    cortical_areas[tGW] = area
    #print('cortical area at {}GW = {} m'.format(tGW, area))  

# export json file 
output_path = './metrics/compute_growth_rates/cortex_areas_dHCPsurface_21_29GW.json'

with open(output_path, 'w') as cortex_areas_json_file:  
    json.dump(cortical_areas, cortex_areas_json_file)