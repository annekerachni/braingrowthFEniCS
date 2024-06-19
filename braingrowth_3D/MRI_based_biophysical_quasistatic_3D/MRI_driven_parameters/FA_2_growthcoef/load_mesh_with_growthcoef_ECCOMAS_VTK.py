import fenics
import argparse
import json
import meshio
import matplotlib.pyplot as plt
import numpy as np
import math
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0]))) 

from FEM_biomechanical_model_quasistatic import preprocessing
from MRI_driven_parameters.FA_2_growthcoef import FA_to_growth_law

# Tangential growth from FA
###########################

def normalize_FA(path_inputmeshloaded_with_FA_VTK):
    """
    Args:
    fa: fenics.Function(S)
    normalized_fa: fenics.Function(S)
    tGW: "21GW"
    t_simu: 0.
    path_inputmeshloaded_with_SegLabels_VTK: .vtk mesh file loaded with segmentation labels, collected from MRI data
    vertex2dofs_S: vertex --> scalar FEniCS DOF
    
    Returns: grTAN FEniCS scalar FEM function (1. in Cortex and 0. in Core)
    """
    
    # Defining tangential growth coefficient 'alphaTAN' in correlation to FA nodal value. (FA ++ => neuronal maturation -- => neuron matures => tangential growth ++)
    mesh_loaded_with_FA = meshio.read(path_inputmeshloaded_with_FA_VTK)
    FA = mesh_loaded_with_FA.point_data["FA"] # nodal array 
    max_FA = np.max(FA)
    
    fa = FA.copy()
    normalized_fa = FA.copy()/max_FA
        
    return fa, normalized_fa # numpy arrays


def tangential_growth_coef_from_FA(linear_coef, normalized_fa):    
    
    """
    Args: 
    - normalized_fa : numpy array
    - alphaTAN: numpy array
    """  
    
    # Defining tangential growth coefficient 'alphaTAN' in correlation to FA nodal value. (FA ++ => neuronal maturation -- => neuron matures => tangential growth ++)
    alphaTAN = np.zeros(len(normalized_fa), dtype=np.float64)
    alphaTAN[:] = FA_to_growth_law.FA_to_tangential_growth_law_linearrelationship(linear_coef, normalized_fa[:] ) # FA should be a float between 0. and 1. => normalized_fa = fa/np.max(fa)  + growth is a linear function of FA. (here we test alphaTAN = 5*FA)
    #alphaTAN[:] = FA_to_tangential_growth_law_gompertz(linear_coef, normalized_fa[:] )
    #alphaTAN[:] = FA_to_tangential_growth_law_exponential(linear_coef, normalized_fa[:] )
        
    return alphaTAN


# Radial growth
###############
def radial_growth_coef(): 

    alphaRAD = 0
    """
    grRAD = 1.
    alphaRAD = 0. # --> no radial growth
    dg_RAD.assign( fenics.project(grRAD * alphaRAD *  ((1 - alphaF) * dt), S) ) # = 0.0
    """
    
    return alphaRAD


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Load "FA", "normalized_FA" and FA-based tangential cortical growth rate "alphaTAN" onto input mesh nodes')
    
    parser.add_argument('-i', '--input', help='Input mesh path (.xml, .xdmf)', type=str, required=False, 
                        default='./MRI_driven_parameters/meshes_loaded_with_MRI_data/dhcpRight21GW_masked_20000faces_98000tets_refinedWidthCoef5/dhcpRight21GW_masked_20000faces_98000tets_refinedWidthCoef5.vtk') 
                        
    parser.add_argument('-fa', '--fractionalanisotropy', help='Path to the mesh loaded with FA values onto nodes (.vtk)', type=json.loads, required=False, 
    default={21: './MRI_driven_parameters/meshes_loaded_with_MRI_data/dhcpRight21GW_masked_20000faces_98000tets_refinedWidthCoef5/dhcpRight21GW_masked_20000faces_98000tets_refinedWidthCoef5.vtk'}) # VTK containing nodal values: T2 intensity, Segmentation and raw FA
    
    parser.add_argument('-o', '--outputfile', help='Path to output folder', type=str, required=False, 
                        default='./MRI_driven_parameters/meshes_loaded_with_MRI_data/dhcpRight21GW_masked_20000faces_98000tets_refinedWidthCoef5/dhcpRight21GW_masked_20000faces_98000tets_refinedWidthCoef5_') 
    
    args = parser.parse_args() 
    
    
    # Input mesh
    ############

    # Mesh
    # ----
    # mesh & boundary mesh
    print("\nimporting .vtk mesh...")
    mesh = meshio.read(args.input)    
        
    # Get FA from .vtk
    dict_FA_VTK = args.fractionalanisotropy
    for i in dict_FA_VTK:
        tGW = i
        path_inputmeshloaded_with_FA_VTK = dict_FA_VTK[i]

    # Normalize FA (FA value should be between 0. and 1.)
    fa, normalized_fa = normalize_FA(path_inputmeshloaded_with_FA_VTK) # numpy arrays

    # Compute tangential growth coefficient from FA values
    linear_coef = 5.
    alphaTAN = tangential_growth_coef_from_FA(linear_coef, normalized_fa) # numpy array

    # Export to .vtk mesh loaded with normalized FA and FA-based alphaTAN
    node_textures = {} 
    node_textures['normalized_FA'] = normalized_fa 
    node_textures['alphaTAN_FAbased'] = alphaTAN 

    for key in node_textures:
        mesh.point_data[key] = node_textures[key]
        
    mesh.write(args.outputfile + "loaded_with_normalizedFA_and_alphaTAN_at{}GW.vtk".format(tGW)) 
    
    