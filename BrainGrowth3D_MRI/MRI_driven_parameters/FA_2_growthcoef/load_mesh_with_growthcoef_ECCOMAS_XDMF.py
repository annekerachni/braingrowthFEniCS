import fenics
import argparse
import json
import meshio
#import matplotlib.pyplot as plt
import numpy as np
import math
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(sys.path[0])))) 

from FEM_biomechanical_model import preprocessing
from MRI_driven_parameters.FA_2_growthcoef import FA_to_growth_law


# Tangential growth from FA
###########################

def normalize_FA(fa, normalized_fa, vertex2dofs_S, mesh_loaded_with_FA):
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
    #mesh_loaded_with_FA = meshio.read(path_inputmeshloaded_with_FA_VTK)
    FA = mesh_loaded_with_FA.point_data["FA"] # nodal array 
    max_FA = np.max(FA)
    
    for vertex, scalarDOF in enumerate(vertex2dofs_S):
        fa.vector()[scalarDOF] = FA[vertex]
        normalized_fa.vector()[scalarDOF] = FA[vertex]/max_FA
        
    return fa, normalized_fa # FEniCS scalar functions


def tangential_growth_coef_from_linear_FA(alphaTAN, vertex2dofs_S, normalized_fa):    
    
    """
    Args: 
    - vertex2dofs_S: vertex --> scalar FEniCS DOF
    - normalized_fa : fenics.Function(S) 
    """  
    
    # Defining tangential growth coefficient 'alphaTAN' in correlation to FA nodal value. (FA ++ => neuronal maturation -- => neuron matures => tangential growth ++)
    for vertex, scalarDOF in enumerate(vertex2dofs_S):
        #alphaTAN.vector()[scalarDOF] = FA_to_growth_law.FA_to_tangential_growth_law_linearrelationship(normalized_fa.vector()[scalarDOF] ) # FA should be a float between 0. and 1. => normalized_fa = fa/np.max(fa)  + growth is a linear function of FA. (here we test alphaTAN = 5*FA)
        #alphaTAN.vector()[scalarDOF] = FA_to_tangential_growth_law_gompertz(linear_coef, normalized_fa.vector()[scalarDOF])
        #alphaTAN.vector()[scalarDOF] = FA_to_tangential_growth_law_exponential(linear_coef, normalized_fa.vector()[scalarDOF])
        alphaTAN.vector()[scalarDOF] = normalized_fa.vector()[scalarDOF]
        
    return alphaTAN

def tangential_growth_coef_from_exp_minus_FA(alphaTAN, vertex2dofs_S, normalized_fa):    
    
    """
    Args: 
    - vertex2dofs_S: vertex --> scalar FEniCS DOF
    - normalized_fa : fenics.Function(S) 
    """  
    
    for vertex, scalarDOF in enumerate(vertex2dofs_S):
        #alphaTAN.vector()[scalarDOF] = FA_to_growth_law.FA_to_tangential_growth_law_exponential_minusFA_Relationship(normalized_fa.vector()[scalarDOF] ) 
        alphaTAN.vector()[scalarDOF] = math.exp(-normalized_fa.vector()[scalarDOF] )
        
    return alphaTAN

def tangential_growth_coef_from_1_over_1_plus_FA(alphaTAN, vertex2dofs_S, normalized_fa):    
    
    """
    Args: 
    - vertex2dofs_S: vertex --> scalar FEniCS DOF
    - normalized_fa : fenics.Function(S) 
    """  
    
    for vertex, scalarDOF in enumerate(vertex2dofs_S):
        #alphaTAN.vector()[scalarDOF] = FA_to_growth_law.FA_to_tangential_growth_law_1_over_1_plus_FA_Relationship(normalized_fa.vector()[scalarDOF] ) 
        alphaTAN.vector()[scalarDOF] = (1 / (1 + normalized_fa.vector()[scalarDOF]) )
        
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
                        default='./data/dHCP_raw/dhcpRight21GW_masked_10000faces_48000tets_refinedWidthCoef5.xdmf') 
                        
    parser.add_argument('-fa', '--fractionalanisotropy', help='Path to the mesh loaded with FA values onto nodes (.vtk)', type=json.loads, required=False, 
    default={21: './MRI_driven_parameters/meshes_loaded_with_MRI_data/dhcpRight21GW_masked_10000faces_48000tets_refinedWidthCoef5/dhcpRight21GW_masked_10000faces_48000tets_refinedWidthCoef5.vtk'}) # VTK containing nodal values: T2 intensity, Segmentation and raw FA
    
    parser.add_argument('-o', '--outputfile', help='Path to output folder', type=str, required=False, 
                        default='./data/dHCP_raw/dhcpRight21GW_masked_10000faces_48000tets_refinedWidthCoef5_') 
    
    args = parser.parse_args() 
    
    
    # Input mesh
    ############

    # Mesh
    # ----
    # mesh & boundary mesh
    print("\nimporting mesh...")
    inputmesh_path = args.input
    inputmesh_format = inputmesh_path.split('.')[-1]

    if inputmesh_format == "xml":
        mesh = fenics.Mesh(inputmesh_path)

    elif inputmesh_format == "xdmf":
        mesh = fenics.Mesh()
        with fenics.XDMFFile(inputmesh_path) as infile:
            infile.read(mesh)
            
    bmesh = fenics.BoundaryMesh(mesh, "exterior") # bmesh at t=0.0 (cortex envelop)

    # mesh characteristics
    characteristics0 = preprocessing.compute_geometrical_characteristics(mesh, bmesh)
    center_of_gravity0 = preprocessing.compute_center_of_gravity(characteristics0) 
    min_mesh_spacing0, average_mesh_spacing0, max_mesh_spacing0 = preprocessing.compute_mesh_spacing(mesh)
    print('input mesh characteristics: {}'.format(characteristics0))
    print('input mesh COG = [xG0:{}, yG0:{}, zG0:{}]'.format(center_of_gravity0[0], center_of_gravity0[1], center_of_gravity0[2]))
    print("input mesh min mesh spacing: {:.3f} mm".format(min_mesh_spacing0))
    print("input mesh mean mesh spacing: {:.3f} mm".format(average_mesh_spacing0))
    print("input mesh max mesh spacing: {:.3f} mm".format(max_mesh_spacing0))        
        
    # Scalar Function Space
    #######################
    S = fenics.FunctionSpace(mesh, "CG", 1) 
    
    # mapping vertex --> scalar DOF
    from FEM_biomechanical_model import mappings
    vertex2dofs_S = mappings.vertex_to_dof_ScalarFunctionSpace(S)
    
    # FEM functions
    fa = fenics.Function(S, name="FA")
    normalized_fa = fenics.Function(S, name="normalizedFA")
    alphaTAN = fenics.Function(S, name="alphaTAN")
        
    # Get FA from .vtk
    dict_FA_VTK = args.fractionalanisotropy
    for i in dict_FA_VTK:
        tGW = i
        path_inputmeshloaded_with_FA_VTK = dict_FA_VTK[i]

    # Normalize FA (FA value should be between 0. and 1.)
    fa, normalized_fa = normalize_FA(fa, normalized_fa, vertex2dofs_S, path_inputmeshloaded_with_FA_VTK)

    # Compute tangential growth coefficient from FA values
    linear_coef = 5.
    alphaTAN = tangential_growth_coef_from_linear_FA(linear_coef, alphaTAN, vertex2dofs_S, normalized_fa)

    # Export .xdmf mesh loaded with normalized FA and FA-based alphaTAN
    mesh_with_MRIDataloaded_XDMF = fenics.XDMFFile(args.outputfile + "loaded_with_normalizedFA_and_alphaTAN_at{}GW.xdmf".format(tGW))
    mesh_with_MRIDataloaded_XDMF.parameters["flush_output"] = True
    mesh_with_MRIDataloaded_XDMF.parameters["functions_share_mesh"] = True
    mesh_with_MRIDataloaded_XDMF.write(normalized_fa, tGW)
    mesh_with_MRIDataloaded_XDMF.write(alphaTAN, tGW)
    
    
    