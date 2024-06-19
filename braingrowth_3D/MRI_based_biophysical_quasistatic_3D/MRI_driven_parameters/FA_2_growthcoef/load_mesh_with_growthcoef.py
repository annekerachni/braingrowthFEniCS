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

def normalize_FA(fa, normalized_fa, vertex2dofs_S, path_inputmeshloaded_with_FA_VTK):
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
    
    for vertex, scalarDOF in enumerate(vertex2dofs_S):
        fa.vector()[scalarDOF] = FA[vertex]
        normalized_fa.vector()[scalarDOF] = FA[vertex]/max_FA
        
    return fa, normalized_fa # FEniCS scalar functions


def tangential_growth_coef_from_FA(linear_coef, alphaTAN, vertex2dofs_S, normalized_fa):    
    
    """
    Args: 
    - vertex2dofs_S: vertex --> scalar FEniCS DOF
    - normalized_fa : fenics.Function(S) 
    """  
    
    # Defining tangential growth coefficient 'alphaTAN' in correlation to FA nodal value. (FA ++ => neuronal maturation -- => neuron matures => tangential growth ++)
    for vertex, scalarDOF in enumerate(vertex2dofs_S):
        alphaTAN.vector()[scalarDOF] = FA_to_growth_law.FA_to_tangential_growth_law_linearrelationship(linear_coef, normalized_fa.vector()[scalarDOF] ) # FA should be a float between 0. and 1. => normalized_fa = fa/np.max(fa)  + growth is a linear function of FA. (here we test alphaTAN = 5*FA)
        #alphaTAN.vector()[scalarDOF] = FA_to_tangential_growth_law_gompertz(linear_coef, normalized_fa.vector()[scalarDOF])
        #alphaTAN.vector()[scalarDOF] = FA_to_tangential_growth_law_exponential(linear_coef, normalized_fa.vector()[scalarDOF])
        
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
    
    parser.add_argument('-i', '--input', help='Input mesh path (.xml, .xdmf)', type=str, required=True, 
                        default='./data/brainmesh.xdmf') 
                        
    parser.add_argument('-n', '--normalization', help='Is normalization of the input mesh required? (required by braingrowthFEniCS)', type=bool, required=False, default=False)

    parser.add_argument('-fa', '--fractionalanisotropy', help='Path to the mesh loaded with FA values onto nodes (.vtk)', type=json.loads, required=True, 
    default={"21GW": './MRI_driven_parameters/meshes_with_nodal_values/brainmesh_loaded_with_21GWMRIvalues.vtk'})
    
    parser.add_argument('-o', '--output', help='Path to output folder', type=str, required=True, 
                        default='./MRI_driven_parameters/meshes_with_nodal_values') 
    
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

    if args.visualization == True:
        fenics.plot(mesh) 
        plt.title("input mesh")
        plt.show() 

    # mesh characteristics
    characteristics0 = preprocessing.compute_geometrical_characteristics(mesh, bmesh)
    center_of_gravity0 = preprocessing.compute_center_of_gravity(characteristics0) 
    min_mesh_spacing0, average_mesh_spacing0, max_mesh_spacing0 = preprocessing.compute_mesh_spacing(mesh)
    print('input mesh characteristics: {}'.format(characteristics0))
    print('input mesh COG = [xG0:{}, yG0:{}, zG0:{}]'.format(center_of_gravity0[0], center_of_gravity0[1], center_of_gravity0[2]))
    print("input mesh min mesh spacing: {:.3f} mm".format(min_mesh_spacing0))
    print("input mesh mean mesh spacing: {:.3f} mm".format(average_mesh_spacing0))
    print("input mesh max mesh spacing: {:.3f} mm".format(max_mesh_spacing0))

    # normalization and boundary mesh update, normalized mesh characteristics
    if args.normalization == True:
        print("\nnormalizing mesh...")
        mesh = preprocessing.normalize_mesh(mesh, characteristics0, center_of_gravity0)
        bmesh = fenics.BoundaryMesh(mesh, "exterior") # update bmesh

        if args.visualization == True:
            fenics.plot(mesh) 
            plt.title("normalized mesh")
            plt.show()  
            #vedo.dolfin.plot(mesh, wireframe=False, text='normalized mesh', style='paraview', axes=4).close()
            
        characteristics = preprocessing.compute_geometrical_characteristics(mesh, bmesh) # n_nodes, coordinates, n_tets, n_faces_Surface, n_faces_Volume, minx, maxx, miny, maxy, minz, maxz 
        center_of_gravity = preprocessing.compute_center_of_gravity(characteristics) 
        min_mesh_spacing, average_mesh_spacing, max_mesh_spacing = preprocessing.compute_mesh_spacing(mesh)
        print('normalized mesh characteristics: {}'.format(characteristics))
        print('normalized mesh COG = [xG:{}, yG:{}, zG:{}]'.format(center_of_gravity[0], center_of_gravity[1], center_of_gravity[2]))
        print("normalized min_mesh_spacing: {:.3f} mm".format(min_mesh_spacing))
        print("normalized mesh mean mesh spacing: {:.3f} mm".format(average_mesh_spacing))
        print("normalized mesh max mesh spacing: {:.3f} mm".format(max_mesh_spacing))
        
        
    # Scalar Function Space
    #######################
    S = fenics.FunctionSpace(mesh, "CG", 1) 
    
    # mapping vertex --> scalar DOF
    from FEM_biomechanical_model_quasistatic import mappings
    vertex2dofs_S = mappings.vertex_to_dof_ScalarFunctionSpace(S)
    
    # FEM functions
    fa = fenics.Function(S, name="FA")
    normalized_fa = fenics.Function(S, name="normalizedFA")
    alphaTAN = fenics.Function(S, name="alphaTAN")
    
    t_simu, tGW = 0., 21 # simulation time and corresponding real time
    
    # Get FA from .vtk
    path_inputmeshloaded_with_FA_VTK = args.fractionalanisotropy
    
    # Normalize FA (FA value should be between 0. and 1.)
    fa, normalized_fa = normalize_FA(fa, normalized_fa, vertex2dofs_S, path_inputmeshloaded_with_FA_VTK)
    
    normalizedFA_file_XDMF = fenics.XDMFFile(os.path.join(args.output, "brainmesh_loaded_with_normalizedFA_at{}GW.xdmf").format(tGW))
    normalizedFA_file_XDMF.parameters["flush_output"] = True
    normalizedFA_file_XDMF.parameters["functions_share_mesh"] = True
    normalizedFA_file_XDMF.write(normalized_fa, t_simu)
    
    # Compute tangential growth coefficient from FA values
    linear_coef = 5.
    alphaTAN = tangential_growth_coef_from_FA(linear_coef, alphaTAN, vertex2dofs_S, normalized_fa)
    
    alphaTAN_file_XDMF = fenics.XDMFFile(os.path.join(args.output, "brainmesh_loaded_with_alphaTAN_at{}GW.xdmf").format(tGW))
    alphaTAN_file_XDMF.parameters["flush_output"] = True
    alphaTAN_file_XDMF.parameters["functions_share_mesh"] = True
    alphaTAN_file_XDMF.write(alphaTAN, t_simu)