import numpy as np
import math
import fenics
import json, csv


def FA_evolution_law_homogeneous(meshFEniCS, T0, times, type_of_law):
    """
    Decreasing exponential FA law. Homogeneous for all mesh nodes
    
    Args:
    - meshFEniCS: fenics.Mesh()
    - times: times = np.linspace(T0, Tmax, Nsteps+1)  (defined in the simulation script main.py)
    - type_of_law: 'exponential'
    """
    
    n_nodes = meshFEniCS.num_vertices()
    #FA_temporal_values = np.zeros((len(times) , n_nodes)) # t0: [FA_node0, FA_node1, ...] ; t1: [FA_node0, FA_node1, ...], ...
    FA_temporal_values = {} 
     
    if type_of_law == 'exponential':

        A = np.zeros((n_nodes)) 
        A[:] = 0.5
        TAU = 1.
        B = 0.5
        
        for i, dt in enumerate(np.diff(times)):
            #FA_temporal_values[str(i)] = np.zeros((n_nodes))
            t = times[i]
            #FA_temporal_values[i][:] = A * math.exp(- (t - T0)/TAU ) + B # size of np.array: n_nodes --> [FA_node0, FA_node1, ...]
            FA_temporal_values[t] = A * math.exp(- (t - T0)/TAU ) + B # size of dictionnary: n_nodes --> [FA_node0, FA_node1, ...]
            
    return FA_temporal_values # dictionnary with the same FA value for each time, and for all Nsteps (size: len(times) x n_nodes)


def FA_evolution_law_heterogeneous(meshFEniCS, T0, times, normalized_FA_T0, vertex2dofs_S, type_of_law):
    """Decreasing exponential FA law. Homogeneous for all mesh nodes"""
    
    #FA_temporal_values = np.zeros((len(times), n_nodes)) # t0: [FA_node0, FA_node1, ...] ; t1: [FA_node0, FA_node1, ...], ...
    FA_temporal_values = {}
    
    A = normalized_FA_T0.vector()[:] # nodal FA collected from fetal dhcp atals at 21GW. Indexation in DOF!
    TAU = 1.
    B = 0.0
     
    if type_of_law == 'exponential':
        for i, dt in enumerate(np.diff(times)):        
            #FA_temporal_values[str(i)] = np.zeros((n_nodes))
            t = times[i]
            #FA_temporal_values[i][scalarDOF] = A[scalarDOF] * math.exp(- (t - T0)/TAU ) + B # size of np.array: scalarDOF --> [FA_node0, FA_node1, ...]
            FA_temporal_values[t] = A * math.exp(- (t - T0)/TAU ) + B # size of dictionnary: scalarDOF --> [FA_node0, FA_node1, ...]

    return FA_temporal_values # dictionnary with all FA values for each time, and for all Nsteps  (size: len(times) x len(scalarDOFs))


if __name__ == '__main__':
    
    ##########
    # CASE 1 #
    ##########

    # Parameters
    ############
    meshFEniCS = fenics.Mesh('./data/brainmesh.xdmf')
    
    T0 = 0.
    Tmax = 1.
    Nsteps = 100
    times = np.linspace(T0, Tmax, Nsteps+1)     
    
    # We associate the FA evolution law to all mesh nodes
    #####################################################
    FA_temporal_values = FA_evolution_law_homogeneous(meshFEniCS, T0, times, type_of_law='exponential') # Expressed in scalarDOF or in vertex indexation (since same value for all node so indexation does not have impact)
    
    # Export dictionnary to csv
    ###########################
    outfilepath = './MRI_driven_parameters/meshes_with_nodal_values/artificial_FA/longitudinal_FA_values/brainmesh_loaded_with_artificial_FAt_homogeneousNodalValue.csv'
    
    for t, fa in FA_temporal_values.items():
        FA_temporal_values[t] = list(FA_temporal_values[t])
    
    with open(outfilepath, 'w') as FA_json_file:  
        json.dump(FA_temporal_values, FA_json_file)

    with open(outfilepath, 'r') as FA_json_file:  
        FA_json_str = FA_json_file.read()
        normalized_FA_values = json.loads(FA_json_str)
    
    
    ##########
    # CASE 2 #
    ##########

    # Parameters
    ############
    meshFEniCS = fenics.Mesh('./data/brainmesh.xdmf')
    
    T0 = 0.
    Tmax = 1.
    Nsteps = 100
    times = np.linspace(T0, Tmax, Nsteps+1)     
    
    # We associate a specific FA evolution law to each mesh node
    ############################################################
    
    # Transfer FA from MRI fetal dhcp atlas nifti to mesh at T0, to have 'realistic' heterogeneous FA
    # -----------------------------------------------------------------------------------------------
    
    # Load Segmentation labels and FA from MRI nifti at T0 (e.g. 21GW)
    T0_GW = 21
    FA_nifti_file_path_T0GW = './fetal_database/diffusion/fa-t{}.00.nii.gz'.format(T0_GW) 
    #Segmentation_nifti_file_path_T0GW = '/home/latim/BrainGrowth_database/dhcp_fetal_atlas/fetal_brain_mri_atlas/parcellations/tissue-t{}.00_dhcp-19.nii.gz'.format(T0_GW)

    # transfer Labels and FA values onto initial mesh
    import os, sys
    sys.path.append(os.path.join(sys.path[0]))
    
    from niftitomesh.niftivalues2meshnodes import transfer_niftivalues_to_meshnodes_in_simulation 
    
    mesh_meshio = transfer_niftivalues_to_meshnodes_in_simulation.load_FA_onto_mesh_nodes(meshFEniCS, 
                                                                                          FA_nifti_file_path_T0GW, 
                                                                                          interpolation_mode='nearest_neighbor') 
    
    mesh_meshio.write('./MRI_driven_parameters/meshes_with_nodal_values/artificial_FA/longitudinal_FA_values/' + 
                      "brainmesh_loaded_with_FA_from_{}GWFAatlas".format(T0) + '.vtk') 
    
    
    # Get nodal FA 
    from MRI_driven_parameters.FA import FA_to_growth_longitudinal 
    from FEM_biomechanical_model import mappings
    
    S = fenics.FunctionSpace(meshFEniCS, "CG", 1) 
    vertex2dofs_S = mappings.vertex_to_dof_ScalarFunctionSpace(S)
    
    fa_T0 = fenics.Function(S, name="FA")
    normalized_fa_T0 = fenics.Function(S, name="NormalizedFA")
    fa_T0, normalized_fa_T0 = FA_to_growth_longitudinal.normalize_FA(fa_T0, normalized_fa_T0, vertex2dofs_S, mesh_meshio) # Normalize FA (FA value should be between 0. and 1.)
    
    
    # Compute FA for all times with the selected evolution law
    # --------------------------------------------------------
    FA_temporal_values = FA_evolution_law_heterogeneous(meshFEniCS, T0, times, normalized_fa_T0, vertex2dofs_S, type_of_law='exponential') # Expressed in scalarDOF
    
    # Export dictionnary to csv
    ###########################
    outfilepath = './MRI_driven_parameters/meshes_with_nodal_values/artificial_FA/longitudinal_FA_values/brainmesh_loaded_with_artificial_FAt_heterogeneousNodalValue_from21GWFAatlas.csv'
    
    for t, fa in FA_temporal_values.items():
        FA_temporal_values[t] = list(FA_temporal_values[t])
    
    with open(outfilepath, 'w') as FA_json_file:  
        json.dump(FA_temporal_values, FA_json_file)    

    with open(outfilepath, 'r') as FA_json_file:  
        FA_json_str = FA_json_file.read()
        normalized_FA_values = json.loads(FA_json_str)


