# source code: https://github.com/rousseau/BrainGrowth/

import numpy as np

def det_dim_3(a):
        size = a.shape[0]
        b = np.zeros(size, dtype=np.float64)
        for i in range(size):
            b[i] = (
            a[i, 0, 0] * a[i, 1, 1] * a[i, 2, 2]
            - a[i, 0, 0] * a[i, 1, 2] * a[i, 2, 1]
            - a[i, 0, 1] * a[i, 1, 0] * a[i, 2, 2]
            + a[i, 0, 1] * a[i, 1, 2] * a[i, 2, 0]
            + a[i, 0, 2] * a[i, 1, 0] * a[i, 2, 1]
            - a[i, 0, 2] * a[i, 1, 1] * a[i, 2, 0]
        )
        return b

def transpose_dim_3(a):
    # Purely equal to np.transpose (a, (0, 2, 1))
    size = a.shape[0]
    b = np.zeros((size, 3, 3), dtype=np.float64)
    for i in range(size):
        b[i, 0, 0] = a[i, 0, 0]
        b[i, 1, 0] = a[i, 0, 1]
        b[i, 2, 0] = a[i, 0, 2]
        b[i, 0, 1] = a[i, 1, 0]
        b[i, 1, 1] = a[i, 1, 1]
        b[i, 2, 1] = a[i, 1, 2]
        b[i, 0, 2] = a[i, 2, 0]
        b[i, 1, 2] = a[i, 2, 1]
        b[i, 2, 2] = a[i, 2, 2]

    return b

def compute_mesh_volume(mesh):
    # mesh volume
    # Code source: https://github.com/rousseau/BrainGrowth/blob/master/geometry.py#L165
    print("Computing the folded mesh volume...")

    tets = mesh.cells()
    coordinates = mesh.coordinates()

    mesh_volume = 0.0
    for tet in tets:

        tmp0 = coordinates[tet[1]] - coordinates[tet[0]]
        tmp1 = coordinates[tet[2]] - coordinates[tet[0]]
        tmp2 = coordinates[tet[3]] - coordinates[tet[0]]
        
        tetrahedron_matrix = np.array([ tmp0, tmp1, tmp2 ])
        #tetrahedron_matrix.shape
        det = np.linalg.det( np.transpose(tetrahedron_matrix) )

        mesh_volume += 1/6 * abs(det)

    return mesh_volume


if __name__ == '__main__':
    
    import sys, os
    sys.path.append(sys.path[0]) 
    sys.path.append(os.path.dirname(os.path.dirname(sys.path[0]))) # braingrowthFEniCS

    import fenics
    from FEM_biomechanical_model import preprocessing
    import json
    
    # OUTPUT FOLDER  
    ###############
    output_folder_path = './metrics/volumes_21_28_36GW/'
    
    # OPTION
    ########
    which_volume = "wholebrain_and_cortex" # "wholebrain"; "wholebrain_and_cortex"
    
    # WHOLE BRAIN MESH VOLUME
    #########################
    if which_volume == "wholebrain":
        
        # brain meshes
        inputmeshes_dict = {21: "./data/21_28_36GW/transformed_niftis_meshes/transformed_dhcp21GW_isotropic_smoothed.xdmf",
                            28: "./data/21_28_36GW/transformed_niftis_meshes/transformed_dhcp28GW_isotropic_smoothed.xdmf",
                            36: "./data/21_28_36GW/transformed_niftis_meshes/transformed_dhcp36GW_isotropic_sigma_0_3.xdmf"
                            }
        
        # output
        volumes = {}
        
        # compute cortical area for each tGW
        for tGW, inputmesh_path in inputmeshes_dict.items():
            inputmesh_format = inputmesh_path.split('.')[-1]

            if inputmesh_format == "xml":
                """mesh0 = fenics.Mesh(inputmesh_path)"""
                mesh = fenics.Mesh(inputmesh_path)

            elif inputmesh_format == "xdmf":
                """
                mesh0 = fenics.Mesh()
                with fenics.XDMFFile(inputmesh_path) as infile:
                    infile.read(mesh0)
                """
                
                mesh = fenics.Mesh()
                with fenics.XDMFFile(inputmesh_path) as infile:
                    infile.read(mesh)
                    
            # convert mesh from mm into m (SI)
            mesh = preprocessing.converting_mesh_from_millimeters_into_meters(mesh)
            
            # compute volume
            volume = compute_mesh_volume(mesh)
            
            #volumes[tGW] = volume
            volumes["whole brain mesh volume at {}GW (m³)".format(tGW)] = volume
            #print('mesh volume at {}GW = {} m'.format(tGW, volume))  
        
        # export json file 
        output_path = os.path.join(output_folder_path, 'volume_21_28_36GW.json')
        
        with open(output_path, 'w') as volumes_json_file:  
            json.dump(volumes, volumes_json_file)

        """
        with open(output_path, 'r') as volumes_json_file:  
            volumes_json_str = volumes_json_file.read()
            volumes_values = json.loads(volumes_json_str)
        """
    
    ###
    
    # CORTEX MESH VOLUME
    ####################
    elif which_volume == "wholebrain_and_cortex":
        
        #import braingrowth_3D.MRI_based_biophysical_quasistatic_3D.niftitomesh
        
        from BrainGrowth3D_MRI.niftitomesh.niftivalues2meshnodes import transfer_niftivalues_to_meshnodes_withITK # to load MRI data onto mesh nodes
        from BrainGrowth3D_MRI.MRI_driven_parameters.FA_2_growthcoef import load_mesh_with_growthcoef_ECCOMAS_XDMF # to compute alphaTAN from FA nodal values
        from BrainGrowth3D_MRI.MRI_driven_parameters.Segmentation_to_grTAN_for_cortical_thickness import load_mesh_with_grTAN # to get Cortical delineation from Segmentation label values

        from FEM_biomechanical_model import mappings
        
        # brain meshes
        inputmeshes_dict = {21: "./data/21_28_36GW/transformed_niftis_meshes/transformed_dhcp21GW_isotropic_smoothed.xdmf",
                            28: "./data/21_28_36GW/transformed_niftis_meshes/transformed_dhcp28GW_isotropic_smoothed.xdmf",
                            36: "./data/21_28_36GW/transformed_niftis_meshes/transformed_dhcp36GW_isotropic_sigma_0_3.xdmf"
                            }
        
        # Load segmentation from MRI nifti at T0 (e.g. 21GW) to mesh nodes
        input_parcellations_dict = {21: "./data/21_28_36GW/transformed_niftis_meshes/transformed-tissue-t21.00_dhcp-19.nii.gz",
                                    28: "./data/21_28_36GW/transformed_niftis_meshes/transformed-tissue-t28.00_dhcp-19.nii.gz",
                                    36: "./data/21_28_36GW/transformed_niftis_meshes/transformed-tissue-t36.00_dhcp-19.nii.gz"
                                    }
        
        # output  
        volumes = {}
        
        for tGW, inputmesh_path in inputmeshes_dict.items():
            
            # load mesh
            inputmesh_format = inputmesh_path.split('.')[-1]
            
            if inputmesh_format == "xml":
                mesh0 = fenics.Mesh(inputmesh_path) # mesh0 (for projection of nifti values) 
                mesh = fenics.Mesh(inputmesh_path)

            elif inputmesh_format == "xdmf":
                
                mesh0 = fenics.Mesh()
                with fenics.XDMFFile(inputmesh_path) as infile:
                    infile.read(mesh0) # mesh0 (for projection of nifti values) 
                
                
                mesh = fenics.Mesh()
                with fenics.XDMFFile(inputmesh_path) as infile:
                    infile.read(mesh)
                    
            # convert mesh from mm into m (SI)
            mesh = preprocessing.converting_mesh_from_millimeters_into_meters(mesh)
            
            # subdomains
            subdomains = fenics.MeshFunction("size_t", mesh, mesh.topology().dim()) 
            subdomains.set_all(0)
        
            # load segmentation
            mesh_meshio_Labels = transfer_niftivalues_to_meshnodes_withITK.load_Segmentation_onto_mesh_nodes(mesh0, 
                                                                                                                     input_parcellations_dict[tGW], 
                                                                                                                     interpolation_mode='nearest_neighbor') # interpolation_mode: 'nearest_neighbor'; 'linear'
            
            """mesh_meshio_Labels.write( os.path.join(output_folder_path, "mesh_Segmentation_values_{}GW".format(int(tGW)) + '.vtk'))""" 
            
            # get nodal label of Cortex or Core belonging from dHCP (whole brain) Segmentation nifti 
            S = fenics.FunctionSpace(mesh, "CG", 1) 
            vertex2dofs_S = mappings.vertex_to_dof_ScalarFunctionSpace(S)
            
            grTAN = fenics.Function(S, name="grTANCortexDelineation")
            
            grTAN.assign( load_mesh_with_grTAN.bilayer_tangential_growth_ponderation_from_SegmentationLabels_WHOLEMESH(grTAN, 
                                                                                                                               vertex2dofs_S, 
                                                                                                                               mesh_meshio_Labels) )
            """
            outputpath = os.path.join(output_folder_path, "cortex_label.xdmf")
            FEniCS_FEM_Functions_file = fenics.XDMFFile(outputpath)
            FEniCS_FEM_Functions_file.parameters["flush_output"] = True
            FEniCS_FEM_Functions_file.parameters["functions_share_mesh"] = True
            FEniCS_FEM_Functions_file.parameters["rewrite_function_mesh"] = True
            
            FEniCS_FEM_Functions_file.write(grTAN, tGW)
            """

            # Get cortical submesh      
            class MyDict(dict): # https://fenicsproject.org/qa/5268/is-that-possible-to-identify-a-facet-by-its-vertices/
                def get(self, key):
                    return dict.get(self, sorted(key))

            tet_2_v = MyDict((tet.index(), tuple(tet.entities(0))) for tet in fenics.cells(mesh))
            
            for tet in fenics.cells(mesh):
                vertex1, vertex2, vertex3, vertex4 = tet_2_v[tet.index()]
                if grTAN.vector()[vertex2dofs_S[vertex1]] == 1. or \
                grTAN.vector()[vertex2dofs_S[vertex2]] == 1. or \
                grTAN.vector()[vertex2dofs_S[vertex3]] == 1. or \
                grTAN.vector()[vertex2dofs_S[vertex4]] == 1. :
                    subdomains.array()[tet.index()] = 1
                
            cortex_submesh = fenics.SubMesh(mesh, subdomains, 1)
            
            """
            from mpi4py import MPI
            with fenics.XDMFFile(MPI.COMM_WORLD, os.path.join(output_folder_path, "cortex_submesh.xdmf")) as xdmf:
                xdmf.write(cortex_submesh)"""
            
            # FEniCS method to compute volumes
            dx = fenics.Measure("dx", domain=mesh, subdomain_data=subdomains)
            volume_Cortex = fenics.assemble(fenics.Constant(1) * dx(1)) # in m³
            print("volume Cortex with FEniCS method : {} m³".format(volume_Cortex))
            
            # Compute volumes : whole brain and cortex meshes
            whole_brain_volume = compute_mesh_volume(mesh) # in m³
            cortex_volume = compute_mesh_volume(cortex_submesh) # in m³

            whole_brain_key = "whole brain mesh volume at {}GW (m3)".format(tGW)
            cortex_key = "cortex mesh volume at{}GW (m3)".format(tGW)
            volumes[whole_brain_key] = whole_brain_volume
            volumes[cortex_key] = cortex_volume

        # export json file 
        output_path = os.path.join(output_folder_path, 'volume_wholebrain_cortex_21_28_36GW.json')
        
        with open(output_path, 'w') as volumes_json_file:  
            json.dump(volumes, volumes_json_file)