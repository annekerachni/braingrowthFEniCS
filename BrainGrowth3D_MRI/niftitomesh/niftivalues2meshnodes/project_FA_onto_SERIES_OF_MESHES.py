
import argparse
import json 
import fenics
import meshio
import numpy as np
import sys, os
import itk

sys.path.append((os.path.dirname(sys.path[0]))) # niftitomesh
# from niftivalues2meshnodes import transfer_nifti_values_to_mesh_nodes

from spatialorientationadapter_to_ras import apply_lps_ras_transformation, nifti_itk_analyzer, itk_coordinate_orientation_system_analyzer


# If input mesh with associated reference MRI nifti (brain) 
# ---------------------------------------------------------
def transfer_nifti_values_to_mesh_nodes(MRI_nifti_path, mesh_coordinates, interpolation_mode):
    """ 
    Interpolation of the input nifti values (segmentation label, diffusion (FA), etc.) over all mesh nodes. 

    Parameters:
    mesh_coordinates (numpy array: (number_nodes, 3)): Sparse 3D nodes coordinates
    reference_nii_path (string): pathway to the reference nifti image

    Returns: 
    numpy array (number of nodes): Interpolated values from nifti for each corresponding mesh node.
    """
    
    image_itk = itk.imread(MRI_nifti_path)
    xyz = mesh_coordinates

    # Reorient the MRI reference nifti from LPS+ coordinate system (itk convention) to RAS+ if coordinates are in RAS+ 
    image_itk_RAS = apply_lps_ras_transformation(image_itk)

    if interpolation_mode == 'linear':
        lin_interpolator = itk.LinearInterpolateImageFunction.New(image_itk_RAS) 

        # Interpolate the initial mri values to the coordinates (ijk space)
        values_loaded_to_mesh = np.zeros(len(xyz)) # MRI intensity values
        for i in range(len(mesh_coordinates)):
            ijk = image_itk_RAS.TransformPhysicalPointToContinuousIndex((float(xyz[i][0]), float(xyz[i][1]), float(xyz[i][2]))) #find the closest pixel to the vertex[i] (continuous index)
            values_loaded_to_mesh[i] = lin_interpolator.EvaluateAtContinuousIndex(ijk) #interpolates value around the index et attribute the interpolation value to the associated mesh node
    
        # Interpolate the initial mri values to the coordinates (ijk space)
    elif interpolation_mode == 'BSpline':
        BSpline_interpolator = itk.BSplineInterpolateImageFunction.New(image_itk_RAS) 

        # Interpolate the initial mri values to the coordinates (ijk space)
        values_loaded_to_mesh = np.zeros(len(xyz)) # MRI intensity values
        for i in range(len(xyz)):
            ijk = image_itk_RAS.TransformPhysicalPointToContinuousIndex((float(xyz[i][0]), float(xyz[i][1]), float(xyz[i][2]))) #find the closest pixel to the vertex[i] (continuous index)
            values_loaded_to_mesh[i] = BSpline_interpolator.EvaluateAtContinuousIndex(ijk) #interpolates value around the index et attribute the interpolation value to the associated mesh node


    elif interpolation_mode == 'nearest_neighbor':
        nearest_neighbor_interpolator = itk.NearestNeighborInterpolateImageFunction.New(image_itk_RAS) 

        # Interpolate the initial mri values to the coordinates (ijk space)
        values_loaded_to_mesh = np.zeros(len(xyz)) # MRI intensity values
        for i in range(len(mesh_coordinates)):
            ijk = image_itk_RAS.TransformPhysicalPointToContinuousIndex((xyz[i][0], xyz[i][1], xyz[i][2])) #find the closest pixel to the vertex[i] (continuous index)
            values_loaded_to_mesh[i] = nearest_neighbor_interpolator.EvaluateAtContinuousIndex(ijk) #interpolates value around the index et attribute the interpolation value to the associated mesh node            
    
    return values_loaded_to_mesh # MRI nifti values converted into a numpy array (size:mesh n_nodes)



if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description='Transfer MRI values (e.g. Segmentation, FA) from nifti to mesh (using itk and  .stl, .vtk or .xml mesh)')
    
    parser.add_argument('-m', '--inputmeshfolder', help='Path to input FEniCS readable formal .xml mesh)', type=str, required=False, 
                        default='/home/anne/Desktop/FA/dHCP_volume/STL_raw/todo/')#  /home/anne/Desktop/FA/dHCP_volume/meshes/         

    parser.add_argument('-niis', '--seriesofniftis', help='Path to the orginal nifti file (.nii) + Path to the associated segmentation file (.nii)', type=json.loads, required=False, 
                        default={ 
                                 
                                 "T2_folder":'/home/anne/Desktop/FA/dHCP_volume/structural/',
                                 
                                 "segmentation_folder":'/home/anne/Desktop/FA/dHCP_volume/parcellations/',
                                 
                                 "FA_folder": '/home/anne/Desktop/FA/dHCP_volume/fractional_anisotropy/'
                                 
                                 } )
    
    parser.add_argument('-of', '--outputfolder', help='Path to output folder where to write mesh + loaded nifti values)', type=str, required=False, 
                        default='/home/anne/Desktop/FA/dHCP_volume/meshes_rawSTL_loaded/')
    
    args = parser.parse_args()

    # Interpolation modes 
    #####################
    interpolation_mode_T2 = 'linear'
    interpolation_mode_segmentation = 'nearest_neighbor' 
    interpolation_mode_FA = 'linear'

    # Input mesh format
    ###################
    input_mesh_folder = args.inputmeshfolder
        
    for mesh_filename in os.listdir(input_mesh_folder):
        inputmesh_format = mesh_filename.split('.')[-1]
        
        mesh_path = input_mesh_folder + mesh_filename
    
    # .xml / .xdmf
    ##############
        if inputmesh_format == 'xml' or inputmesh_format == 'xdmf':
            
            tGW = mesh_path.split('.')[0].split('GW')[0].split('dhcp')[1]
            
            # Input mesh
            ############
            if inputmesh_format == 'xml':
                meshFEniCS = fenics.Mesh(mesh_path)
                
            elif inputmesh_format == 'xdmf':
                meshFEniCS = fenics.Mesh()
                with fenics.XDMFFile(mesh_path) as infile:
                    infile.read(meshFEniCS)
                    
            """
            vedo.dolfin.plot(mesh, 
                            mode='mesh', 
                            wireframe=True, 
                            style='meshlab').clear() 
            """
            
            mesh_coordinates = meshFEniCS.coordinates() # list of nodes coordinates
            
            # Revert X<>Y coordinates inversion performed by Netgen
            X = mesh_coordinates[:,0].copy()
            Y = mesh_coordinates[:,1].copy()
            mesh_coordinates[:,0] = Y
            mesh_coordinates[:,1] = X 
            
            # Convert mesh into vtk mesh
            ############################
            tets = meshFEniCS.cells()    
            mesh = meshio.Mesh(points=mesh_coordinates, cells={'tetra': tets})
                
            # Interpolate MRI niftis onto mesh nodes and get nodal arrays
            #############################################################
            # Read MRI nifti values onto mesh nodes (get n_nodes numpy array)
            t2_MRI_folder_path = args.seriesofniftis["T2_folder"]
            segmentation_folder_path = args.seriesofniftis["segmentation_folder"]
            diffusion_FA_folder_path = args.seriesofniftis["FA_folder"]
            
            t2_MRI_path = t2_MRI_folder_path + 't2-t{}.00.nii.gz'.format(tGW)
            segmentation_path = segmentation_folder_path + 'tissue-t{}.00_dhcp-19.nii.gz'.format(tGW)
            diffusion_FA_path = diffusion_FA_folder_path + 'fa-t{}.00.nii.gz'.format(tGW)
            
            # check theoritical values of labels in the input segmentation (to get back on the mesh) 
            import nibabel
            segmentation_image = nibabel.load(segmentation_path)
            data = segmentation_image.get_fdata()
            input_labels = np.unique(data).astype(int)
            print('input labels from the segmentation nifti: {}'.format(input_labels))
            print('labels used to mask the nifti and generate the mesh: [4 6 8 9 15 17 18]')
            
            # Load MRI nifti values onto mesh nodes (get n_nodes numpy array)
            T2intensity_array = transfer_nifti_values_to_mesh_nodes(t2_MRI_path, mesh_coordinates, interpolation_mode_T2)
            segmentation_labels_array = transfer_nifti_values_to_mesh_nodes(segmentation_path, mesh_coordinates, interpolation_mode_segmentation)
            FA_values_array = transfer_nifti_values_to_mesh_nodes(diffusion_FA_path, mesh_coordinates, interpolation_mode_FA)
            
            # Build dictionnary from nodal arrays and load mesh
            ###################################################
            node_textures = {} 
            node_textures['T2'] = T2intensity_array
            node_textures['Segmentation'] = segmentation_labels_array
            node_textures['FA'] = FA_values_array 
            
            for key in node_textures:
                mesh.point_data[key] = node_textures[key]
            
            # Change labels to get correct label transfer to mesh
            #####################################################
            ### sum_nodes should be equal to mesh.num_vertices() to be sure all nodes have been labeled.  
            ### labels 0, 1, 2, 9, 10, 18 should not be labeled (since where not used to pre-mask the original nifti image used to generate mesh...). They are present since mesh smoothing + interpolation  
            
            # labels O (background on nifti) should be either label 3, either label 4: use the label of closest nodes 
            # labels 10 (cerebral trunc on nifti) should be either label 3, either label 4: use the label of closest nodes 
            # label 1 --> label 3 (Cortex)
            # label 2 --> label 4 (Cortex)
            
            # label 9 & 18 --> OK. not to remove since within the masked mesh
            
            """ labels_to_change = [0., 10., 1., 2.] """
            
            ###
            from scipy.spatial import cKDTree
        
            tree = cKDTree(mesh_coordinates) # the whole mesh

            # Label 0. to change
            # ------------------
            list_of_index_label_0 = np.where(mesh.point_data['Segmentation'][:] == 0.) # number of nodes for which to change label: len(list_of_scalar_dofs_label_0[0])
            for node_IDX in list(list_of_index_label_0[0]):
                #Find closest node that has non-0 label 
                _, nodeIDX_of_closest_nodes = tree.query(mesh_coordinates[node_IDX], k=10) # returns [distance to closest node, closest node scalar DOF]
                for nodeIDX_closest_node in list(nodeIDX_of_closest_nodes):
                    if mesh.point_data['Segmentation'][nodeIDX_closest_node] != 0.:
                        break # stops at first closest node that have label != 0.
                
                # Replace label 0 by the label of closest node which has non-0 label
                mesh.point_data['Segmentation'][node_IDX] = mesh.point_data['Segmentation'][nodeIDX_closest_node] 
            
            # Label 10. to change
            # -------------------
            list_of_index_label_10 = np.where(mesh.point_data['Segmentation'][:] == 10.) # number of nodes for which to change label: len(list_of_scalar_dofs_label_0[0])
            for node_IDX in list(list_of_index_label_10[0]):
                #Find closest node that has non-10 label 
                _, nodeIDX_of_closest_nodes = tree.query(mesh_coordinates[node_IDX], k=10) # returns [distance to closest node, closest node scalar DOF]
                for nodeIDX_closest_node in list(nodeIDX_of_closest_nodes):
                    if mesh.point_data['Segmentation'][nodeIDX_closest_node] != 10.:
                        break # stops at first closest node that have label != 10.
                
                # Replace label 10 by the label of closest node which has non-10 label
                mesh.point_data['Segmentation'][node_IDX] = mesh.point_data['Segmentation'][nodeIDX_closest_node] 
                
            ###
            # Label 1. to change
            # ------------------        
            list_of_scalar_dofs_label_1 = np.where(mesh.point_data['Segmentation'][:] == 1.)
            for scalar_dof in list_of_scalar_dofs_label_1:
                mesh.point_data['Segmentation'][scalar_dof] = 3.
            
            # Label 2. to change
            # ------------------
            list_of_scalar_dofs_label_2 = np.where(mesh.point_data['Segmentation'][:] == 2.)
            for scalar_dof in list_of_scalar_dofs_label_2:
                mesh.point_data['Segmentation'][scalar_dof] = 4.
            
            # Check if each node has a label
            ################################
            list_of_labels = np.unique( mesh.point_data['Segmentation']) 
            
            values_of_labels_transfered_onto_mesh = []
            number_of_node_per_label = []
            for i, label in enumerate(list_of_labels):
                values_of_labels_transfered_onto_mesh.append(label)
                number_of_node_per_label.append( [label, len(np.where(segmentation_labels_array == label)[0]) ] ) # e.g. [label 1., 4276 nodes]
            
            sum_nodes = 0
            for label, number_of_nodes_with_this_label in number_of_node_per_label:
                sum_nodes += number_of_nodes_with_this_label # sum_nodes should be equal to mesh.num_vertices() to be sure all nodes have been labeled.

            # Export .vtk
            #############
            output_mesh_path = args.outputfolder + 'volume/dhcp{}_volume_MRIloaded.vtk'.format(tGW)
            mesh.write(output_mesh_path)   
            print("dhcp{}_volume_MRIloaded.vtk written down".format(tGW))
    
        # .vtk
        ######
        elif inputmesh_format == 'vtk':
            
            tGW = mesh_path.split('.')[0].split('GW')[0].split('dhcp')[1]
            
            # Input mesh
            ############
            mesh = meshio.read(mesh_path)
            mesh_coordinates = mesh.points
            
            # Revert X<>Y coordinates inversion performed by Netgen
            X = mesh_coordinates[:,0].copy()
            Y = mesh_coordinates[:,1].copy()
            mesh_coordinates[:,0] = Y
            mesh_coordinates[:,1] = X

            # Interpolate MRI niftis onto mesh nodes and get nodal arrays
            #############################################################
            # Read MRI nifti values onto mesh nodes (get n_nodes numpy array)
            t2_MRI_folder_path = args.seriesofniftis["T2_folder"]
            segmentation_folder_path = args.seriesofniftis["segmentation_folder"]
            diffusion_FA_folder_path = args.seriesofniftis["FA_folder"]
            
            t2_MRI_path = t2_MRI_folder_path + 't2-t{}.00.nii.gz'.format(tGW)
            segmentation_path = segmentation_folder_path + 'tissue-t{}.00_dhcp-19.nii.gz'.format(tGW)
            diffusion_FA_path = diffusion_FA_folder_path + 'fa-t{}.00.nii.gz'.format(tGW)
            
            # check theoritical values of labels in the input segmentation (to get back on the mesh) 
            import nibabel
            segmentation_image = nibabel.load(segmentation_path)
            data = segmentation_image.get_fdata()
            input_labels = np.unique(data).astype(int)
            print('input labels from the segmentation nifti: {}'.format(input_labels))
            print('labels used to mask the nifti and generate the mesh: [4 6 8 9 15 17 18]')
            
            # Load MRI nifti values onto mesh nodes (get n_nodes numpy array)
            T2intensity_array = transfer_nifti_values_to_mesh_nodes(t2_MRI_path, mesh_coordinates, interpolation_mode_T2)
            segmentation_labels_array = transfer_nifti_values_to_mesh_nodes(segmentation_path, mesh_coordinates, interpolation_mode_segmentation)
            FA_values_array = transfer_nifti_values_to_mesh_nodes(diffusion_FA_path, mesh_coordinates, interpolation_mode_FA)
            
            # Build dictionnary from nodal arrays and load mesh
            ###################################################
            node_textures = {} 
            node_textures['T2'] = T2intensity_array
            node_textures['Segmentation'] = segmentation_labels_array
            node_textures['FA'] = FA_values_array 

            for key in node_textures:
                mesh.point_data[key] = node_textures[key]
                
            # Export .vtk
            #############
            output_mesh_path = args.outputfolder + 'volume/dhcp{}_volume_MRIloaded.vtk'.format(tGW)
            mesh.write(output_mesh_path) 
            print("dhcp{}_volume_MRIloaded.vtk written down".format(tGW))
    
    
        # . stl
        #######
        elif inputmesh_format == 'stl':
            
            tGW = mesh_path.split('.')[0].split('GW')[0].split('dhcp')[1]
            
            # Input mesh
            ############
            mesh = meshio.read(mesh_path)
            mesh_coordinates = mesh.points

            # Interpolate MRI niftis onto mesh nodes and get nodal arrays
            #############################################################
            # Read MRI nifti values onto mesh nodes (get n_nodes numpy array)
            t2_MRI_folder_path = args.seriesofniftis["T2_folder"]
            segmentation_folder_path = args.seriesofniftis["segmentation_folder"]
            diffusion_FA_folder_path = args.seriesofniftis["FA_folder"]
            
            t2_MRI_path = t2_MRI_folder_path + 't2-t{}.00.nii.gz'.format(tGW)
            segmentation_path = segmentation_folder_path + 'tissue-t{}.00_dhcp-19.nii.gz'.format(tGW)
            diffusion_FA_path = diffusion_FA_folder_path + 'fa-t{}.00.nii.gz'.format(tGW)
            
            # Load MRI nifti values onto mesh nodes (get n_nodes numpy array)            
            T2intensity_array = transfer_nifti_values_to_mesh_nodes(t2_MRI_path, mesh_coordinates, interpolation_mode_T2)
            segmentation_labels_array = transfer_nifti_values_to_mesh_nodes(segmentation_path, mesh_coordinates, interpolation_mode_segmentation)
            FA_values_array = transfer_nifti_values_to_mesh_nodes(diffusion_FA_path, mesh_coordinates, interpolation_mode_FA)
            
            # Build dictionnary from nodal arrays and load mesh
            ###################################################
            node_textures = {} 
            node_textures['T2'] = T2intensity_array
            node_textures['Segmentation'] = segmentation_labels_array
            node_textures['FA'] = FA_values_array 
            
            for key in node_textures:
                mesh.point_data[key] = node_textures[key]
            
            # Change labels to get correct label transfer to mesh
            #####################################################
            ### sum_nodes should be equal to mesh.num_vertices() to be sure all nodes have been labeled.  
            ### labels 0, 1, 2, 9, 10, 18 should not be labeled (since where not used to pre-mask the original nifti image used to generate mesh...). They are present since mesh smoothing + interpolation  
            
            # labels O (background on nifti) should be either label 3, either label 4: use the label of closest nodes 
            # labels 10 (cerebral trunc on nifti) should be either label 3, either label 4: use the label of closest nodes 
            # label 1 --> label 3 (Cortex)
            # label 2 --> label 4 (Cortex)
            
            # label 9 & 18 --> OK. not to remove since within the masked mesh
            
            """ labels_to_change = [0., 10., 1., 2.] """
            
            ###
            from scipy.spatial import cKDTree
        
            tree = cKDTree(mesh_coordinates) # the whole mesh

            # Label 0. to change
            # ------------------
            list_of_index_label_0 = np.where(mesh.point_data['Segmentation'][:] == 0.) # number of nodes for which to change label: len(list_of_scalar_dofs_label_0[0])
            for node_IDX in list(list_of_index_label_0[0]):
                #Find closest node that has non-0 label 
                _, nodeIDX_of_closest_nodes = tree.query(mesh_coordinates[node_IDX], k=10) # returns [distance to closest node, closest node scalar DOF]
                for nodeIDX_closest_node in list(nodeIDX_of_closest_nodes):
                    if mesh.point_data['Segmentation'][nodeIDX_closest_node] != 0.:
                        break # stops at first closest node that have label != 0.
                
                # Replace label 0 by the label of closest node which has non-0 label
                mesh.point_data['Segmentation'][node_IDX] = mesh.point_data['Segmentation'][nodeIDX_closest_node] 
            
            # Label 10. to change
            # -------------------
            list_of_index_label_10 = np.where(mesh.point_data['Segmentation'][:] == 10.) # number of nodes for which to change label: len(list_of_scalar_dofs_label_0[0])
            for node_IDX in list(list_of_index_label_10[0]):
                #Find closest node that has non-10 label 
                _, nodeIDX_of_closest_nodes = tree.query(mesh_coordinates[node_IDX], k=10) # returns [distance to closest node, closest node scalar DOF]
                for nodeIDX_closest_node in list(nodeIDX_of_closest_nodes):
                    if mesh.point_data['Segmentation'][nodeIDX_closest_node] != 10.:
                        break # stops at first closest node that have label != 10.
                
                # Replace label 10 by the label of closest node which has non-10 label
                mesh.point_data['Segmentation'][node_IDX] = mesh.point_data['Segmentation'][nodeIDX_closest_node] 
                
            ###
            # Label 1. to change
            # ------------------        
            list_of_scalar_dofs_label_1 = np.where(mesh.point_data['Segmentation'][:] == 1.)
            for scalar_dof in list_of_scalar_dofs_label_1:
                mesh.point_data['Segmentation'][scalar_dof] = 3.
            
            # Label 2. to change
            # ------------------
            list_of_scalar_dofs_label_2 = np.where(mesh.point_data['Segmentation'][:] == 2.)
            for scalar_dof in list_of_scalar_dofs_label_2:
                mesh.point_data['Segmentation'][scalar_dof] = 4.
            
            # Check if each node has a label
            ################################
            list_of_labels = np.unique( mesh.point_data['Segmentation']) 
            
            values_of_labels_transfered_onto_mesh = []
            number_of_node_per_label = []
            for i, label in enumerate(list_of_labels):
                values_of_labels_transfered_onto_mesh.append(label)
                number_of_node_per_label.append( [label, len(np.where(segmentation_labels_array == label)[0]) ] ) # e.g. [label 1., 4276 nodes]
            
            sum_nodes = 0
            for label, number_of_nodes_with_this_label in number_of_node_per_label:
                sum_nodes += number_of_nodes_with_this_label # sum_nodes should be equal to mesh.num_vertices() to be sure all nodes have been labeled.

            # Export .vtk
            #############
            output_mesh_path = args.outputfolder + 'surface/dhcp{}_surface_MRIloaded.vtk'.format(tGW)
            mesh.write(output_mesh_path)  
            
            print("dhcp{}_surface_MRIloaded.vtk written down".format(tGW))