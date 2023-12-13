import fenics
import numpy as np
import itk
import argparse
import json
import meshio
import vedo.dolfin

import sys
sys.path.append(".")
from niftitomesh.spatialorientationadapter_to_ras import apply_lps_ras_transformation, nifti_itk_analyzer, itk_coordinate_orientation_system_analyzer


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
    
    """
    python3 -i niftivalues_itk.py 
            --inputmesh './data/dhcp/dhcp_atlas/longitudinal_STL/dhcp24GW_masked_SlicerAutomaticSmoothing.stl'
            --seriesofniftis '{ "T2":'/home/latim/BrainGrowth_database/dhcp_fetal_atlas/fetal_brain_mri_atlas/structural/t2-t24.00.nii.gz',
                   "segmentation":'/home/latim/BrainGrowth_database/dhcp_fetal_atlas/fetal_brain_mri_atlas/parcellations/tissue-t24.00_dhcp-19.nii.gz',
                   "diffusion_FA": '/home/latim/BrainGrowth_database/dhcp_fetal_atlas/fetal_brain_mri_atlas/diffusion/fa-t24.00.nii.gz'}'
            --interpolationmode 'nearest_neighbor'
            --outputfolder './utils/nifti_mesh/niftitomesh/transfer_niftivalues_to_meshnodes/results/'
            --outputfilename 'meshSTL_MRIniivalues_NearestNeighbor_24GW_homogeneizeCortexLabels.vtk'
    """
    
    """ python3 -i ./utils/nifti_mesh/niftitomesh/transfer_niftivalues_to_meshnodes/niftivalues_itk.py --inputmesh './data/dhcp/dhcp_atlas/longitudinal_STL/dhcp24GW_masked_SlicerAutomaticSmoothing.stl' --seriesofniftis '{ "T2":'/home/latim/BrainGrowth_database/dhcp_fetal_atlas/fetal_brain_mri_atlas/structural/t2-t24.00.nii.gz', "segmentation":'/home/latim/BrainGrowth_database/dhcp_fetal_atlas/fetal_brain_mri_atlas/parcellations/tissue-t24.00_dhcp-19.nii.gz', "diffusion_FA": '/home/latim/BrainGrowth_database/dhcp_fetal_atlas/fetal_brain_mri_atlas/diffusion/fa-t24.00.nii.gz'}' --interpolationmode 'nearest_neighbor' --outputfolder './utils/nifti_mesh/niftitomesh/transfer_niftivalues_to_meshnodes/results/' --outputfilename '24GW/meshSTL_MRIniivalues_NearestNeighbor_24GW_homogeneizeCortexLabels.vtk' """
    
    parser = argparse.ArgumentParser(description='Transfer MRI values (e.g. Segmentation, FA) from nifti to mesh (using itk and  .stl, .vtk or .xml mesh)')
    
    parser.add_argument('-m', '--inputmesh', help='Path to input FEniCS readable formal .xml mesh)', type=str, required=False, 
                        default='./data/fetal_dhcp_atlas/21GW/dhcp21GW_728Ktets.xml')         

    parser.add_argument('-niis', '--seriesofniftis', help='Path to the orginal nifti file (.nii) + Path to the associated segmentation file (.nii)', type=json.loads, required=False, 
                        default={ 
                                 
                                 "T2":'/home/latim/BrainGrowth_database/dhcp_fetal_atlas/fetal_brain_mri_atlas/structural/t2-t21.00.nii.gz',
                                 
                                 "segmentation":'/home/latim/BrainGrowth_database/dhcp_fetal_atlas/fetal_brain_mri_atlas/parcellations/tissue-t21.00_dhcp-19.nii.gz',
                                 
                                 "diffusion_FA": '/home/latim/BrainGrowth_database/dhcp_fetal_atlas/fetal_brain_mri_atlas/diffusion/fa-t21.00.nii.gz'
                                 #"diffusion_FA": './utils/nifti_mesh/niftitomesh/transfer_niftivalues_to_meshnodes/results/21GW/cortical_fa_21_max.nii.gz'
                                 
                                 } )
    
    parser.add_argument('-im', '--interpolationmode', help='Interpolation mode (nearest_neighbor; linear)', type=str, required=False, 
                        default='nearest_neighbor')
    
    parser.add_argument('-of', '--outputfolder', help='Path to output folder where to write mesh + loaded nifti values)', type=str, required=False, 
                        default='./MRI_driven_parameters/meshes_with_nodal_values/')
    
    parser.add_argument('-ofn', '--outputfilename', help='Output file name (.xdmf; .vtk))', type=str, required=False, 
                        default='21GW/meshXML145Ktets_MRIniivalues_NearestNeighbor_21GW_homogeneizeCortexLabels.vtk') # if .xml inputmesh --> '.xdmf'; if .stl inputmesh --> '.vtk'
    
    args = parser.parse_args()


    # Input mesh format
    ###################
    inputmesh_format = args.inputmesh.split('.')[-1]
    
    if inputmesh_format == 'xml':
        
    	# Input mesh
    	############
        meshFEniCS = fenics.Mesh(args.inputmesh)
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
        #bmeshFEniCS = fenics.BoundaryMesh(meshFEniCS, "exterior")
        #faces = bmeshFEniCS.cells()
    
        #meshio.write(output_file_xml, meshio.Mesh(points=coordinates, cells={'tetra': tets})) 
        #mesh = meshio.Mesh(points=mesh_coordinates, cells={'tetra': tets, 'triangle': faces})
        mesh = meshio.Mesh(points=mesh_coordinates, cells={'tetra': tets})
            
        # Interpolate MRI niftis onto mesh nodes and get nodal arrays
        #############################################################
        # Read MRI nifti values onto mesh nodes (get n_nodes numpy array)
        t2_MRI_path = args.seriesofniftis["T2"]
        segmentation_path = args.seriesofniftis["segmentation"]
        diffusion_FA_path = args.seriesofniftis["diffusion_FA"]
        
        # check theoritical values of labels in the input segmentation (to get back on the mesh) 
        import nibabel
        segmentation_image = nibabel.load(segmentation_path)
        data = segmentation_image.get_fdata()
        input_labels = np.unique(data).astype(int)
        print('input labels from the segmentation nifti: {}'.format(input_labels))
        print('labels used to mask the nifti and generate the mesh: [3 4 5 6 7 8 14 15 16 17]')
        
        # Load MRI nifti values onto mesh nodes (get n_nodes numpy array)
        interpolation_mode = args.interpolationmode # 'nearest_neighbor'; 'linear'
        T2intensity_array = transfer_nifti_values_to_mesh_nodes(t2_MRI_path, mesh_coordinates, interpolation_mode)
        segmentation_labels_array = transfer_nifti_values_to_mesh_nodes(segmentation_path, mesh_coordinates, interpolation_mode)
        FA_values_array = transfer_nifti_values_to_mesh_nodes(diffusion_FA_path, mesh_coordinates, interpolation_mode)
        
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
        mesh.write(args.outputfolder + args.outputfilename)   
    
    
        """ 
    if inputmesh_format == 'xml':
    	# Input mesh
    	############
        mesh = fenics.Mesh(args.inputmesh)
        
        # vedo.dolfin.plot(mesh, 
        #                 mode='mesh', 
        #                 wireframe=True, 
        #                 style='meshlab').clear() 
        
        
        mesh_coordinates = mesh.coordinates() # list of nodes coordinates
        
        # Revert X<>Y coordinates inversion performed by Netgen
        X = mesh_coordinates[:,0].copy()
        Y = mesh_coordinates[:,1].copy()
        mesh_coordinates[:,0] = Y
        mesh_coordinates[:,1] = X
        
        # Interpolate MRI niftis onto mesh nodes and get nodal arrays
        #############################################################
        # Read MRI nifti values onto mesh nodes (get n_nodes numpy array)
        t2_MRI_path = args.seriesofniftis["T2"]
        segmentation_path = args.seriesofniftis["segmentation"]
        diffusion_FA_path = args.seriesofniftis["diffusion_FA"]
        
        # check theoritical values of labels in the input segmentation (to get back on the mesh) 
        import nibabel
        segmentation_image = nibabel.load(segmentation_path)
        data = segmentation_image.get_fdata()
        input_labels = np.unique(data).astype(int)
        print('input labels from the segmentation nifti: {}'.format(input_labels))
        print('labels used to mask the nifti and generate the mesh: [3 4 5 6 7 8 14 15 16 17]')
        
        # Load MRI nifti values onto mesh nodes (get n_nodes numpy array)
        interpolation_mode = args.interpolationmode # 'nearest_neighbor'; 'linear'
        T2intensity_array = transfer_nifti_values_to_mesh_nodes(t2_MRI_path, mesh_coordinates, interpolation_mode)
        segmentation_labels_array = transfer_nifti_values_to_mesh_nodes(segmentation_path, mesh_coordinates, interpolation_mode)
        FA_values_array = transfer_nifti_values_to_mesh_nodes(diffusion_FA_path, mesh_coordinates, interpolation_mode)
        
        # Build FEM Functions from nodal arrays
        #######################################
        # Define FEniCS FEM Function Spaces
        S = fenics.FunctionSpace(mesh, "CG", 1) 
        vertex2dofs_S = fenics.vertex_to_dof_map(S)
        
        # Define Vector functions
        t2 = fenics.Function(S, name="T2")
        segmentation = fenics.Function(S, name="Segmentation")
        fa = fenics.Function(S, name="FA")
        
        # Load nifti values array
        for vertex, scalarDOF in enumerate(vertex2dofs_S):
            t2.vector()[scalarDOF] = T2intensity_array[vertex]
            segmentation.vector()[scalarDOF] = segmentation_labels_array.astype(int)[vertex]
            fa.vector()[scalarDOF] = FA_values_array[vertex]
            
        # Check if each node has a label
        ################################
        list_of_labels = np.unique(segmentation.vector()[:]) 
        
        values_of_labels_transfered_onto_mesh = [] # [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 14.0, 15.0, 16.0, 17.0, 18.0] for mesh_21GW
        number_of_node_per_label = []
        for i, label in enumerate(list_of_labels):
            values_of_labels_transfered_onto_mesh.append(label)
            number_of_node_per_label.append( [label, len(np.where(segmentation_labels_array == label)[0]) ] ) # e.g. [label 1., 4276 nodes]
        
        sum_nodes = 0
        for label, number_of_nodes_with_this_label in number_of_node_per_label:
            sum_nodes += number_of_nodes_with_this_label 
            
        # Change labels to get correct label transfer to mesh 
        #####################################################
        
        ### sum_nodes should be equal to mesh.num_vertices() to be sure all nodes have been labeled.  
        ### labels 0, 1, 2, 9, 10, 18 should not be labeled (since where not used to pre-mask the original nifti image used to generate mesh...). They are present since mesh smoothing + interpolation  
        
        # labels O (background on nifti) should be either label 3, either label 4: use the label of closest nodes 
        # labels 10 (cerebral trunc on nifti) should be either label 3, either label 4: use the label of closest nodes 
        # label 1 --> label 3 (Cortex)
        # label 2 --> label 4 (Cortex)
        
        # label 9 & 18 --> OK. not to remove since within the masked mesh
        
        # labels_to_change = [0., 10., 1., 2.]
        
        ###
        from scipy.spatial import cKDTree
        
        DOF_coords_S = S.tabulate_dof_coordinates()
        tree_S = cKDTree(DOF_coords_S) # the whole mesh

        # Label 0. to change
        # ------------------
        list_of_scalar_dofs_label_0 = np.where(segmentation.vector()[:] == 0.) # number of nodes for which to change label: len(list_of_scalar_dofs_label_0[0])
        for scalar_dof in list(list_of_scalar_dofs_label_0[0]):
            #Find closest node that has non-0 label 
            _, scalarDOF_of_closest_nodes = tree_S.query(DOF_coords_S[scalar_dof], k=10) # returns [distance to closest node, closest node scalar DOF]
            for scalarDOF_closest_node in list(scalarDOF_of_closest_nodes):
                if segmentation.vector()[scalarDOF_closest_node] != 0.:
                    break # stops at first closest node that have label != 0.
            
            # Replace label 0 by the label of closest node which has non-0 label
            segmentation.vector()[scalar_dof] = segmentation.vector()[scalarDOF_closest_node] 
        
        # Label 10. to change
        # -------------------
        list_of_scalar_dofs_label_10 = np.where(segmentation.vector()[:] == 10.) # number of nodes for which to change label: len(list_of_scalar_dofs_label_0[0])
        for scalar_dof in list(list_of_scalar_dofs_label_10[0]):
            #Find closest node that has non-10 label 
            _, scalarDOF_of_closest_nodes = tree_S.query(DOF_coords_S[scalar_dof], k=10) # returns [distance to closest node, closest node scalar DOF]
            for scalarDOF_closest_node in list(scalarDOF_of_closest_nodes):
                if segmentation.vector()[scalarDOF_closest_node] != 10.:
                    break # stops at first closest node that have label != 10.
            
            # Replace label 10 by the label of closest node which has non-10 label
            segmentation.vector()[scalar_dof] = segmentation.vector()[scalarDOF_closest_node] 
            
        ###
        # Label 1. to change
        # ------------------        
        list_of_scalar_dofs_label_1 = np.where(segmentation.vector()[:] == 1.)
        for scalar_dof in list_of_scalar_dofs_label_1:
            segmentation.vector()[scalar_dof] = 3.
        
        # Label 2. to change
        # ------------------
        list_of_scalar_dofs_label_2 = np.where(segmentation.vector()[:] == 2.)
        for scalar_dof in list_of_scalar_dofs_label_2:
            segmentation.vector()[scalar_dof] = 4.
        
        
        # Visualize FEM Functions
        #########################
        
        # vedo.dolfin.plot(t2, style='paraview').clear()  # style=meshlab; bw; matplotlib
                        
        # vedo.dolfin.plot(segmentation, style='paraview').clear()  # style=meshlab; bw; matplotlib
        
        # vedo.dolfin.plot(fa, style='paraview').clear()  # style=meshlab; bw; matplotlib

        
        # Export FEM Functions
        ######################
        # Export nodal values to .xdmf readable in Paraview
        result_file = fenics.XDMFFile(args.outputfolder + args.outputfilename)
        result_file.parameters["flush_output"] = True
        result_file.parameters["functions_share_mesh"] = True
        result_file.parameters["rewrite_function_mesh"] = False

        # Saving mesh + values
        result_file.write(t2, 0.0)
        result_file.write(segmentation, 0.0)
        result_file.write(fa, 0.0) """
    
    
    elif inputmesh_format == 'vtk':
        # Input mesh
        ############
        mesh = meshio.read(args.inputmesh)
        mesh_coordinates = mesh.points
        
        # Revert X<>Y coordinates inversion performed by Netgen
        X = mesh_coordinates[:,0].copy()
        Y = mesh_coordinates[:,1].copy()
        mesh_coordinates[:,0] = Y
        mesh_coordinates[:,1] = X

        # Interpolate MRI niftis onto mesh nodes and get nodal arrays
        #############################################################
        # Read MRI nifti values onto mesh nodes (get n_nodes numpy array)
        t2_MRI_path = args.seriesofniftis["T2"]
        segmentation_path = args.seriesofniftis["segmentation"]
        diffusion_FA_path = args.seriesofniftis["diffusion_FA"]
        
        # check theoritical values of labels in the input segmentation (to get back on the mesh) 
        import nibabel
        segmentation_image = nibabel.load(segmentation_path)
        data = segmentation_image.get_fdata()
        input_labels = np.unique(data).astype(int)
        print('input labels from the segmentation nifti: {}'.format(input_labels))
        print('labels used to mask the nifti and generate the mesh: [3 4 5 6 7 8 14 15 16 17]')
        
        # Load MRI nifti values onto mesh nodes (get n_nodes numpy array)
        interpolation_mode = args.interpolationmode # 'nearest_neighbor'; 'linear'
        T2intensity_array = transfer_nifti_values_to_mesh_nodes(t2_MRI_path, mesh_coordinates, interpolation_mode)
        segmentation_labels_array = transfer_nifti_values_to_mesh_nodes(segmentation_path, mesh_coordinates, interpolation_mode)
        FA_values_array = transfer_nifti_values_to_mesh_nodes(diffusion_FA_path, mesh_coordinates, interpolation_mode)
        
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
        mesh.write(args.outputfolder + args.outputfilename) 
    
    
    elif inputmesh_format == 'stl':
        # Input mesh
        ############
        mesh = meshio.read(args.inputmesh)
        mesh_coordinates = mesh.points

        # Interpolate MRI niftis onto mesh nodes and get nodal arrays
        #############################################################
        # Read MRI nifti values onto mesh nodes (get n_nodes numpy array)
        t2_MRI_path = args.seriesofniftis["T2"]
        segmentation_path = args.seriesofniftis["segmentation"]
        diffusion_FA_path = args.seriesofniftis["diffusion_FA"]
        
        # Load MRI nifti values onto mesh nodes (get n_nodes numpy array)
        interpolation_mode = args.interpolationmode # 'nearest_neighbor'; 'linear'
        T2intensity_array = transfer_nifti_values_to_mesh_nodes(t2_MRI_path, mesh_coordinates, interpolation_mode)
        segmentation_labels_array = transfer_nifti_values_to_mesh_nodes(segmentation_path, mesh_coordinates, interpolation_mode)
        FA_values_array = transfer_nifti_values_to_mesh_nodes(diffusion_FA_path, mesh_coordinates, interpolation_mode)
        
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
        mesh.write(args.outputfolder + args.outputfilename)             
        
