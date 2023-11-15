import fenics
import numpy as np
import itk
import argparse
import json
import meshio
import vedo.dolfin

import sys
sys.path.append(".")
from utils.nifti_mesh.spatialorientationadapter_to_ras import apply_lps_ras_transformation, nifti_itk_analyzer, itk_coordinate_orientation_system_analyzer


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
            ijk = image_itk_RAS.TransformPhysicalPointToContinuousIndex((float(xyz[i][0]), float(xyz[i][1]), float(xyz[i][2]))) #find the closest pixel to the vertex[i] (continuous index)
            values_loaded_to_mesh[i] = nearest_neighbor_interpolator.EvaluateAtContinuousIndex(ijk) #interpolates value around the index et attribute the interpolation value to the associated mesh node            
    
    return values_loaded_to_mesh # MRI nifti values converted into a numpy array (size:mesh n_nodes)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Transfer MRI values (e.g. Segmentation, FA) from nifti to mesh')
    
    parser.add_argument('-m', '--inputmesh', help='Path to input FEniCS readable formal .xml mesh)', type=str, required=False, 
                        default='./data/dhcp/dhcp_atlas/21GW/dhcp21GW_29296faces_107908tets.xml')     

    parser.add_argument('-niis', '--seriesofniftis', help='Path to the orginal nifti file (.nii) + Path to the associated segmentation file (.nii)', type=json.loads, required=False, 
                        default={ 
                                 
                                 "T2":'./dhcp_fetal_atlas/fetal_brain_mri_atlas/structural/t2-t21.00.nii.gz',
                                 
                                 "segmentation":'./dhcp_fetal_atlas/fetal_brain_mri_atlas/parcellations/tissue-t21.00_dhcp-19.nii.gz',
                                 
                                 "diffusion_FA": './dhcp_fetal_atlas/fetal_brain_mri_atlas/diffusion/fa-t21.00.nii.gz'
                                 
                                 } )
    
    parser.add_argument('-im', '--interpolationmode', help='Interpolation mode (nearest_neighbor; linear)', type=str, required=False, 
                        default='nearest_neighbor')
    
    parser.add_argument('-of', '--outputfolder', help='Path to output folder where to write mesh + loaded nifti values)', type=str, required=False, 
                        default='./utils/nifti_mesh/niftitomesh/transfer_niftivalues_to_meshnodes/results/')
    
    parser.add_argument('-ofn', '--outputfilename', help='Output file name (.xdmf; .vtk))', type=str, required=False, 
                        default='meshXML_MRIniivalues_NearestNeighbor_21GW.xdmf')
    
    args = parser.parse_args()


    # Input mesh format
    ###################
    inputmesh_format = args.inputmesh.split('.')[-1]
    
    if inputmesh_format == 'xml':
    	# Input mesh
    	############
        mesh = fenics.Mesh(args.inputmesh)
        """
        vedo.dolfin.plot(mesh, 
                        mode='mesh', 
                        wireframe=True, 
                        style='meshlab').clear() 
        """
        
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
            segmentation.vector()[scalarDOF] = segmentation_labels_array[vertex]
            fa.vector()[scalarDOF] = FA_values_array[vertex]
        
        # Visualize FEM Functions
        #########################
        """
        vedo.dolfin.plot(t2, style='paraview').clear()  # style=meshlab; bw; matplotlib
                        
        vedo.dolfin.plot(segmentation, style='paraview').clear()  # style=meshlab; bw; matplotlib
        
        vedo.dolfin.plot(fa, style='paraview').clear()  # style=meshlab; bw; matplotlib
        """
        
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
        result_file.write(fa, 0.0)
    
    
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
            
        # Export .vtk
        #############
        mesh.write(args.outputfolder + args.outputfilename) 
            
        
