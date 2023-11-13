import fenics
import numpy as np
import itk
import argparse
import json
import vedo.dolfin

import sys
sys.path.append(".")
from utils.nifti_mesh.spatialorientationadapter_to_ras import apply_lps_ras_transformation, nifti_itk_analyzer, itk_coordinate_orientation_system_analyzer


# If input mesh with associated reference MRI nifti (brain) 
# ---------------------------------------------------------
def transfer_nifti_values_to_mesh_nodes(nii_path, mesh_coordinates):
    """ 
    Interpolation of the input nifti values (segmentation label, diffusion (FA), etc.) over all mesh nodes. 

    Parameters:
    mesh_coordinates (numpy array: (number_nodes, 3)): Sparse 3D nodes coordinates
    reference_nii_path (string): pathway to the reference nifti image

    Returns: 
    numpy array (number of nodes): Interpolated values from nifti for each corresponding mesh node.
    """
    
    nii_img_itk = itk.imread(nii_path)

    # Reorient the MRI reference nifti from LPS+ coordinate system (itk convention) to RAS+ if coordinates are in RAS+ 
    nii_img_itk_ras = apply_lps_ras_transformation(nii_img_itk)

    # 
    niivalues_interpolator = itk.BSplineInterpolateImageFunction.New(nii_img_itk) # NearestNeighborInterpolateImageFunction; LinearInterpolateImageFunction; BSplineInterpolateImageFunction'

    # Interpolate the initial mri values to the coordinates (ijk space)
    values_loaded_to_mesh = np.zeros(len(mesh_coordinates)) # MRI intensity values
    for i in range(len(mesh_coordinates)):
        coordinates_in_image_system = nii_img_itk_ras.TransformPhysicalPointToContinuousIndex(mesh_coordinates[i]) #find the closest pixel to the vertex[i] (continuous index)
        values_loaded_to_mesh[i] = niivalues_interpolator.EvaluateAtContinuousIndex(coordinates_in_image_system) #interpolates value around the index et attribute the interpolation value to the associated mesh node

    return values_loaded_to_mesh  


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='NiftiToMesh: Transfer MRI values from nifti to mesh (e.g. Cortex label, FA)')
    
    parser.add_argument('-m', '--inputmesh', help='Path to input FEniCS readable formal .xml mesh)', type=str, required=False, 
                        default='./data/dhcp/dhcp_atlas/21GW/dhcp21GW_29296faces_107908tets.xml') # './data/dhcp/refined/dhcp21GW_17K_refined10.xml'
    
    """
    parser.add_argument('-nii', '--nifti', help='Path to the MRI nifti that contain values to transfer onto mesh', type=str, required=False, 
                        default='/home/latim/BrainGrowth_database/dhcp_fetal_atlas/fetal_brain_mri_atlas/parcellations/tissue-t21.00_dhcp-19.nii.gz')
    """
    
    parser.add_argument('-niis', '--seriesofniftis', help='Path to the orginal nifti file (.nii) + Path to the associated segmentation file (.nii)', type=json.loads, required=False, 
                        default={ 
                                 
                                 "T2":'/home/anne/Bureau/GitHub/dhcp_atlas/t2-t21.00.nii.gz',
                                 
                                 "segmentation":'/home/anne/Bureau/GitHub/dhcp_atlas/tissue-t21.00_dhcp-19.nii.gz',
                                 
                                 "diffusion_FA": '/home/anne/Bureau/GitHub/dhcp_atlas/fa-t21.00.nii.gz'
                                 
                                 } )
    
    
    parser.add_argument('-o', '--output', help='Path to output folder where to write mesh + loaded nifti values (.xdmf))', type=str, required=False, 
                        default='./utils/nifti_mesh/niftitomesh/transfer_niftivalues_to_meshnodes/results/')
    
    args = parser.parse_args()


    # Get mesh nodes coordinates from FEniCS mesh
    mesh = fenics.Mesh(args.inputmesh)
    """
    vedo.dolfin.plot(mesh, 
                     mode='mesh', 
                     wireframe=True, 
                     style='meshlab').clear() 
    """
    
    mesh_coordinates = mesh.coordinates() # list of nodes coordinates

 
    # Read MRI nifti values onto mesh nodes (get n_nodes numpy array)
    t2_MRI_path = args.seriesofniftis["T2"]
    segmentation_path = args.seriesofniftis["segmentation"]
    diffusion_FA_path = args.seriesofniftis["diffusion_FA"]
    
    segmentation_img = itk.imread(segmentation_path)
    fa_img = itk.imread(diffusion_FA_path)
    
    # Load MRI nifti values onto mesh nodes (get n_nodes numpy array)
    T2intensity_array = transfer_nifti_values_to_mesh_nodes(t2_MRI_path, mesh_coordinates)
    segmentation_labels_array = transfer_nifti_values_to_mesh_nodes(segmentation_path, mesh_coordinates)
    FA_values_array = transfer_nifti_values_to_mesh_nodes(diffusion_FA_path, mesh_coordinates)
    
    # Visualize nifti values onto mesh with FEniCS
    # --------------------------------------------
    # Define FEniCS FEM Function Spaces
    S = fenics.FunctionSpace(mesh, "CG", 1) 
    #V = fenics.VectorFunctionSpace(mesh, "CG", 1)
    
    # Define Vector functions
    t2 = fenics.Function(S, name="T2")
    segmentation = fenics.Function(S, name="Segmentation")
    fa = fenics.Function(S, name="FA")
    
    # Load nifti values array
    t2.vector()[:] = T2intensity_array[:]
    segmentation.vector()[:] = segmentation_labels_array[:]
    fa.vector()[:] = FA_values_array[:]
    
    # Visualization
    """
    vedo.dolfin.plot(t2, 
                     style='paraview').clear()  # style=meshlab; bw; matplotlib
                     
    vedo.dolfin.plot(segmentation, 
                     style='paraview').clear()  # style=meshlab; bw; matplotlib
    
    vedo.dolfin.plot(fa, 
                     style='paraview').clear()  # style=meshlab; bw; matplotlib
    """
    
    # EXPORT
    # ------
    # Export nodal values to .xdmf readable in Paraview
    result_file = fenics.XDMFFile(args.output + "mesh_loaded_with_MRInifti_values.xdmf")
    result_file.parameters["flush_output"] = True
    result_file.parameters["functions_share_mesh"] = True
    result_file.parameters["rewrite_function_mesh"] = False

    # Saving mesh + values
    result_file.write(t2, 0.0)
    result_file.write(segmentation, 0.0)
    result_file.write(fa, 0.0)
    
    