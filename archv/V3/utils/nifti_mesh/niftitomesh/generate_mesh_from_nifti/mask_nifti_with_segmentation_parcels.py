"""
Input data: brain MRI nifti + segmentation parcels
Description: Mask the original nifti with selected parcels of interest (e.g. only keep Cortex and White Matter)
Output: 
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from numba import prange, jit


def read_nifti(img_path):
    """
    Read brain MRI nifti and get related informations. 
    """
    img_nib = nib.load(img_path)  
    img_nib_data = img_nib.get_fdata()
    affine = img_nib.affine    
    header = img_nib.header # print(header)
    shape = np.shape(img_nib) 
    #shape_i, shape_j, shape_k = shape[0], shape[1], shape[2]

    return img_nib, img_nib_data, affine, header, shape 

@jit(parallel=True, forceobj=True)
def generate_binary_mask_from_selected_parcels(segmentation_file_path, selected_parcels): 
    """
    Generate binary mask from the selected parcels among the provided segmentation parcels. 
    """
    provided_parcels_nib = nib.load(segmentation_file_path)  
    provided_parcels_nib_data = provided_parcels_nib.get_fdata()
    
    binary_mask_data = provided_parcels_nib_data.copy()

    # Generate binary mask from parcels on interest regions (cortex & WM)
    print("\ngenerating binary mask from selected brain region parcels...")
    for k in prange(len(provided_parcels_nib_data)): # explore first z shape of the image
        for j in prange(len(provided_parcels_nib_data[k])): # then, explore y shape of the image
            for i in prange(len(provided_parcels_nib_data[k][j])): # at last step, explore x shape of the image

                if provided_parcels_nib_data[k][j][i] in selected_parcels:
                    binary_mask_data[k][j][i] = 1.

                else:
                    binary_mask_data[k][j][i] = 0.
    
    return binary_mask_data

def apply_binary_mask_to_nifti(img_nib_data, binary_mask_data):
    img_masked_data = img_nib_data * binary_mask_data
    return img_masked_data

def write_masked_nifti(img_masked_data, affine, masked_img_output_path):
    img_masked = nib.Nifti1Image(img_masked_data, affine=affine) 
    nib.save(img_masked, masked_img_output_path)  
    return

def show_slices(slices): # code source: https://nipy.org/nibabel/coordinate_systems.html
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    return


import argparse
import json
if __name__ == '__main__':
    
    """
    python3 -i mask_nifti_with_segmentation_parcels.py 
            -i { "nifti": './database/dhcp_fetal_atlas/fetal_brain_mri_atlas/structural/t2-t21.00.nii.gz',
                 "segmentation": './database/dhcp_fetal_atlas/fetal_brain_mri_atlas/parcellations/tissue-t21.00_dhcp-19.nii.gz'}
            -hw 'whole'
            -par '{ "left": {"cortex": 3., "wm": 5., "caudate_putamen_globuspallidus": 14., "thalamus": 16., "ventricles": 7.},
                    "right": {"cortex": 4., "wm": 6., "caudate_putamen_globuspallidus": 15., "thalamus": 17., "ventricles": 8.} }'
            - o './data/21GW/dhcp21GW_masked_keeping_Cortex_WM_CaudatePutamenGlobuspallidus_Thalamus_Ventricules.nii.gz'
            - d True
    """

    parser = argparse.ArgumentParser(description='Mask brain MRI nifti with selected parcels from provided segmentation')
    
    parser.add_argument('-i', '--inputdata', help='Path to the orginal nifti file (.nii) + Path to the associated segmentation file (.nii)', type=json.loads, required=False, 
                        default={ 
                                 
                                 "nifti":'./database/dhcp_fetal_atlas/fetal_brain_mri_atlas/structural/t2-t21.00.nii.gz',
                                 
                                 "segmentation":'./database/dhcp_fetal_atlas/fetal_brain_mri_atlas/parcellations/tissue-t21.00_dhcp-19.nii.gz'
                                 
                                 } )
    
    parser.add_argument('-hw', '--halforwholebrain', help='Half or whole brain: choose between "whole"; "left" or "right"', type=str, required=False, 
                        default='whole')
    
    parser.add_argument('-par', '--segmentationparcelstokeep', help='Brain regions and associated value of the segmentation parcels to keep', type=json.loads, required=False, 
                        default={   
                                    "left":{
                                            "cortex": 3.,
                                            "wm": 5.,
                                            "caudate_putamen_globuspallidus": 14.,
                                            "thalamus": 16.,
                                            "ventricles": 7.
                                            },

                                    "right":{
                                            "cortex": 4.,
                                            "wm": 6.,
                                            "caudate_putamen_globuspallidus": 15.,
                                            "thalamus": 17.,
                                            "ventricles": 8.
                                            }   
                                
                                }) # e.g. dhcp segmentation parcels (See data from https://gin.g-node.org/kcl_cdb/fetal_brain_mri_atlas/src/master/parcellations)
    
    parser.add_argument('-o', '--outputmaskednifti', help='Path to masked nifti (.nii.gz)', type=str, required=False, 
                        default='./data/dhcp/dhcp_atlas/21GW/dhcp21GW_masked.nii.gz')
    
    parser.add_argument('-d', '--displaymode', help='Display nifti before and after masking', type=bool, required=False, 
                        default=True)
    
    args = parser.parse_args() 
    
    # load data
    ###########
    nifti_path = args.inputdata["nifti"]
    segmentation_path = args.inputdata["segmentation"]

    # Get the data of the original structural nifti
    ###############################################
    # read original nifti
    img_nib, img_nib_data, affine, header, shape = read_nifti(nifti_path)
    #print("\nshapeX:{}, shapeY:{}, shapeZ:{}".format(shape[0], shape[1], shape[2])) # dhcp 3D atlas: 180, 221, 180

    # display slices of orginal nifti 
    if args.displaymode == True:
        slice_x = 70 #int(shape[0]/2) # 60
        slice_y = int(shape[1]/2)
        slice_z = int(shape[2]/2) # 120
        show_slices([img_nib_data[slice_x, :, :], 
                        img_nib_data[:, slice_y, :],
                        img_nib_data[:, :, slice_z]]) 
        plt.suptitle("Structural dhcp nifti. \nSlices: {}, {}, {}.".format(slice_x, slice_y, slice_z))  
        plt.show()

    # Apply mask to nifti using selected parcels from the segmentation
    ##################################################################
    # dhcp segmentation parcels (See data from https://gin.g-node.org/kcl_cdb/fetal_brain_mri_atlas/src/master/parcellations)
    dhcp_parcels = args.segmentationparcelstokeep # dictionnary
    
    # select labels to apply
    selected_parcels = []

    if args.halforwholebrain == 'whole':
        for brain_hemisphere, set_of_labels in dhcp_parcels.items():
            for brain_region, label in set_of_labels.items():
                selected_parcels.append(label)

    elif args.halforwholebrain == 'left':
        for brain_region, label in dhcp_parcels["left"].items():
            selected_parcels.append(label)

    elif args.halforwholebrain == 'right':
        for brain_region, label in dhcp_parcels["right"].items():
            selected_parcels.append(label)

    # define and compute labels of interest (e.g. left cortex + left WM)
    binary_mask_data = generate_binary_mask_from_selected_parcels(segmentation_path, selected_parcels)

    # apply selected binar parcellation to original nifti 
    img_masked_data = apply_binary_mask_to_nifti(img_nib_data, binary_mask_data)

    # Check the masked nifti
    ########################
    # display slices of segmented nifti (quality check)
    if args.displaymode == True:
        show_slices([img_masked_data[slice_x, :, :],
                     img_masked_data[:, slice_y, :],
                     img_masked_data[:, :, slice_z]])
        plt.suptitle("Masked dhcp nifti. \nSlices: {}, {}, {}.".format(slice_x, slice_y, slice_z))  
        plt.show()

    # Save the masked nifti
    #######################   
    write_masked_nifti(img_masked_data, affine, args.outputmaskednifti)

    print("nifti was masked and saved.")