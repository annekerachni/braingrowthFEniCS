"""
Visualize structural nifti (T1, T2) to mask
Visualize associated segmentation + parcels values (parcels values to be used in 'mask_nifti_xith_segmentation_parcels.py')
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


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


def show_slices(slices, colormap): # code source: https://nipy.org/nibabel/coordinate_systems.html
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap=colormap, origin="lower") # structural --> cmap="gray"; segmentation --> cmap="nipy_spectral", "jet", "gist_ncar", "viridis"
    return



import argparse
import json
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Mask brain MRI nifti with selected parcels from provided segmentation')
    
    parser.add_argument('-i', '--inputdata', help='Path to the orginal nifti file (.nii) + Path to the associated segmentation file (.nii)', type=json.loads, required=True, 
                        default={ 
                                 
                                 "nifti_T1":'./fetal_database/structural/t1-t21.00.nii.gz',
                                 
                                 "nifti_T2":'./fetal_database/structural/t2-t21.00.nii.gz',
                                 
                                 "segmentation":'./fetal_database/parcellations/tissue-t21.00_dhcp-19.nii.gz'
                                 
                                 } )
    
    
    args = parser.parse_args() 
    
    # load data
    ###########
    nifti_T1_path = args.inputdata["nifti_T1"]
    nifti_T2_path = args.inputdata["nifti_T2"]
    segmentation_path = args.inputdata["segmentation"]

    # Get the data of the original structural nifti
    ###############################################
    # read original nifti
    img_nib_T1, img_nib_data_T1, affine_T1, header_T1, shape_T1 = read_nifti(nifti_T1_path)
    img_nib_T2, img_nib_data_T2, affine_T2, header_T2, shape_T2 = read_nifti(nifti_T2_path)
    #print("\nshapeX:{}, shapeY:{}, shapeZ:{}".format(shape[0], shape[1], shape[2])) # dhcp 3D atlas: 180, 221, 180
    
    # read associated segmentation
    img_nib_seg, img_nib_data_seg, affine_seg, header_seg, shape_seg = read_nifti(segmentation_path)

    # Visualize structural and segmentation niftis 
    ##############################################
    # T1
    # --
    slice_x_T1 = int(shape_T1[0]/2) # 70
    slice_y_T1 = int(shape_T1[1]/2)
    slice_z_T1 = int(shape_T1[2]/2) # 120
    show_slices([   img_nib_data_T1[slice_x_T1, :, :], 
                    img_nib_data_T1[:, slice_y_T1, :],
                    img_nib_data_T1[:, :, slice_z_T1]],
                "gray") 
    plt.suptitle("dhcp nifti T1. \nSlices: {}, {}, {}.".format(slice_x_T1, slice_y_T1, slice_z_T1))  
    plt.show()
    
    # T2
    # --
    slice_x_T2 = int(shape_T2[0]/2) # 70
    slice_y_T2 = int(shape_T2[1]/2)
    slice_z_T2 = int(shape_T2[2]/2) # 120
    show_slices([   img_nib_data_T2[slice_x_T2, :, :], 
                    img_nib_data_T2[:, slice_y_T2, :],
                    img_nib_data_T2[:, :, slice_z_T2]],
                "gray") 
    plt.suptitle("dhcp nifti T2. \nSlices: {}, {}, {}.".format(slice_x_T2, slice_y_T2, slice_z_T2))    
    plt.show()
    
    # segmentation
    # ------------
    slice_x_seg = int(shape_seg[0]/2) # 70
    slice_y_seg = int(shape_seg[1]/2)
    slice_z_seg = int(shape_seg[2]/2) # 120
    show_slices([   img_nib_data_seg[slice_x_seg, :, :], 
                    img_nib_data_seg[:, slice_y_seg, :],
                    img_nib_data_seg[:, :, slice_z_seg]],
                "nipy_spectral") 
    plt.suptitle("segmentation and parcel values. \nSlices: {}, {}, {}.".format(slice_x_seg, slice_y_seg, slice_z_seg))    
    plt.show()
        
        
        