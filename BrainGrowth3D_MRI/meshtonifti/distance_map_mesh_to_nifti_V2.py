import fenics
import sys, os
import argparse
from scipy.spatial import cKDTree

sys.path.append(sys.path[0]) # meshtonifti 
sys.path.append(os.path.dirname((os.path.dirname(sys.path[0])))) # braingrowthFEniCS

from FEM_biomechanical_model import mappings, differential_layers, projection, preprocessing
from mesh2nifti_image import transfer_reference_MRI_values_to_mesh_nodes, generate_mriref_nifti_from_meshvalues

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build signed distance map to the brain surface in nifti format from brain mesh')

    parser.add_argument('-i', '--input', help='Input mesh path (.xml, .xdmf); mesh unit: millimeters', type=str, required=False, 
                        default="./data/dHCP_surface_vs_volume_RAW/Vf/dhcp_surface_t21_raw_130000faces_455983tets_reoriented_dHCPVolume.xdmf") 
                        # from dHCP volume niftis: "./data/dHCP_surface_vs_volume_RAW/Vf/dhcp_surface_t21_raw_130000faces_455983tets_reoriented_dHCPVolume.xdmf"--> in millimeters
                        # from dHCP surface gifti mesh: "./data/dHCP_surface_vs_volume_RAW/Vf/dhcp_volume_t21_raw_130000faces_480112tets.xdmf" -->  in millimeters

    parser.add_argument('-c', '--convertmesh0frommetersintomillimeters', help='Convert mesh from meters into millimeters', type=bool, required=False, default=False)
    # only for fetal_week21_left_right_merged_V2 (in meters while niftis affine are in millimeters)
    
    parser.add_argument('-nii', '--niftipath', help='Path to the original nifti (FA, parcellation) to be deformed from dHCP volume into dHCP surface reference', type=str, required=False, 
                        default='./data/dHCP_surface_vs_volume_RAW/fa-t21.00.nii.gz') 
                        # './data/21_28_36GW/transformed_niftis_meshes/transformed-fa-t21.00.nii.gz' 
                        # './data/21_28_36GW/transformed_niftis_meshes/transformed-tissue-t21.00_dhcp-19.nii.gz'
                        # N.B. At this juncture, the images are necessary for the generation of a distance map that is referenced in the same manner (resolution and affine) as the NIFTI file, which will subsequently be deformed.
    
    parser.add_argument('-on', '--outputnifti', help='Output folder path', type=str, required=False, 
                        default='./results/distance_map_dhcp_surface_t21_raw.nii.gz') 
                           
    #parser.add_argument('-v', '--visualization', help='Visualization during simulation', type=bool, required=False, default=False)
    parser.add_argument('-v', '--visualization', help='Visualization during simulation', action='store_true')
    
    args = parser.parse_args()

# import mesh
#############
inputmesh_path = args.input # in millimeters
inputmesh_format = inputmesh_path.split('.')[-1]

if inputmesh_format == "xml":
    mesh = fenics.Mesh(inputmesh_path)

elif inputmesh_format == "xdmf":
    mesh = fenics.Mesh()
    with fenics.XDMFFile(inputmesh_path) as infile:
        infile.read(mesh)

if args.convertmesh0frommetersintomillimeters == True:
    mesh = preprocessing.converting_mesh_from_meters_into_millimeters(mesh)
    
mesh_coordinates = mesh.coordinates()

# compute boundary mesh
#######################
bmesh = fenics.BoundaryMesh(mesh, "exterior")

bmesh_cortexsurface_bbtree = fenics.BoundingBoxTree()
bmesh_cortexsurface_bbtree.build(bmesh) 

tree_bmesh = cKDTree(bmesh.coordinates()) # the whole mesh

# FEM space and function
########################
#S = fenics.FunctionSpace(mesh, "CG", 1) 
#vertex2dofs_S = mappings.vertex_to_dof_ScalarFunctionSpace(S)
#d2s = fenics.Function(S, name="d2s")

# Get nifti information (affine, origin, resolution)
####################################################
import itk
from spatialorientationadapter_to_ras import apply_lps_ras_transformation
import nibabel as nib
import numpy as np
from numba import prange

"""
nii_img_itk = itk.imread(args.niftipath)
nii_img_itk_ras = apply_lps_ras_transformation(nii_img_itk) # reorient the MRI reference nifti from LPS+ coordinate system (itk convention) to RAS+ if coordinates are in RAS+ 
"""

img_nib = nib.load(args.niftipath)  
img_nib_data = img_nib.get_fdata()
affine = img_nib.affine    
header = img_nib.header # print(header)
shape = np.shape(img_nib) 

def voxel_to_world(affine, i, j, k):
    voxel_coords = np.array([i, j, k, 1])
    world_coords = affine.dot(voxel_coords)
    return world_coords[:3]
    
# compute distance to surface for each mesh node
################################################
#d2s_ = differential_layers.compute_distance_to_cortexsurface(vertex2dofs_S, d2s, mesh, bmesh_cortexsurface_bbtree) # init at t=0.0
#for idx, x in enumerate(mesh.coordinates()): # point: idx (x[0]), coords (x[1])

for k in prange(len(img_nib_data)): # explore first z shape of the image
    for j in prange(len(img_nib_data[k])): # then, explore y shape of the image
        for i in prange(len(img_nib_data[k][j])): # at last step, explore x shape of the image

            #x,y,z = nii_img_itk_ras.TransformContinuousIndexToPhysicalPoint((i,j,k))
            x, y, z = voxel_to_world(affine, i, j, k) 
    
            #idx = np.where(mesh_coordinates[:] == x,y,z)[0]
                    
            #_, distance_to_cortexsurface_ijk = bmesh_cortexsurface_bbtree.compute_closest_entity(fenics.Point(*(x,y,z))) 
            distance_to_cortexsurface_ijk, _ = tree_bmesh.query((y, x, z)) # in meshes, X<>Y coordinates inversion performed by Netgen
            #d2s.vector()[vertex2dofs_S[idx]] = distance_to_cortexsurface_x
        
            img_nib_data[i, j, k] = distance_to_cortexsurface_ijk # img_nib_data[i, j, k]
        
#projection.local_project(d2s_, S, d2s)

# export
########
new_img = nib.Nifti1Image(img_nib_data, affine, header)
nib.save(new_img, args.outputnifti)

print("The distance map of the mesh (.nii.gz) was well registered.")

"""
outputpath = os.path.join(args.outputmesh)
FEniCS_FEM_Functions_file = fenics.XDMFFile(outputpath)
FEniCS_FEM_Functions_file.parameters["flush_output"] = True
FEniCS_FEM_Functions_file.parameters["functions_share_mesh"] = True
#FEniCS_FEM_Functions_file.parameters["rewrite_function_mesh"] = False

FEniCS_FEM_Functions_file.write(d2s, 0.0)
"""

# get distance map in the image reference (i,j,k) (nifti format)
################################################################
"""
from spatialorientationadapter_to_ras import apply_lps_ras_transformation

reference_mri_values = transfer_reference_MRI_values_to_mesh_nodes(args.niftipath, mesh_coordinates)
generate_mriref_nifti_from_meshvalues(mesh_coordinates, reference_mri_values, args.niftipath, 'linear', args.outputnifti)
"""

