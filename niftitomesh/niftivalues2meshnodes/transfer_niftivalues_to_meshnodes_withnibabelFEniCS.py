import fenics
import numpy as np
import argparse
import json
import vedo.dolfin
import nibabel
from nibabel.affines import apply_affine
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))  # braingrowthFEniCS
from FEM_biomechanical_model import mappings

"""
def get_vox2ras_tkr(image):
        
        ds = image.header._structarr['pixdim'][1:4]
        ns = image.header._structarr['dim'][1:4] * ds / 2.0
        
        vox2ras = np.array([[-ds[0], 0, 0, ns[0]],
                            [0, 0, ds[2], -ns[2]],
                            [0, -ds[1], 0, ns[1]],
                            [0, 0, 0, 1]],
                            dtype=np.float64                           
                            )
        
        return vox2ras
"""

def transfer_nifti_values_to_FEnicSmesh_nodes(MRI_nifti_path, MRIniftiname, coordinates): # mesh --> must be a fenics.Mesh()
    """
    source: https://www.researchgate.net/publication/358873266_Working_with_magnetic_resonance_images_of_the_brain
    """
    
    # MRI nifti pre-processing
    # ------------------------
    """
    image = itk.imread(MRI_nifti_path)
    image = apply_lps_ras_transformation(image) # Reorient the MRI reference nifti from LPS+ coordinate system (itk convention) to RAS+ if coordinates are in RAS+ 
    """
    image = nibabel.load(MRI_nifti_path)
    
    resolution = image.header.get_zooms()
    affine = image.affine # complete matrix: affine + origin
    data = image.get_fdata()
    #print(data.shape)
    
    # .xml mesh pre-processing
    # ------------------------
    n = mesh.topology().dim()
    regions = fenics.MeshFunction("size_t", mesh, n, 0)
    
    # Get the specific mesh coordinates array to use as input of inverse affine matrix https://nipy.org/nibabel/coordinate_systems.html (world space)
    xyz1 = np.concatenate((coordinates, np.ones((coordinates.shape[0], 1))), axis=1)

    # Transform mesh world coordinates to image coordinates: apply inverse affine matrix (image space)
    matrix_wc_2_img = np.linalg.inv(affine) # = ras2vox
    ijk1 = (matrix_wc_2_img.dot(xyz1.T)).T # ijk1[:,0:3]

    
    # 1.
    S = fenics.FunctionSpace(mesh, "CG", 1) 
    vertex2dofs_S = fenics.vertex_to_dof_map(S)
    functionFEM = fenics.Function(S, name=MRIniftiname) # name=name
    
    for vertex, scalarDOF in enumerate(vertex2dofs_S):
        i, j, k = np.rint(ijk1[vertex, 0:3]).astype("int") 
        if i < data.shape[0] and j < data.shape[1] and k < data.shape[2] : # maybe because of mesh smoothing, max(ijk) in X direction overcomes X shape of nifti image --> exclude outer nodes? 
            functionFEM.vector()[scalarDOF] = int(data[i, j, k])
    
    #vedo.dolfin.plot(functionFEM, style='paraview').clear()  # style=meshlab; bw; matplotlib
    
    """
    for vertex in range(len(mesh.coortinates())):
        i, j, k = np.rint(ijk1[vertex, 0:3]).astype("int")
        int(data[i, j, k])
    """
    # 2.
    #vox2ras = get_vox2ras_tkr(image)
    #ras2vox = np.linalg.inv(vox2ras)
    
    # Iterating over all cells
    for cell in fenics.cells(mesh):
        c = cell.index()
        xyz = cell.midpoint()[:] # Extract RAS coordinates of cell midpoint
        xyz1 = list(xyz)
        xyz1.append(1.)
        #ijk = apply_affine(ras2vox, xyz) # # Convert to voxel space np.dot(matrix_wc_2_img[:3,:3], xyz)
        ijk1 = np.dot(matrix_wc_2_img, np.array(xyz1))
        i, j, k = np.rint(ijk1[0:3]).astype("int") # Round off to nearest integers to find voxel indices
        if i < data.shape[0] and j < data.shape[1] and k < data.shape[2] : 
            regions.array()[c] = int(data[i, j, k]) # Insert image data into the mesh function
    

    return regions, functionFEM


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Transfer MRI values (e.g. Segmentation, FA) from nifti to mesh (using nibabel and FEniCS .xml mesh)')
    
    parser.add_argument('-m', '--inputmesh', help='Path to input mesh (.xml, .xdmf, .stl)', type=str, required=True, 
                        default='./data/brainmesh.xdmf') 
    
    parser.add_argument('-niis', '--seriesofniftis', help='Path to the orginal nifti file (.nii) + Path to the associated segmentation file (.nii)', type=json.loads, required=True, 
                        default={ 
                                 
                                 "T2":'./fetal_database/structural/t2-t21.00.nii.gz',
                                 
                                 "segmentation":'./fetal_database/parcellations/tissue-t21.00_dhcp-19.nii.gz',
                                 
                                 "diffusion_FA": './fetal_database/diffusion/fa-t21.00.nii.gz'
                                 
                                 } )
    
    
    parser.add_argument('-o', '--output', help='Path to output folder where to write mesh + loaded nifti values (.xdmf))', type=str, required=True, 
                        default='./MRI_driven_parameters/meshes_with_nodal_values/')
    
    parser.add_argument('-ofn', '--outputfilename', help='Name basis for the output files (.xdmf))', type=str, required=True, 
                        default='_21GW.xdmf')
    
    args = parser.parse_args()

 
    # Read MRI nifti values onto mesh nodes (get n_nodes numpy array)
    nifti_path_T2 = args.seriesofniftis["T2"]
    nifti_path_segmentation = args.seriesofniftis["segmentation"]
    nifti_path_FA = args.seriesofniftis["diffusion_FA"] # diffusion
    
    # Input mesh format
    inputmesh_name = args.inputmesh.split('.')[0]
    inputmesh_format = args.inputmesh.split('.')[-1]
    
    # Input mesh
    if inputmesh_format == 'xml' or inputmesh_format == 'xdmf':
        
    	# Input mesh
    	############
        if inputmesh_format == 'xml':
            meshFEniCS = fenics.Mesh(args.inputmesh)
            
        elif inputmesh_format == 'xdmf':
            mesh = fenics.Mesh()
            with fenics.XDMFFile(args.inputmesh) as infile:
                infile.read(mesh)
                
        coordinates = mesh.coordinates()
        
        # Revert X<>Y coordinates inversion performed by Netgen:
        X = coordinates[:,0].copy()
        Y = coordinates[:,1].copy()
        coordinates[:,0] = Y
        coordinates[:,1] = X
    
    elif inputmesh_format == 'stl': # surface mesh
        from utils.converters import convert_meshformats_to_xml
        inputmesh_XML_path = inputmesh_name + '_SURFACE.xml'
        convert_meshformats_to_xml.stl_to_xml_2D(args.inputmesh, inputmesh_XML_path) # convert .stl into surface FEniCS readable .xml format
        mesh = fenics.Mesh(inputmesh_XML_path)
        coordinates = mesh.coordinates()
    
    # Load MRI nifti values onto mesh nodes (get n_nodes numpy array)
    regions_T2, functionFEM_T2 = transfer_nifti_values_to_FEnicSmesh_nodes(nifti_path_T2, "T2", coordinates)
    regions_segmentation, functionFEM_segmentation  = transfer_nifti_values_to_FEnicSmesh_nodes(nifti_path_segmentation, "Segmentation", coordinates) 
    regions_FA, functionFEM_FA = transfer_nifti_values_to_FEnicSmesh_nodes(nifti_path_FA, "FA", coordinates)
    
    # Store regions in XDMF
    #######################
    # T2
    # --
    xdmf = fenics.XDMFFile(mesh.mpi_comm(), args.output + "mesh_T2" + args.outputfilename)
    xdmf.write(functionFEM_T2)
    xdmf.close()
    
    xdmf = fenics.XDMFFile(mesh.mpi_comm(), args.output + "mesh_T2_regions" + args.outputfilename)
    xdmf.write(regions_T2)
    xdmf.close()
    
    # Segmentation
    # ------------
    xdmf = fenics.XDMFFile(mesh.mpi_comm(), args.output + "mesh_Seg" + args.outputfilename)
    xdmf.write(functionFEM_segmentation)
    xdmf.close()
    
    xdmf = fenics.XDMFFile(mesh.mpi_comm(), args.output + "mesh_Seg_regions" + args.outputfilename)
    xdmf.write(regions_segmentation)
    xdmf.close()
    
    # FA
    # --
    xdmf = fenics.XDMFFile(mesh.mpi_comm(), args.output + "mesh_FA" + args.outputfilename)
    xdmf.write(functionFEM_FA)
    xdmf.close()
    
    xdmf = fenics.XDMFFile(mesh.mpi_comm(), args.output + "mesh_FA_regions" + args.outputfilename)
    xdmf.write(regions_FA)
    xdmf.close()
    
    
        
    