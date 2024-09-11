import fenics
import argparse
import json
import matplotlib.pyplot as plt
import os, sys
import meshio

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(sys.path[0])))) 

from FEM_biomechanical_model import preprocessing

# Growth ponderation & associated Cortex/Core layers from Segmentation labels
#############################################################################
def bilayer_tangential_growth_ponderation_from_SegmentationLabels(grTAN, vertex2dofs_S, mesh_loaded_with_Seglabels):
    """
    Args:
    grTAN: fenics.Function(S)
    t_simu: 0.
    mesh_meshio_with_Seg: .vtk mesh file loaded with segmentation labels, collected from MRI data
    vertex2dofs_S: vertex --> scalar FEniCS DOF
    
    Returns: grTAN FEniCS scalar FEM function (1. in Cortex and 0. in Core)
    """
    
    # Defining gr from Cortex segmentation label
    #mesh_loaded_with_Seglabels = meshio.read(path_inputmeshloaded_with_SegLabels_VTK)
    seg = mesh_loaded_with_Seglabels.point_data["Segmentation"] # nodal array 
    
    for vertex, scalarDOF in enumerate(vertex2dofs_S):
        if seg[vertex] == 4. : # labels for Cortex from fetal dhcp atlas. See https://gin.g-node.org/kcl_cdb/fetal_brain_mri_atlas/src/master/ + script './utils/nifti_mesh/niftitomesh/generate_mesh_from_nifti/mask_nifti_with_segmentation_parcels.py'
            grTAN.vector()[scalarDOF] = 1.
    
    """
    # if mesh_loaded_with_Seg is a .xdmf file
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mesh_loaded_with_Seg = fenics.XDMFFile(comm, path_inputmeshloaded_with_SegLabels_XDMF["21GW"])
    
    seg_label = fenics.Function(S)     
    mesh_loaded_with_Seg.read_checkpoint(seg_label, "Segmentation", 0) 
    
    mesh_loaded_with_Seg.read(mesh, True)
    #fenics.plot(mesh)
    """
    
    return grTAN # FEniCS scalar FEM function

def bilayer_tangential_growth_ponderation_from_SegmentationLabels_WHOLEMESH(grTAN, vertex2dofs_S, mesh_loaded_with_Seglabels):
    """
    Args:
    grTAN: fenics.Function(S)
    t_simu: 0.
    mesh_meshio_with_Seg: .vtk mesh file loaded with segmentation labels, collected from MRI data
    vertex2dofs_S: vertex --> scalar FEniCS DOF
    
    Returns: grTAN FEniCS scalar FEM function (1. in Cortex and 0. in Core)
    """
    
    # Defining gr from Cortex segmentation label
    #mesh_loaded_with_Seglabels = meshio.read(path_inputmeshloaded_with_SegLabels_VTK)
    seg = mesh_loaded_with_Seglabels.point_data["Segmentation"] # nodal array 
    
    for vertex, scalarDOF in enumerate(vertex2dofs_S):
        if seg[vertex] == 4. or seg[vertex] == 3.: # labels for Cortex from fetal dhcp atlas. See https://gin.g-node.org/kcl_cdb/fetal_brain_mri_atlas/src/master/ + script './utils/nifti_mesh/niftitomesh/generate_mesh_from_nifti/mask_nifti_with_segmentation_parcels.py'
            grTAN.vector()[scalarDOF] = 1.
    
    """
    # if mesh_loaded_with_Seg is a .xdmf file
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mesh_loaded_with_Seg = fenics.XDMFFile(comm, path_inputmeshloaded_with_SegLabels_XDMF["21GW"])
    
    seg_label = fenics.Function(S)     
    mesh_loaded_with_Seg.read_checkpoint(seg_label, "Segmentation", 0) 
    
    mesh_loaded_with_Seg.read(mesh, True)
    #fenics.plot(mesh)
    """
    
    return grTAN # FEniCS scalar FEM function



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Get growth zones ponderation (FEniCS-readable format) from MRI Segmentation data (vtk)')
    
    parser.add_argument('-i', '--input', help='Input mesh path (.xml, .xdmf)', type=str, required=False, 
                        default='./data/dHCP_raw/dhcpRight21GW_masked_20000faces_98000tets_refinedWidthCoef5.xdmf') 
                        
    parser.add_argument('-seg', '--segmentationlabels', help='Path to the mesh loaded with Segmentation labels onto nodes (.vtk)', type=json.loads, required=False, 
    default={21: './MRI_driven_parameters/meshes_loaded_with_MRI_data/dhcpRight21GW_masked_20000faces_98000tets_refinedWidthCoef5/dhcpRight21GW_masked_20000faces_98000tets_refinedWidthCoef5.vtk'})
    
    parser.add_argument('-o', '--output', help='Path to output folder', type=str, required=False, 
                        default='./data/dHCP_raw/dhcpRight21GW_masked_20000faces_98000tets_refinedWidthCoef5') 
    
    args = parser.parse_args() 
    
    # Input mesh
    ############

    # Mesh
    # ----
    # mesh & boundary mesh
    print("\nimporting mesh...")
    inputmesh_path = args.input
    inputmesh_format = inputmesh_path.split('.')[-1]

    if inputmesh_format == "xml":
        mesh = fenics.Mesh(inputmesh_path)

    elif inputmesh_format == "xdmf":
        mesh = fenics.Mesh()
        with fenics.XDMFFile(inputmesh_path) as infile:
            infile.read(mesh)
            
    bmesh = fenics.BoundaryMesh(mesh, "exterior") # bmesh at t=0.0 (cortex envelop)

    # mesh characteristics
    characteristics0 = preprocessing.compute_geometrical_characteristics(mesh, bmesh)
    center_of_gravity0 = preprocessing.compute_center_of_gravity(characteristics0) 
    min_mesh_spacing0, average_mesh_spacing0, max_mesh_spacing0 = preprocessing.compute_mesh_spacing(mesh)
    print('input mesh characteristics: {}'.format(characteristics0))
    print('input mesh COG = [xG0:{}, yG0:{}, zG0:{}]'.format(center_of_gravity0[0], center_of_gravity0[1], center_of_gravity0[2]))
    print("input mesh min mesh spacing: {:.3f} mm".format(min_mesh_spacing0))
    print("input mesh mean mesh spacing: {:.3f} mm".format(average_mesh_spacing0))
    print("input mesh max mesh spacing: {:.3f} mm".format(max_mesh_spacing0))
        
    # Scalar Function Space
    #######################
    S = fenics.FunctionSpace(mesh, "CG", 1) 
    
    # mapping vertex --> scalar DOF
    from FEM_biomechanical_model import mappings
    vertex2dofs_S = mappings.vertex_to_dof_ScalarFunctionSpace(S)
    
    # FEM functions
    grTAN = fenics.Function(S, name="grTAN")  
    
    # Get Segmentation labels from .vtk
    dict_SegLabels_VTK = args.segmentationlabels
    for i in dict_SegLabels_VTK:
        tGW = i
        path_inputmeshloaded_with_SegLabels_VTK = dict_SegLabels_VTK[i]
    
    # Build grTAN (growth ponderation function, defining two layers. Allocating the followinf lables: Cortex-->1., Core-->0.)
    grTAN = bilayer_tangential_growth_ponderation_from_SegmentationLabels(grTAN, vertex2dofs_S, path_inputmeshloaded_with_SegLabels_VTK)
    
    # Export tangential growth ponderation 
    grTAN_file_XDMF = fenics.XDMFFile(args.output + "_loaded_with_grTAN_for_H0_delineation_at{}GW.xdmf".format(tGW))
    grTAN_file_XDMF.parameters["flush_output"] = True
    grTAN_file_XDMF.parameters["functions_share_mesh"] = True
    grTAN_file_XDMF.write(grTAN, tGW)