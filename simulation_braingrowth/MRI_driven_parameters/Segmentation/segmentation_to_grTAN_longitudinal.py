import fenics
import argparse
import json
import meshio
from FEM_biomechanical_model import preprocessing
import matplotlib.pyplot as plt
import numpy as np

# Growth ponderation & associated Cortex/Core layers from Segmentation labels
#############################################################################
def bilayer_tangential_growth_ponderation_from_SegmentationLabels(grTAN, vertex2dofs_S, mesh_meshio_with_Seg):
    """
    Args:
    grTAN: fenics.Function(S)
    t_simu: 0.
    mesh_meshio_with_Seg: .vtk mesh file loaded with segmentation labels, collected from MRI data
    vertex2dofs_S: vertex --> scalar FEniCS DOF
    
    Returns: grTAN FEniCS scalar FEM function (1. in Cortex and 0. in Core)
    """
    
    # Defining gr from Cortex segmentation label
    seg = mesh_meshio_with_Seg.point_data["Segmentation"] # nodal array 
    
    for vertex, scalarDOF in enumerate(vertex2dofs_S):
        if seg[vertex] == 3. or seg[vertex] == 4. : # labels for Cortex from fetal dhcp atlas. See https://gin.g-node.org/kcl_cdb/fetal_brain_mri_atlas/src/master/ + script './utils/nifti_mesh/niftitomesh/generate_mesh_from_nifti/mask_nifti_with_segmentation_parcels.py'
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

    parser = argparse.ArgumentParser(description='braingrowthFEniCS: brain growth elastodynamics 3D model')
    
    parser.add_argument('-i', '--input', help='Input mesh path (xml)', type=str, required=False, 
                        default='./data/dhcp_mesh.xml') 
                        
    parser.add_argument('-n', '--normalization', help='Is normalization of the input mesh required? (required by braingrowthFEniCS)', type=bool, required=False, default=True)

    parser.add_argument('-seg', '--segmentationlabels', help='Path to the mesh loaded with Segmentation labels onto nodes (.vtk)', type=json.loads, required=False, 
    default={"21GW": './niftitomesh/transfer_niftivalues_to_meshnodes/results/21GW/MRIniivalues_NearestNeighbor_21GW_homogeneizeCortexLabels.vtk'})
    
    args = parser.parse_args() 
    
    # Input mesh
    ############

    # Mesh
    # ----
    # mesh & boundary mesh
    print("\nimporting mesh...")
    inputmesh_path = args.input
    mesh = fenics.Mesh(inputmesh_path)
    bmesh = fenics.BoundaryMesh(mesh, "exterior") # bmesh at t=0.0 (cortex envelop)

    if args.visualization == True:
        fenics.plot(mesh) 
        plt.title("input mesh")
        plt.show() 

    # mesh characteristics
    characteristics0 = preprocessing.compute_geometrical_characteristics(mesh, bmesh)
    center_of_gravity0 = preprocessing.compute_center_of_gravity(characteristics0) 
    min_mesh_spacing0, average_mesh_spacing0, max_mesh_spacing0 = preprocessing.compute_mesh_spacing(mesh)
    print('input mesh characteristics: {}'.format(characteristics0))
    print('input mesh COG = [xG0:{}, yG0:{}, zG0:{}]'.format(center_of_gravity0[0], center_of_gravity0[1], center_of_gravity0[2]))
    print("input mesh min mesh spacing: {:.3f} mm".format(min_mesh_spacing0))
    print("input mesh mean mesh spacing: {:.3f} mm".format(average_mesh_spacing0))
    print("input mesh max mesh spacing: {:.3f} mm".format(max_mesh_spacing0))

    # normalization and boundary mesh update, normalized mesh characteristics
    if args.normalization == True:
        print("\nnormalizing mesh...")
        mesh = preprocessing.normalize_mesh(mesh, characteristics0, center_of_gravity0)
        bmesh = fenics.BoundaryMesh(mesh, "exterior") # update bmesh

        if args.visualization == True:
            fenics.plot(mesh) 
            plt.title("normalized mesh")
            plt.show()  
            #vedo.dolfin.plot(mesh, wireframe=False, text='normalized mesh', style='paraview', axes=4).close()
            
        characteristics = preprocessing.compute_geometrical_characteristics(mesh, bmesh) # n_nodes, coordinates, n_tets, n_faces_Surface, n_faces_Volume, minx, maxx, miny, maxy, minz, maxz 
        center_of_gravity = preprocessing.compute_center_of_gravity(characteristics) 
        min_mesh_spacing, average_mesh_spacing, max_mesh_spacing = preprocessing.compute_mesh_spacing(mesh)
        print('normalized mesh characteristics: {}'.format(characteristics))
        print('normalized mesh COG = [xG:{}, yG:{}, zG:{}]'.format(center_of_gravity[0], center_of_gravity[1], center_of_gravity[2]))
        print("normalized min_mesh_spacing: {:.3f} mm".format(min_mesh_spacing))
        print("normalized mesh mean mesh spacing: {:.3f} mm".format(average_mesh_spacing))
        print("normalized mesh max mesh spacing: {:.3f} mm".format(max_mesh_spacing))
        
        
    # Scalar Function Space
    #######################
    S = fenics.FunctionSpace(mesh, "CG", 1) 
    
    # mapping vertex --> scalar DOF
    from FEM_biomechanical_model import mappings
    vertex2dofs_S = mappings.vertex_to_dof_ScalarFunctionSpace(S)
    
    # FEM functions
    grTAN = fenics.Function(S, name="grTAN")  
    
    t_simu, tGW = 0., "21GW" # simulation time and corresponding real time
    outputfolderpath = './simulation_braingrowth/results/'
    
    # Get Segmentation labels from .vtk
    path_inputmeshloaded_with_SegLabels_VTK = args.segmentationlabels 
    
    # Build grTAN (growth ponderation function, defining two layers. Allocating the followinf lables: Cortex-->1., Core-->0.)
    grTAN = bilayer_tangential_growth_ponderation_from_SegmentationLabels(grTAN, vertex2dofs_S, tGW, path_inputmeshloaded_with_SegLabels_VTK)
    
    # Export tangential growth ponderation 
    grTAN_file_XDMF = fenics.XDMFFile(args.output + "grTAN_{}.xdmf".format(t_simu))
    grTAN_file_XDMF.parameters["flush_output"] = True
    grTAN_file_XDMF.parameters["functions_share_mesh"] = True
    grTAN_file_XDMF.write(grTAN, t_simu)