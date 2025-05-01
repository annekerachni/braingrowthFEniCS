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