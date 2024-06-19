import fenics
from PIL import Image
import glob
import matplotlib.pyplot as plt
import os

# File-writing functions
########################

def export_XMLfile(output_folderpath, name, mesh):
    path = os.path.join(output_folderpath, name, '.xml')
    fenics.File(path) << mesh
    print('tetra_{}.xml was written'.format(name))
    return 

def export_PVDfile(output_folderpath, name, geometry_entity):
    path = os.path.join(output_folderpath, name + '.pvd')
    fenics.File(path).write(geometry_entity)
    print('{}.pvd was written'.format(name))
    return 

def export_XDMFfile(output_folderpath, value):
    # Set name and options for elastodynamics solution XDMF file export
    results_file_path = os.path.join(output_folderpath, str(value), ".xdmf")
    xdmf_file = fenics.XDMFFile(results_file_path)
    xdmf_file.parameters["flush_output"] = True
    xdmf_file.parameters["functions_share_mesh"] = True
    xdmf_file.parameters["rewrite_function_mesh"] = True
    return xdmf_file



