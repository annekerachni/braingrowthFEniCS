import fenics
from PIL import Image
import glob
import matplotlib.pyplot as plt

# File-writing functions
########################

def export_XMLfile(output_folderpath, name, mesh):
    fenics.File( str(output_folderpath + name + '.xml') ) << mesh
    print('tetra_{}.xml was written'.format(name))
    return 

def export_PVDfile(output_folderpath, name, geometry_entity):
    fenics.File( str(output_folderpath + name + '.pvd') ).write(geometry_entity)
    print('{}.pvd was written'.format(name))
    return 

def export_XDMFfile(output_folderpath, value):
    # Set name and options for elastodynamics solution XDMF file export
    results_file_path = str(output_folderpath) + str(value) + ".xdmf"
    xdmf_file = fenics.XDMFFile(results_file_path)
    xdmf_file.parameters["flush_output"] = True
    xdmf_file.parameters["functions_share_mesh"] = True
    xdmf_file.parameters["rewrite_function_mesh"] = True
    return xdmf_file



