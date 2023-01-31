# code source: https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-to-mesh-xdmf-from-dolfin-convert-to-meshio/412/86
# meshio 4.0.8

import meshio
import fenics
import os
import numpy as np

def mesh3D_to_mesh2D(mesh, cell_type, prune_z=False): 
    """convert 3D gmsh mesh into 2D gmsh mesh"""
    # Code source: https://fenicsproject.discourse.group/t/how-to-use-meshvaluecollection/5106
    cells = np.vstack([cell.data for cell in mesh.cells if cell.type==cell_type])
    cell_data = np.hstack([mesh.cell_data_dict["gmsh:geometrical"][key]
                        for key in mesh.cell_data_dict["gmsh:geometrical"].keys() if key==cell_type])
    # Remove z-coordinates from mesh if we have a 2D cell and all points have the same third coordinate
    points= mesh.points
    if prune_z:
        points = points[:,:2] # remove z
    mesh_new = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})

    return mesh_new

def convert_msh_to_xml(MSHmesh_path): 

    msh = meshio.read(MSHmesh_path) # e.g. "./data/sphere/sphere.msh"
    A = str(os.path.splitext(MSHmesh_path)[0]) 
    B = os.path.splitext(A)[0] # "e.g. ./data/sphere/sphere"
    XMLmesh_output_path = str(B + ".xml")
    print('xml path = {}'.format(XMLmesh_output_path))
    meshio.write(XMLmesh_output_path, meshio.Mesh(points = msh.points, cells = {'triangle': msh.cells_dict['triangle']})) # 'triangle' for 2D; 'tetra' for 3D

    return XMLmesh_output_path


def convert_msh_to_xdmf(MSHmesh_path):

    msh = meshio.read(MSHmesh_path)
    A = str(os.path.splitext(MSHmesh_path)[0]) 
    B = os.path.splitext(A)[0] # "e.g. ./data/sphere/sphere"
    XDMFmesh_output_path = str(B + ".xdmf")
    meshio.write(XDMFmesh_output_path, meshio.Mesh(points = msh.points, cells = {'triangle': msh.cells_dict['triangle']}))

    return XDMFmesh_output_path


def convert_xdmf_to_pvd(XDMFmesh_path):

    mesh = fenics.Mesh()
    with fenics.XDMFFile(XDMFmesh_path) as infile: # e.g. sphere_mesh_xdmffile_path = "./data/sphere/sphere.xdmf"
        infile.read(mesh)
    mvc = fenics.MeshValueCollection("size_t", mesh, 2) 
    A = str(os.path.splitext(XDMFmesh_path)[0]) 
    B = os.path.splitext(A)[0] # "e.g. ./data/sphere/sphere"
    PVDmesh_output_path = str( B + ".pvd")
    fenics.File(PVDmesh_output_path).write(mesh)

    return PVDmesh_output_path
