# code source: 
# original BrainGrowth: https://github.com/rousseau/BrainGrowth/blob/master/geometry.py
# https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-to-mesh-xdmf-from-dolfin-convert-to-meshio/412/86

# meshio 4.0.8

import meshio
import fenics
import os
from numba.typed import List
import numpy as np
from numba import njit, prange

def load_MESHmesh(MESHmesh_path):

    # Import mesh as a list
    mesh = []
    with open(MESHmesh_path) as inputfile:
        for line in inputfile:
            mesh.append(line.strip().split(' '))
        for i in range(len(mesh)):
            mesh[i] = list(filter(None, mesh[i]))
            mesh[i] = np.array([float(a) for a in mesh[i]])

    mesh = List(mesh) # added to avoid "reflected list" for "mesh" argument issue with numba

    return mesh

@njit(parallel=True)
def get_nodes(mesh):
  '''
  Extract coordinates and number of nodes from mesh
  Args:
  mesh (list): mesh file as a list of np.arrays
  Returns:
  coordinates (np.array): list of 3 cartesian points
  n_nodes (int): number of nodes
  '''
  n_nodes = np.int64(mesh[0][0])
  coordinates = np.zeros((n_nodes,3), dtype=np.float64) # Undeformed coordinates of nodes
  for i in prange(n_nodes):
    coordinates[i] = np.array([float(mesh[i+1][1]),float(mesh[i+1][0]),float(mesh[i+1][2])]) # Change x, y (Netgen)
  
  return coordinates, n_nodes

@njit(parallel=True)
def get_tetrahedrons(mesh, n_nodes):
  '''
  Takes a list of arrays as an input and returns tets and number of tets. Tets are defined as 4 indexes of vertices from the coordinate list
  Args:
  mesh (list): mesh file as a list of np.arrays
  n_nodes (int): number of nodes
  Returns:
  tets (np.array): list of 4 vertices indexes
  n_tets(int): number of tets in the mesh
  '''
  n_tets = np.int64(mesh[n_nodes+1][0])
  tets = np.zeros((n_tets,4), dtype=np.int64) # Index of four vertices of tetrahedra
  for i in prange(n_tets):
    tets[i] = np.array([int(mesh[i+n_nodes+2][1])-1,int(mesh[i+n_nodes+2][2])-1,int(mesh[i+n_nodes+2][4])-1,int(mesh[i+n_nodes+2][3])-1])  # Note the switch of handedness (1,2,3,4 -> 1,2,4,3) - the code uses right handed tets
  
  return tets, n_tets

def convert_mesh_to_xml_definingpath(MESHmeshpath, coordinates, tets): 
    # Define XML output path
    A = str(os.path.splitext(MESHmeshpath)[0]) 
    B = os.path.splitext(A)[0] # "e.g. ./data/ellipsoid/ellipsoid"
    XMLoutputpath = str(B + ".xml")
    print('xml path = {}'.format(XMLoutputpath))
    meshio.write(XMLoutputpath, meshio.Mesh(points = coordinates, cells = [("tetra", tets)]))  

    return XMLoutputpath

def convert_mesh_to_xml(XMLoutputpath, coordinates, tets): 
    meshio.write(XMLoutputpath, meshio.Mesh(points = coordinates, cells = [("tetra", tets)])) 
    return

def convert_msh_to_xml(MSHmesh_path): 

    msh = meshio.read(MSHmesh_path) # e.g. ./data/sphere/sphere.msh"
    A = str(os.path.splitext(MSHmesh_path)[0]) 
    B = os.path.splitext(A)[0] # "e.g. ./data/sphere/sphere"
    XMLmesh_output_path = str(B + ".xml")
    print('xml path = {}'.format(XMLmesh_output_path))
    meshio.write(XMLmesh_output_path, meshio.Mesh(points = msh.points, cells = {'tetra': msh.cells_dict['tetra']})) 

    return XMLmesh_output_path


def convert_msh_to_xdmf(MSHmesh_path):

    msh = meshio.read(MSHmesh_path)
    A = str(os.path.splitext(MSHmesh_path)[0]) 
    B = os.path.splitext(A)[0] # "e.g. ./data/sphere/sphere"
    XDMFmesh_output_path = str(B + ".xdmf")
    meshio.write(XDMFmesh_output_path, meshio.Mesh(points = msh.points, cells = {'tetra': msh.cells_dict['tetra']}))

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

def vtk_to_xml(vtk_tetra_path, output_geometry): 
    """
    FEniCS reads XML mesh files with tetrahedrons cells. 

    Input: mesh in VTK format, generated from BrainGrowth input mesh
    Output: mesh in XML format, explotable by FEniCS
    """
    # Read BrainGrowth input mesh
    msh = meshio.read(vtk_tetra_path) 

    # https://fenicsproject.discourse.group/t/pygmsh-tutorial/2506/3
    for cell in msh.cells:

        if cell.type == "triangle":
            triangle_cells = cell.data
            triangle_mesh =meshio.Mesh(points=msh.points, cells=[("triangle", triangle_cells)])
            meshio.write(output_geometry + "_trianglecells.xml", triangle_mesh)
            vtk_file_path = output_geometry + "_trianglecells.xml"
            print('XML file with triangle cells, readable in FEniCS, well written down')

        elif  cell.type == "tetra":
            tetra_cells = cell.data
            tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells})
            meshio.write(output_geometry + "_tetracells.xml", tetra_mesh)
            xml_file_path = output_geometry + "_tetracells.xml"
            print('XML file with tetra cells, readable in FEniCS, well written down')
    
    return xml_file_path
