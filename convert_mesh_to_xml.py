"""
python convert_mesh_to_xml.py -i '/home/latim/FEniCS/GitHub/data_test/dhcpRAS_iso_fine.mesh' -o '/home/latim/FEniCS/GitHub/data_test/dhcpRAS_iso_fine.xml'
"""

# code source: 
# original BrainGrowth: https://github.com/rousseau/BrainGrowth/blob/master/geometry.py
# https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-to-mesh-xdmf-from-dolfin-convert-to-meshio/412/86

import meshio
from numba.typed import List
import numpy as np
from numba import njit, prange

def vtk_to_xml(input_file_vtk, output_file_xml): 
    """
    Args: mesh in VTK format
    Returns: mesh in XML format, explotable by FEniCS (FEniCS reads XML mesh files with tetrahedrons cells). 
    # https://fenicsproject.discourse.group/t/pygmsh-tutorial/2506/3
    """
    # Read input mesh (.vtk) 
    mesh = meshio.read(input_file_vtk) 
    
    for cell in mesh.cells:
        if cell.type == "tetra": 
            # write output mesh (.xml)
            tetra_mesh = meshio.Mesh(points=mesh.points, cells={"tetra": cell.data})
            meshio.write(output_file_xml, tetra_mesh)
            print('output file (with tetra cells) well written down in XML format')

    return 

def msh_to_xml(input_file_msh, output_file_xml): 
    """
    Args: mesh in MSH format (Gmsh)
    Returns: mesh in XML format, explotable by FEniCS (FEniCS reads XML mesh files with tetrahedrons cells). 
    """
    mesh = meshio.read(input_file_msh) 
    meshio.write(output_file_xml, meshio.Mesh(points = mesh.points, cells = {'tetra': mesh.cells_dict['tetra']})) 

    return 

def load_mesh(input_file_mesh):
    """
    Args: mesh in MESH format (Netgen)
    Returns: mesh List object
    """
    mesh = [] # Import mesh as a list
    with open(input_file_mesh) as inputfile:
        for line in inputfile:
            mesh.append(line.strip().split(' '))
        for i in range(len(mesh)):
            mesh[i] = list(filter(None, mesh[i]))
            mesh[i] = np.array([float(a) for a in mesh[i]])

    mesh = List(mesh) # added to avoid "reflected list" for "mesh" argument issue with numba

    return mesh

@njit(parallel=True)
def get_nodes(mesh):
  """
  Extract coordinates and number of nodes from mesh
  Args: mesh (list): mesh file as a list of np.arrays
  Returns:
  - coordinates (np.array): list of 3 cartesian points
  - n_nodes (int): number of nodes
  """
  n_nodes = np.int64(mesh[0][0])
  coordinates = np.zeros((n_nodes,3), dtype=np.float64) # Undeformed coordinates of nodes
  for i in prange(n_nodes):
    coordinates[i] = np.array([float(mesh[i+1][1]),float(mesh[i+1][0]),float(mesh[i+1][2])]) # Change x, y (Netgen)
  
  return coordinates, n_nodes 

@njit(parallel=True)
def get_tetrahedrons(mesh, n_nodes):
  """
  Takes a list of arrays as an input and returns tets and number of tets. Tets are defined as 4 indexes of vertices from the coordinate list
  Args:
  - mesh (list): mesh file as a list of np.arrays
  - n_nodes (int): number of nodes
  Returns:
  - tets (np.array): list of 4 vertices indexes
  - n_tets(int): number of tets in the mesh
  """
  n_tets = np.int64(mesh[n_nodes+1][0])
  tets = np.zeros((n_tets,4), dtype=np.int64) # Index of four vertices of tetrahedra
  for i in prange(n_tets):
    tets[i] = np.array([int(mesh[i+n_nodes+2][1])-1,int(mesh[i+n_nodes+2][2])-1,int(mesh[i+n_nodes+2][4])-1,int(mesh[i+n_nodes+2][3])-1])  # Note the switch of handedness (1,2,3,4 -> 1,2,4,3) - the code uses right handed tets

  return tets

def mesh_to_xml(output_file_xml, coordinates, tets): 
    meshio.write(output_file_xml, meshio.Mesh(points = coordinates, cells = [("tetra", tets)])) 

    return

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Mesh to XML Fenics')
    parser.add_argument('-i', '--input', help='Input mesh path (vtk, msh, mesh) ', type=str, required=True)
    parser.add_argument('-o', '--output', help='Output XML mesh path', type=str, required=True)

    args = parser.parse_args()

    input_file = args.input
    output_file= args.output
    suffix = input_file.split('.')[-1]

    if suffix == 'vtk':
        vtk_to_xml(input_file, output_file) 

    elif suffix == 'msh': # Gmsh mesh
        msh_to_xml(input_file, output_file)
    
    elif suffix == 'mesh': # Netgen mesh
        mesh = load_mesh(input_file)   
        coordinates, n_nodes = get_nodes(mesh) 
        tets = get_tetrahedrons(mesh, n_nodes)
        mesh_to_xml(output_file, coordinates, tets)




