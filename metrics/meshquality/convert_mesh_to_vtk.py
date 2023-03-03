import meshio
from numba import njit, prange
from numba.typed import List
import numpy as np

# code source: https://github.com/rousseau/BrainGrowth

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

def mesh_to_vtk(output_file_vtk, coordinates, tets): 
    meshio.write(output_file_vtk, meshio.Mesh(points = coordinates, cells = [("tetra", tets)])) 

    return


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert .mesh to .vtk')
    parser.add_argument('-i', '--input', help='Input mesh (.mesh) path ', type=str, required=False, default='./data/ellipsoid.mesh')

    args = parser.parse_args()

    input_mesh_path = args.input
    output_mesh_path_VTK = str(input_mesh_path.split('.')[0] + '.vtk')

    mesh = load_mesh(input_mesh_path)   
    coordinates, n_nodes = get_nodes(mesh) 
    tets = get_tetrahedrons(mesh, n_nodes)
    mesh_to_vtk(output_mesh_path_VTK, coordinates, tets) # convert netgen mesh into .vtk and write it