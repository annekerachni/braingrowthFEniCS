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
import fenics

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
    Args: 3D mesh in MSH format (Gmsh)
    Returns: mesh in XML format, explotable by FEniCS (FEniCS reads XML mesh files with tetrahedrons cells). 
    """
    mesh = meshio.read(input_file_msh) 
    #meshio.write(output_file_xml, meshio.Mesh(points=mesh.points, cells={'tetra': mesh.cells_dict['tetra']})) 
    meshio.write(output_file_xml, meshio.Mesh(points=mesh.points, cells={'tetra': mesh.cells_dict['tetra'], 'triangle': mesh.cells_dict['triangle']}))

    return 


def stl_to_xml_2D(input_file_stl, output_file_xml): 

    """
    Args: 2D mesh in STL format (stl)
    Returns: mesh in XML format, explotable by FEniCS (FEniCS reads XML mesh files with tetrahedrons cells). 
    """
    mesh = meshio.read(input_file_stl) 
    meshio.write(output_file_xml, meshio.Mesh(points = mesh.points, cells = {'triangle': mesh.cells_dict['triangle']})) 

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

  return n_tets, tets

@njit(parallel=True)
def get_face_indices(mesh, n_nodes, n_tets):
  '''
  Takes a list of arrays as an input and returns faces and number of faces. Faces are defined as 3 indexes of vertices from the coordinate list, only on the surface
  Args:
  mesh (list): mesh file as a list of np.arrays
  n_nodes (int): number of nodes
  Returns:
  faces (np.array): list of 3 vertices indexes
  n_faces(int): number of faces in the mesh
  '''
  n_faces = np.int64(mesh[n_nodes+n_tets+2][0])
  faces = np.zeros((n_faces,3), dtype=np.int64) # Index of three vertices of triangles
  for i in prange(n_faces):
    faces[i] = np.array([int(mesh[i+n_nodes+n_tets+3][1])-1,int(mesh[i+n_nodes+n_tets+3][2])-1,int(mesh[i+n_nodes+n_tets+3][3])-1])

  return faces, n_faces

def mesh_to_xml(output_file_xml, coordinates, tets): 
    #can be also use as 'mesh_to_vtk': in that case, use output_file_path in .vtk 
    meshio.write(output_file_xml, meshio.Mesh(points=coordinates, cells=[("tetra", tets)])) 

    return


import argparse
if __name__ == '__main__':
    """python3 -i convert_meshformats_to_xml.py -i './data/dhcp/dhcp_atlas/dhcp21GW_29296faces_107908tets.mesh' -o './data/dhcp/dhcp_atlas/dhcp21GW_29296faces_107908tets.xml' """
  
    parser = argparse.ArgumentParser(description='Convert Mesh to XML Fenics')
    
    parser.add_argument('-i', '--input', help='Input mesh (.vtk, .msh, .mesh) path', type=str, required=False, 
                        default='./data/dhcp/dhcp_atlas/21GW/dhcp21GW_29296faces_107908tets.mesh' ) 
    
    parser.add_argument('-o', '--output', help='Output mesh path (.xml)', type=str, required=False, 
                        default='./data/dhcp/dhcp_atlas/21GW/dhcp21GW_29296faces_107908tets.xml') 

    args = parser.parse_args()

    input_file_path = args.input
    inputmesh_format = input_file_path.split('.')[-1]

    output_file_path = args.output
    outputmesh_format = output_file_path.split('.')[-1]


    if inputmesh_format == 'vtk':
        vtk_to_xml(input_file_path, output_file_path) 


    elif inputmesh_format == 'msh': # Gmsh mesh
        msh_to_xml(input_file_path, output_file_path) # resp. msh_to_vtk

    
    elif inputmesh_format == 'mesh': # Netgen mesh
            mesh = load_mesh(input_file_path)   
            coordinates, n_nodes = get_nodes(mesh) 
            n_tets, tets = get_tetrahedrons(mesh, n_nodes)

            mesh_to_xml(output_file_path, coordinates, tets) # resp. msh_to_vtk
            
    elif inputmesh_format == 'stl': # surface mesh
        stl_to_xml_2D(input_file_path, output_file_path)

      
