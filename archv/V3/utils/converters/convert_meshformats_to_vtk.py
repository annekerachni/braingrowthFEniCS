# code source: 
# original BrainGrowth: https://github.com/rousseau/BrainGrowth/blob/master/geometry.py
# https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-to-mesh-xdmf-from-dolfin-convert-to-meshio/412/86

import meshio
from numba.typed import List
import numpy as np
from numba import njit, prange
import fenics

def xml_to_vtk(input_file_xml, output_file_vtk): 
    """
    From FEniCS mesh formats, get .vtk format
    """

    mesh = fenics.Mesh(input_file_xml) # FEnicS object

    # n_nodes = mesh.num_vertices()
    coordinates = mesh.coordinates()

    # n_tets = mesh.num_cells() 
    tets = mesh.cells()

    bmesh = fenics.BoundaryMesh(mesh, "exterior")
    # n_faces = bmesh.num_faces() 
    faces = bmesh.cells()
    

    #meshio.write(output_file_xml, meshio.Mesh(points=coordinates, cells={'tetra': tets})) 
    vtk_mesh = meshio.Mesh(points=coordinates, cells={'tetra': tets, 'triangle': faces})
    meshio.write(output_file_vtk, vtk_mesh)

    return 




def msh_to_vtk(input_file_msh, output_file_vtk): 
    """
    Args: 3D mesh in MSH format (Gmsh)
    Returns: mesh in XML format, explotable by FEniCS (FEniCS reads XML mesh files with tetrahedrons cells). 
    """
    mesh = meshio.read(input_file_msh) 
    meshio.write(output_file_vtk, meshio.Mesh(points = mesh.points, cells = {'tetra': mesh.cells_dict['tetra']})) 

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


def msh_to_vtk(input_file_path, output_file_vtk): 
    #can be also use as 'mesh_to_vtk': in that case, use output_file_path in .vtk 
    
    # .msh to .stl
    msh = meshio.read(input_file_path)
    for cell in msh.cells:
        if cell.type == "tetra":
            tetra_cells = cell.data
    triangle_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells})
    meshio.write(output_file_vtk, triangle_mesh)

    return 


def mesh_to_vtk(output_file_stl, faces_coords, tets): 
    #can be also use as 'mesh_to_vtk': in that case, use output_file_path in .vtk 

    meshio.write(output_file_stl, meshio.Mesh(points=faces_coords, cells=[("tetra", tets)])) 

    return 


import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert Mesh to XML Fenics')

    parser.add_argument('-i', '--input', help='Input mesh (.msh, .mesh, .xml) path', type=str, required=False, 
                        default='./data/dhcp21_145K_refinedcoef20.xml' )
    
    parser.add_argument('-o', '--output', help='Output mesh path (.vtk)', type=str, required=False, 
                        default='./data/dhcp21GW.vtk') 

    args = parser.parse_args()

    input_file_path = args.input
    inputmesh_format = input_file_path.split('.')[-1]

    output_file_path = args.output
    outputmesh_format = output_file_path.split('.')[-1]


    if inputmesh_format == 'msh': # Gmsh mesh
        msh_to_vtk(input_file_path, output_file_path)

    
    elif inputmesh_format == 'mesh': # Netgen mesh
        if outputmesh_format == 'vtk': # Netgen mesh
            mesh = load_mesh(input_file_path)   
            coordinates, n_nodes = get_nodes(mesh) 
            n_tets, tets = get_tetrahedrons(mesh, n_nodes)
            faces, n_faces = get_face_indices(mesh, n_nodes, n_tets)

            coordinates_faces = []
            for face in faces: # [node1_idx, node2_idx, node3_idx]
                node1_idx, node2_idx, node3_idx = face[0], face[1], face[2]
                coordinates_faces.append( coordinates[node1_idx, :] )
                coordinates_faces.append( coordinates[node2_idx, :] )
                coordinates_faces.append( coordinates[node3_idx, :] )
            coordinates_faces = np.array(coordinates_faces)

            mesh_to_vtk(output_file_path, coordinates_faces, tets) 


    elif inputmesh_format == 'xml': # FEniCS mesh
        if outputmesh_format == 'vtk':
            xml_to_vtk(input_file_path, output_file_path)
