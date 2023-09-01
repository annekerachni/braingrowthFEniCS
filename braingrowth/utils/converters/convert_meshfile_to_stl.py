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
    meshio.write(output_file_vtk, meshio.Mesh(points=coordinates, cells={'tetra': tets, 'triangle': faces}))

    return 


def xml_to_stl(input_file_xml, output_file_vtk): 
    """
    From FEniCS mesh formats, get .stl format readable in meshlab (Then, smooth, apply filter with the software)
    """

    mesh = fenics.Mesh(input_file_xml) # FEnicS object
    bmesh = fenics.BoundaryMesh(mesh, "exterior")

    # n_faces = bmesh.num_faces() 
    faces = bmesh.cells()
    bmesh_coordinates = bmesh.coordinates()

    #meshio.write(output_file_xml, meshio.Mesh(points=coordinates, cells={'tetra': tets})) 
    meshio.write(output_file_vtk, meshio.Mesh(points=bmesh_coordinates, cells={'triangle': faces}))

    return


def msh_to_stl(input_file_msh, output_file_xml): 
    """
    Args: 3D mesh in MSH format (Gmsh)
    Returns: mesh in XML format, explotable by FEniCS (FEniCS reads XML mesh files with tetrahedrons cells). 
    """
    mesh = meshio.read(input_file_msh) 
    

    meshio.write(output_file_xml, meshio.Mesh(points = mesh.points, cells = {'tetra': mesh.cells_dict['tetra']})) 

    return 


def vtk_to_stl(input_file_vtk, output_file_stl): 
    """
    Args: mesh in VTK format
    Returns: mesh in XML format, explotable by FEniCS (FEniCS reads XML mesh files with tetrahedrons cells). 
    # https://fenicsproject.discourse.group/t/pygmsh-tutorial/2506/3
    """
    
    mesh = meshio.read(input_file_vtk) # Read input mesh (.vtk) 
    meshio.write(output_file_stl, meshio.Mesh(points = mesh.points, cells = {'tetra': mesh.cells_dict['tetra']})) 

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


def msh_to_stl(input_file_path, output_file_stl): 
    #can be also use as 'mesh_to_vtk': in that case, use output_file_path in .vtk 
    
    # .msh to .stl
    msh = meshio.read(input_file_path)
    for cell in msh.cells:
        if cell.type == "triangle":
            triangle_cells = cell.data
    triangle_mesh = meshio.Mesh(points=msh.points, cells={"triangle": triangle_cells})
    meshio.write(output_file_stl, triangle_mesh)

    """ 
    mesh = fenics.Mesh(input_file_path) # FEniCS object
    tets = mesh.cells()
    n_faces_Volume = mesh.num_facets()
    tets_coords = mesh.coordinates()

    bmesh = fenics.BoundaryMesh(mesh, "exterior") # FEniCS object
    n_faces_Surface = bmesh.num_faces()
    n_nodes_Surface = bmesh.num_vertices()
    faces = bmesh.cells()
    faces_coords = bmesh.coordinates()

    meshio.write(output_file_stl, meshio.Mesh(points=faces_coords, cells=[("triangle", faces)]))  
    """

    return 


def mesh_to_stl(output_file_stl, faces_coords, faces): 
    #can be also use as 'mesh_to_vtk': in that case, use output_file_path in .vtk 

    meshio.write(output_file_stl, meshio.Mesh(points=faces_coords, cells=[("triangle", faces)])) 

    return 


import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert mesh to .stl')

    parser.add_argument('-i', '--input', help='Input mesh (.msh, .vtk, .mesh) path', type=str, required=False, 
                        default='./data/ellipsoid/sphere5.mesh' ) # './data/dhcp/mesh/dhcpWholefull/dhcp21Whole_veryfine_3M.mesh'; './data/ellipsoid_284K.mesh'
    
    parser.add_argument('-o', '--output', help='Output mesh path (.stl,)', type=str, required=False, 
                        default='./data/ellipsoid/sphere5.stl') # './data/dhcp/mesh/dhcpWholefull/dhcp21Whole_veryfine_3M.xml'; './data/ellipsoid_284K.xml'

    args = parser.parse_args()

    input_file_path = args.input
    inputmesh_format = input_file_path.split('.')[-1]

    output_file_path = args.output
    outputmesh_format = output_file_path.split('.')[-1]


    if inputmesh_format == 'msh': # Gmsh mesh
        msh_to_stl(input_file_path, output_file_path)

    
    if inputmesh_format == 'vtk':
        vtk_to_stl(input_file_path, output_file_path) 

    
    elif inputmesh_format == 'mesh': # Netgen mesh
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

        mesh_to_stl(output_file_path, coordinates_faces, faces) 

