import meshio
from numba.typed import List
import numpy as np
from numba import njit, prange
import fenics
import argparse


# BrainGrowth fonction (to read .mesh)
######################################
# original BrainGrowth: https://github.com/rousseau/BrainGrowth/blob/master/geometry.py

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

##
# https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-to-mesh-xdmf-from-dolfin-convert-to-meshio/412/86

# Convert to .xdmf
##################
def xml_to_xdmf(input_file_xml, output_file_xdmf):
    mesh = fenics.Mesh(input_file_xml)
    meshio.write(output_file_xdmf, meshio.Mesh(points=mesh.coordinates(), cells={'tetra': mesh.cells()})) 

    return

# Convert to .xml
#################
def xdmf_to_xml(input_file_xdmf, output_file_xml):
    mesh = fenics.Mesh()
    with fenics.XDMFFile(input_file_xdmf) as infile:
        infile.read(mesh)
            
    meshio.write(output_file_xml, meshio.Mesh(points=mesh.coordinates(), cells={'tetra': mesh.cells()})) 

    return

def vtk_to_xml_xdmf(input_file_vtk, output_file_xml_xdmf): 
    """
    Args: mesh in VTK format
    Returns: mesh in XML format, explotable by FEniCS (FEniCS reads XML mesh files with tetrahedrons cells). 
    # https://fenicsproject.discourse.group/t/pygmsh-tutorial/2506/3
    """
    # Read input mesh (.vtk) 
    mesh = meshio.read(input_file_vtk) 
    # if .vtk generated from .msh: len(mesh.cells)=2 --> mesh.cells[0]='tetra', mesh.cells[1]='triangle'
    
    # write output mesh (.xml)
    mesh_tetra = meshio.Mesh(points=mesh.points, cells={"tetra": mesh.cells_dict['tetra']})
    #mesh_triangle = meshio.Mesh(points=mesh.points, cells={"triangle": mesh.cells_dict['triangle']}) # --> only surface triangle will be written (surface mesh in .xml, .xdmf)
    #mesh_tetra_triangle = meshio.Mesh(points = mesh.points, cells = {'tetra': mesh.cells_dict['tetra'], 'triangle': mesh.cells_dict['triangle']})
    meshio.write(output_file_xml_xdmf, mesh_tetra) # .xml and .xdmf formats do not recognize "mixed" type. Keep tetraedrons only.

    return 

def msh_to_xml_xdmf(input_file_msh, output_file_xml_xdmf): 
    """
    Args: 3D mesh in MSH format (Gmsh)
    Returns: mesh in XML format, explotable by FEniCS (FEniCS reads XML mesh files with tetrahedrons cells). 
    """
    mesh = meshio.read(input_file_msh) 
    meshio.write(output_file_xml_xdmf, meshio.Mesh(points=mesh.points, cells={'tetra': mesh.cells_dict['tetra']})) 
    #meshio.write(output_file_xml_xdmf, meshio.Mesh(points=mesh.points, cells={'tetra': mesh.cells_dict['tetra'], 'triangle': mesh.cells_dict['triangle']}))

    outputmesh_format = output_file_xml_xdmf.split('.')[-1]
    print('\nmesh file was well converted from "msh" to "{}" format\n'.format(outputmesh_format))
    
    return 


def stl_to_xml_xdmf_2D(input_file_stl, output_file_xml_xdmf): 
    """
    Args: 2D mesh in STL format (stl)
    Returns: mesh in XML format, explotable by FEniCS (FEniCS reads XML mesh files with tetrahedrons cells). 
    """
    mesh = meshio.read(input_file_stl) 
    meshio.write(output_file_xml_xdmf, meshio.Mesh(points = mesh.points, cells = {'triangle': mesh.cells_dict['triangle']})) 
    
    outputmesh_format = output_file_xml_xdmf.split('.')[-1]
    print('\nmesh file was well converted from "stl" to "{}" format (surface mesh)\n'.format(outputmesh_format))

    return 


def mesh_to_xml_xdmf(output_file_xml_xdmf, coordinates, tets): 
    #can be also use as 'mesh_to_vtk': in that case, use output_file_path in .vtk 
    meshio.write(output_file_xml_xdmf, meshio.Mesh(points=coordinates, cells=[("tetra", tets)]))  # .xml and .xdmf formats do not recognize "mixed" type. Keep tetraedrons only.

    outputmesh_format = output_file_xml_xdmf.split('.')[-1]
    print('\nmesh file was well converted from "mesh" to "{}" format\n'.format(outputmesh_format))
    
    return

# Convert to .stl
#################
def msh_to_stl(input_file_path, output_file_stl): 
    #can be also use as 'mesh_to_vtk': in that case, use output_file_path in .vtk 
    
    # .msh to .stl
    msh = meshio.read(input_file_path)
    for cell in msh.cells:
        if cell.type == "triangle":
            triangle_cells = cell.data
    triangle_mesh = meshio.Mesh(points=msh.points, cells={"triangle": triangle_cells})
    meshio.write(output_file_stl, triangle_mesh)
    
    print('\nmesh file was well converted from "msh" to "stl" format (triangle elements only were kept)\n')

    return 

def vtk_to_stl(input_file_vtk, output_file_stl): 
    #can be also use as 'mesh_to_vtk': in that case, use output_file_path in .vtk 
    
    # .msh to .stl
    mesh = meshio.read(input_file_vtk)
    for cell in mesh.cells:
        if cell.type == "triangle":
            triangle_cells = cell.data
    triangle_mesh = meshio.Mesh(points=mesh.points, cells={"triangle": triangle_cells})
    meshio.write(output_file_stl, triangle_mesh)
    
    print('\nmesh file was well converted from "vtk" to "stl" format (triangle elements only were kept)\n')

    return 


def mesh_to_stl(output_file_stl, faces_coords, faces): 
    #can be also use as 'mesh_to_vtk': in that case, use output_file_path in .vtk 

    meshio.write(output_file_stl, meshio.Mesh(points=faces_coords, cells=[("triangle", faces)])) 
    
    print('\nmesh file was well converted from "mesh" to "stl" format (triangle elements only were kept)\n')

    return

def xml_to_stl(input_file_xml, output_file_stl): 
    """
    From FEniCS mesh formats, get .stl format readable in meshlab (Then, smooth, apply filter with the software)
    """

    mesh = fenics.Mesh(input_file_xml) # FEnicS object
    bmesh = fenics.BoundaryMesh(mesh, "exterior")

    # n_faces = bmesh.num_faces() 
    faces = bmesh.cells()
    bmesh_coordinates = bmesh.coordinates()

    #meshio.write(output_file_xml, meshio.Mesh(points=coordinates, cells={'tetra': tets})) 
    meshio.write(output_file_stl, meshio.Mesh(points=bmesh_coordinates, cells={'triangle': faces}))
    
    print('\nmesh file was well converted from "xml" to "stl" format (triangle elements only were kept)\n')

    return

def xdmf_to_stl(input_file_xdmf, output_file_stl): 
    """
    From FEniCS mesh formats, get .stl format readable in meshlab (Then, smooth, apply filter with the software)
    """
    mesh = fenics.Mesh()
    with fenics.XDMFFile(input_file_xdmf) as infile:
        infile.read(mesh)
            
    bmesh = fenics.BoundaryMesh(mesh, "exterior")

    # n_faces = bmesh.num_faces() 
    faces = bmesh.cells()
    bmesh_coordinates = bmesh.coordinates()

    meshio.write(output_file_stl, meshio.Mesh(points=bmesh_coordinates, cells={'triangle': faces}))
    
    print('\nmesh file was well converted from "xdmf" to "stl" format (triangle elements only were kept)\n')

    return

# Convert to .vtk
#################
def gifti_to_vtk(input_file_gii, output_file_vtk):

    surf_img = nib.load(input_file_gii)
    coords = surf_img.agg_data('pointset')
    faces = surf_img.agg_data('triangle')

    meshio.write(output_file_vtk, meshio.Mesh(points=coords, cells=[("triangle", faces)]))

    return

def msh_to_vtk(input_file_msh, output_file_vtk): 
    """
    Args: 3D mesh in MSH format (Gmsh)
    Returns: mesh in XML format, explotable by FEniCS (FEniCS reads XML mesh files with tetrahedrons cells). 
    """
    # .msh: len(mesh.cells)=5 --> mesh.cells[0]='vertex', mesh.cells[1]='vertex', mesh.cells[2]='line', mesh.cells[3]='tetra', mesh.cells[4]='triangle'
    mesh = meshio.read(input_file_msh) 
    meshio.write(output_file_vtk, meshio.Mesh(points = mesh.points, cells = {'tetra': mesh.cells_dict['tetra'], 'triangle': mesh.cells_dict['triangle']})) 

    print('\nmesh file was well converted from "msh" to "vtk" format\n')
    
    return 

def mesh_to_vtk(output_file_mesh, coordinates, tets, faces): 
    #can be also use as 'mesh_to_vtk': in that case, use output_file_path in .vtk 
    meshio.write(output_file_mesh, meshio.Mesh(points=coordinates, cells=[("tetra", tets), ("triangle", faces)])) 

    return 

def xml_to_vtk(input_file_xml, output_file_vtk): 
    """
    From FEniCS mesh formats, get .vtk format
    """
    # .xdmf only contains tetraedron elements
    mesh = fenics.Mesh(input_file_xml) # FEnicS object
    
    coordinates = mesh.coordinates()
    tets = mesh.cells()

    vtk_mesh = meshio.Mesh(points=coordinates, cells={'tetra': tets}) # only tetraedron elements 
    meshio.write(output_file_vtk, vtk_mesh)
    
    print('\nmesh file was well converted from "xml" to "vtk" format\n')

    return 

def xdmf_to_vtk(input_file_xdmf, output_file_vtk): 
    
    # .xdmf only contains tetraedron elements
    mesh = fenics.Mesh()
    with fenics.XDMFFile(input_file_xdmf) as infile:
        infile.read(mesh)
            
    coordinates = mesh.coordinates()
    tets = mesh.cells()

    vtk_mesh = meshio.Mesh(points=coordinates, cells={'tetra': tets}) # only tetraedron elements 
    meshio.write(output_file_vtk, vtk_mesh) 
    
    print('\nmesh file was well converted from "xdmf" to "vtk" format\n')

    return 

# Convert to .gii
#################
import nibabel as nib

def stl_to_gii(input_file_stl, output_file_gii):
    
    # read .stl
    # ---------
    triangle_mesh = meshio.read(input_file_stl)

    # vertices coords
    vertices = triangle_mesh.points

    # faces
    faces = triangle_mesh.cells_dict["triangle"]
    print("number of faces: {}".format(len(triangle_mesh.cells_dict["triangle"])))
    
    #write .gii
    # ---------
     #meshio.write(output_file_gii, meshio.Mesh(points=mesh.points, cells={'triangle': mesh.cells_dict['triangle']}))

    # https://netneurolab.github.io/neuromaps/_modules/neuromaps/images.html

    # Prepare img
    vert = nib.gifti.GiftiDataArray(vertices, 'NIFTI_INTENT_POINTSET',
                                    'NIFTI_TYPE_FLOAT32',
                                    coordsys=nib.gifti.GiftiCoordSystem(3, 3))
    
    tri = nib.gifti.GiftiDataArray(faces, 'NIFTI_INTENT_TRIANGLE',
                                   'NIFTI_TYPE_INT32')
    
    img = nib.GiftiImage(darrays=[vert, tri])

    # Save .gii
    """
    fn = Path(output_file_gii)
    img = nib.load(fn)
    for attr in ('dataspace', 'xformspace'):
        setattr(img.darrays[0].coordsys, attr, val=3)"""
    nib.save(img, output_file_gii)
    
    print('\nmesh file was well converted from "stl" to "gii" format\n')
    
    return 


if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='Convert mesh formats to XML (FEniCS format)')
    
    parser.add_argument('-i', '--input', help='Path to input mesh. Consider using following input formats depending on output format required: .msh, .mesh, .vtk, .stl -> .xml / .msh, .mesh, .vtk, .stl, .xml -> .xdmf / .msh, .mesh, .vtk, .xml, .xdmf -> . stl / .msh, .mesh, .xml, .xdmf -> .vtk / .stl -> .gii', type=str, required=False, 
                        default='./data/brain.mesh')  
    
    parser.add_argument('-o', '--output', help='Path to output mesh. Possible output formats for each input format: \n.msh -> .xml, .xdmf, .stl / .mesh -> .xml, .xdmf, .stl / .stl -> .xml, .xdmf, .gii / .vtk -> .xml, .xdmf, .stl / .xml -> .xdmf, .stl / .xdmf -> .stl', type=str, required=False, 
                        default='./data/brain.xdmf') 

    args = parser.parse_args()

    input_file_path = args.input
    inputmesh_format = input_file_path.split('.')[-1]

    output_file_path = args.output
    outputmesh_format = output_file_path.split('.')[-1]
    
    
    if inputmesh_format == 'msh': # Gmsh mesh
        if outputmesh_format == 'xml' or outputmesh_format == 'xdmf':
            msh_to_xml_xdmf(input_file_path, output_file_path) 
        elif outputmesh_format == 'stl':
            msh_to_stl(input_file_path, output_file_path)
        elif outputmesh_format == 'vtk':
            msh_to_vtk(input_file_path, output_file_path)
            
    
    elif inputmesh_format == 'mesh': # Netgen mesh
            mesh = load_mesh(input_file_path)   
            coordinates, n_nodes = get_nodes(mesh) 
            n_tets, tets = get_tetrahedrons(mesh, n_nodes)
            faces, n_faces = get_face_indices(mesh, n_nodes, n_tets)

            if outputmesh_format == 'xml' or outputmesh_format == 'xdmf':
                mesh_to_xml_xdmf(output_file_path, coordinates, tets) 
            elif outputmesh_format == 'stl':
                mesh_to_stl(output_file_path, coordinates, faces)
            elif outputmesh_format == 'vtk':
                mesh_to_vtk(output_file_path, coordinates, tets, faces)
            
            
    elif inputmesh_format == 'stl': # surface mesh
        if outputmesh_format == 'xml' or outputmesh_format == 'xdmf':
            stl_to_xml_xdmf_2D(input_file_path, output_file_path)
        elif outputmesh_format == 'gii':
            stl_to_gii(input_file_path, output_file_path)
    
    
    elif inputmesh_format == 'vtk':
        if outputmesh_format == 'xml'or outputmesh_format == 'xdmf':
            vtk_to_xml_xdmf(input_file_path, output_file_path) 
        elif outputmesh_format == 'stl':
            vtk_to_stl(input_file_path, output_file_path)


    elif inputmesh_format == 'xml': # legacy FEniCS (.xml) 
        if outputmesh_format == 'xdmf':
            xml_to_xdmf(input_file_path, output_file_path)            
        elif outputmesh_format == 'stl':
            xml_to_stl(input_file_path, output_file_path)
        elif outputmesh_format == 'vtk':
            xml_to_vtk(input_file_path, output_file_path)
    
    
    elif inputmesh_format == 'xdmf':
        if outputmesh_format == 'stl':
            xdmf_to_stl(input_file_path, output_file_path)
        elif outputmesh_format == 'vtk':
            xdmf_to_vtk(input_file_path, output_file_path)
        elif outputmesh_format == 'xml':
            xdmf_to_xml(input_file_path, output_file_path)
  