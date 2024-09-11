import numpy as np
from scipy.spatial import cKDTree
from numba import prange
import nibabel as nib
import nibabel.gifti as gifti
from numba import jit
import math
import fenics
import meshio


### for half brain preprocessing ###
####################################
@jit(forceobj=True, parallel=True)
def compute_mesh_BLL_vector(mesh): 

    mesh_coords = mesh.coordinates()
    n_nodes = mesh.num_vertices()

    #minY, maxY = np.min(mesh_coords[:, 1]), np.max(mesh_coords[:, 1])
    #minY_nodeIDX = np.where(mesh_coords[:, 1] == minY)[0][0]

    farthestneighb_distances = np.zeros((n_nodes), dtype=np.float64)  
    farthest_node_IDX = np.zeros((n_nodes), dtype=np.float64) 
    mesh_BLL_vectors = np.zeros((n_nodes, 3), dtype=np.float64) 
    
    # Get the max length
    tree = cKDTree(mesh_coords[:])
    for ele in prange(len(mesh_coords[:])):
        farthestneighb_distance_maxY, index_of_nodes_in_mesh = tree.query(mesh_coords[:], k=n_nodes) # list of distances between surface_node to other nodes, from closest to farthest.
        farthestneighb_distances[ele] = index_of_nodes_in_mesh[-1] 
        farthest_node_IDX[ele] = index_of_nodes_in_mesh[-1]
        mesh_BLL_vectors[ele][:] = mesh_coords[ele] - mesh_coords[farthest_node_IDX[ele]]
        
    """ distance_to_farthest_other_surfacenode = np.zeros((n_surface_nodes), dtype=np.float64)
    for i in range(n_surface_nodes):
        distance_to_farthest_other_surfacenode[i] = distances[i][n_surface_nodes-1] # dist to farthest other surface node """  
    
    mesh_BLL = np.max(farthestneighb_distances)
    ele_BLL = np.where(np.max(mesh_BLL_vectors[:]) == mesh_BLL)
    mesh_BLL_vector = mesh_BLL_vectors[ele_BLL]
    #mesh_BLL_vector = mesh_coords[real_maxY_IDX] - mesh_coords[minY_nodeIDX] # max distance between two surface nodes  
    
    # Get the indices of the two farthest corresponding surface nodes
    #BLL_nodes_indices = np.where(farthestneighb_distances == mesh_BLL)  

    return mesh_BLL_vector 

def register_mesh_towards_Y_axis(mesh, mesh_BLL_vector):
    # https://fenicsproject.discourse.group/t/how-to-rotate-a-mesh-in-fenics/7162

    Y = np.array([0., 1., 0.])
    BLL_vect_unit = mesh_BLL_vector / np.linalg.norm(mesh_BLL_vector)

    dot_product = np.dot(Y, BLL_vect_unit)
    angle_RAD = np.arccos(dot_product) # in radians
    angle_DEG = angle_RAD * 180 / math.pi

    minY_coords = np.min(mesh.coordinates()[:, 1])
    mesh.rotate(angle_DEG, 2, fenics.Point(*minY_coords)) # angle (degree), axis around which to rotate (Z:2), point 

    return mesh

### for whole brain preprocessing ###
#####################################
def compute_geometrical_characteristics(mesh, bmesh):
    characteristics = {}

    characteristics["n_nodes"] = mesh.num_vertices() 
    characteristics["coordinates"] = mesh.coordinates()

    characteristics["n_tets"] = mesh.num_cells() 
    characteristics["n_faces_Surface"] = bmesh.num_faces() 
    characteristics["n_faces_Volume"] = mesh.num_facets()

    maxx = max(mesh.coordinates()[:,0])
    minx = min(mesh.coordinates()[:,0])
    maxy = max(mesh.coordinates()[:,1])
    miny = min(mesh.coordinates()[:,1])
    maxz = max(mesh.coordinates()[:,2])
    minz = min(mesh.coordinates()[:,2])
    characteristics['minx'] = minx
    characteristics['maxx'] = maxx
    characteristics['miny'] = miny
    characteristics['maxy'] = maxy
    characteristics['minz'] = minz
    characteristics['maxz'] = maxz

    #data = mesh.data()
    #domains = mesh.domains()
    #topology = mesh.topology()

    return characteristics

def compute_center_of_gravity(characteristics):
    center_of_gravity_X = 0.5 * (characteristics['minx'] + characteristics['maxx'] )
    center_of_gravity_Y = 0.5 * (characteristics['miny'] + characteristics['maxy'])
    center_of_gravity_Z = 0.5 * (characteristics['minz'] + characteristics['maxz'] )    
    cog = np.array([center_of_gravity_X, center_of_gravity_Y, center_of_gravity_Z])

    return cog


def compute_mesh_spacing(mesh):
    # For each node, calculate the closest other node and distance
    tree = cKDTree(mesh.coordinates())
    distance, idex_of_node_in_mesh = tree.query(mesh.coordinates(), k=2) 
    distance_2 = np.zeros(( mesh.num_vertices() ), dtype=np.float64)

    for i in prange( mesh.num_vertices() ):
        distance_2[i] = distance[i][1]

    min_mesh_spacing = np.min(distance_2)
    max_mesh_spacing = np.max(distance_2)
    average_mesh_spacing = np.mean(distance_2)

    return min_mesh_spacing, average_mesh_spacing, max_mesh_spacing 

def normalize_mesh(mesh, characteristics, center_of_gravity):
    """
    Normalize initial mesh coordinates
    Code source: original BrainGrowth https://github.com/rousseau/BrainGrowth/blob/master/normalisation.py
    """
    
    #with objmode(): 
    """
    if halforwholebrain == "half":
        # Compute maximum distance to barycenter 
        maxd = max(
                    max(max(abs(characteristics['maxx']-center_of_gravity[0]), abs(characteristics['minx']-center_of_gravity[0])), abs(characteristics['maxy']-characteristics['miny'])), 
                    max(abs(characteristics['maxz']-center_of_gravity[2]), abs(characteristics['minz']-center_of_gravity[2]))
                    )
        
        # Normalize coordinates: change referential to the COG one and normalize coordinates with maximum distance to barycenter           
        mesh.coordinates()[:,0] = -(mesh.coordinates()[:,0] - center_of_gravity[0])/maxd

        if leftorrightlobe == "left": 
            mesh.coordinates()[:,1] = (mesh.coordinates()[:,1] - characteristics['maxy'])/maxd # conserves negatives Y coordinates 
        elif leftorrightlobe == "right": 
            mesh.coordinates()[:,1] = (mesh.coordinates()[:,1] - characteristics['miny'])/maxd 

        mesh.coordinates()[:,2] = -(mesh.coordinates()[:,2] - center_of_gravity[2])/maxd
    """

    # Compute maximum distance to barycenter 
    maxX = max( abs(characteristics['maxx'] - center_of_gravity[0]), abs(characteristics['minx'] - center_of_gravity[0]) )
    maxY = max( abs(characteristics['maxy'] - center_of_gravity[1]), abs(characteristics['miny'] - center_of_gravity[1]) )
    maxZ = max( abs(characteristics['maxz'] - center_of_gravity[2]), abs(characteristics['minz'] - center_of_gravity[2]) )

    max_distance_to_COG = max( max(maxX, maxY), maxZ)

    # longitudinal axis of the brain should be oriented in the Y direction (RAS orientation)
    # supposing the longidtudinal axis of the initial mesh is well parallel to one axis x, y or z. If oblique initial mesh, does not wotk.
    if max_distance_to_COG == maxX:
        # Normalize coordinates: change referential to the COG one and normalize coordinates with maximum distance to barycenter 
        tmp = mesh.coordinates()[:,0].copy()
        mesh.coordinates()[:,0] = - (mesh.coordinates()[:,1] - center_of_gravity[1])/max_distance_to_COG 
        mesh.coordinates()[:,1] = (tmp - center_of_gravity[0])/max_distance_to_COG
        mesh.coordinates()[:,2] = - (mesh.coordinates()[:,2] - center_of_gravity[2])/max_distance_to_COG
        
    elif max_distance_to_COG == maxY:
        # Normalize coordinates: change referential to the COG one and normalize coordinates with maximum distance to barycenter 
        mesh.coordinates()[:,0] = - (mesh.coordinates()[:,0] - center_of_gravity[0])/max_distance_to_COG 
        mesh.coordinates()[:,1] = (mesh.coordinates()[:,1] - center_of_gravity[1])/max_distance_to_COG
        mesh.coordinates()[:,2] = - (mesh.coordinates()[:,2] - center_of_gravity[2])/max_distance_to_COG

    elif max_distance_to_COG == maxZ: # 
        # Normalize coordinates: change referential to the COG one and normalize coordinates with maximum distance to barycenter 
        mesh.coordinates()[:,0] = - (mesh.coordinates()[:,0] - center_of_gravity[0])/max_distance_to_COG 
        mesh.coordinates()[:,1] = (mesh.coordinates()[:,2] - center_of_gravity[2])/max_distance_to_COG
        mesh.coordinates()[:,2] = - (mesh.coordinates()[:,1] - center_of_gravity[1])/max_distance_to_COG


    print('normalized minx: {}, normalized maxx: {}'.format(min(mesh.coordinates()[:,0]), max(mesh.coordinates()[:,0])))
    print('normalized miny: {}, normalized maxy: {}'.format(min(mesh.coordinates()[:,1]), max(mesh.coordinates()[:,1])))
    print('normalized minz: {}, normalized maxz: {}'.format(min(mesh.coordinates()[:,2]), max(mesh.coordinates()[:,2])))

    return mesh

######
######

def reorient_mesh(mesh, characteristics, center_of_gravity):
    """
    place mesh COG to Origin (0., 0., 0.) + Reorient brain: the 2 hemispheres in the X direction, length in Y direction, height in Z direction
    """
    
    # Compute maximum distance to barycenter 
    maxX = max( abs(characteristics['maxx'] - center_of_gravity[0]), abs(characteristics['minx'] - center_of_gravity[0]) )
    maxY = max( abs(characteristics['maxy'] - center_of_gravity[1]), abs(characteristics['miny'] - center_of_gravity[1]) )
    maxZ = max( abs(characteristics['maxz'] - center_of_gravity[2]), abs(characteristics['minz'] - center_of_gravity[2]) )

    max_distance_to_COG = max( max(maxX, maxY), maxZ)

    # longitudinal axis of the brain should be oriented in the Y direction (RAS orientation)
    # supposing the longidtudinal axis of the initial mesh is well parallel to one axis x, y or z. If oblique initial mesh, does not wotk.
    if max_distance_to_COG == maxX:
        # Normalize coordinates: change referential to the COG one and normalize coordinates with maximum distance to barycenter 
        tmp = mesh.coordinates()[:,0].copy()
        mesh.coordinates()[:,0] = - (mesh.coordinates()[:,1] - center_of_gravity[1]) 
        mesh.coordinates()[:,1] = (tmp - center_of_gravity[0])
        mesh.coordinates()[:,2] = - (mesh.coordinates()[:,2] - center_of_gravity[2])
        
    elif max_distance_to_COG == maxY:
        # Normalize coordinates: change referential to the COG one and normalize coordinates with maximum distance to barycenter 
        mesh.coordinates()[:,0] = - (mesh.coordinates()[:,0] - center_of_gravity[0]) 
        mesh.coordinates()[:,1] = (mesh.coordinates()[:,1] - center_of_gravity[1])
        mesh.coordinates()[:,2] = - (mesh.coordinates()[:,2] - center_of_gravity[2])

    elif max_distance_to_COG == maxZ: # 
        # Normalize coordinates: change referential to the COG one and normalize coordinates with maximum distance to barycenter 
        mesh.coordinates()[:,0] = - (mesh.coordinates()[:,0] - center_of_gravity[0]) 
        mesh.coordinates()[:,1] = (mesh.coordinates()[:,2] - center_of_gravity[2])
        mesh.coordinates()[:,2] = - (mesh.coordinates()[:,1] - center_of_gravity[1])


    print('normalized minx: {}, normalized maxx: {}'.format(min(mesh.coordinates()[:,0]), max(mesh.coordinates()[:,0])))
    print('normalized miny: {}, normalized maxy: {}'.format(min(mesh.coordinates()[:,1]), max(mesh.coordinates()[:,1])))
    print('normalized minz: {}, normalized maxz: {}'.format(min(mesh.coordinates()[:,2]), max(mesh.coordinates()[:,2])))

    return mesh

def converting_mesh_from_millimeters_into_meters(mesh): # FEniCS mesh (.xdmf; .xml)
    coords_in_mm = mesh.coordinates()[:].copy()
    mesh.coordinates()[:] = coords_in_mm / 1000
    return mesh

def converting_mesh_from_meters_into_millimeters(mesh): # FEniCS mesh (.xdmf; .xml)
    coords_in_m = mesh.coordinates()[:].copy()
    mesh.coordinates()[:] = coords_in_m * 1000
    return mesh