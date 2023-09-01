import numpy as np
from scipy.spatial import cKDTree
from numba import prange


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
    maxd = max(
                max(max(abs(characteristics['maxx']-center_of_gravity[0]), abs(characteristics['minx']-center_of_gravity[0])), 
                    max(abs(characteristics['maxy']-center_of_gravity[1]), abs(characteristics['miny']-center_of_gravity[1]))), 

                max(abs(characteristics['maxz']-center_of_gravity[2]), abs(characteristics['minz']-center_of_gravity[2]))
                )
    
    # Normalize coordinates: change referential to the COG one and normalize coordinates with maximum distance to barycenter 
    mesh.coordinates()[:,0] = -(mesh.coordinates()[:,0] - center_of_gravity[0])/maxd 
    mesh.coordinates()[:,1] = (mesh.coordinates()[:,1] - center_of_gravity[1])/maxd
    mesh.coordinates()[:,2] = -(mesh.coordinates()[:,2] - center_of_gravity[2])/maxd

    print('normalized minx is {}, normalized maxx is {}'.format(min(mesh.coordinates()[:,0]), max(mesh.coordinates()[:,0])))
    print('normalized miny is {}, normalized maxy is {}'.format(min(mesh.coordinates()[:,1]), max(mesh.coordinates()[:,1])))
    print('normalized minz is {}, normalized maxz is {}'.format(min(mesh.coordinates()[:,2]), max(mesh.coordinates()[:,2])))

    return mesh
    