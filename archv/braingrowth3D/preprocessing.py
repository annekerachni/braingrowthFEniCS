"""Create input geometry meshes to be used by FEniCS"""

import fenics 
import numpy as np
from scipy.spatial import cKDTree
from numba import prange, jit, njit, objmode


class Mesh:
    

    def __init__(self, meshpath):
    
        self.meshpath = meshpath # FEniCS-format mesh (Mesh())
        self.get_FEniCS_mesh()
        self.get_brainsurface_bmesh()


    def get_FEniCS_mesh(self):
        print("\nloading mesh...")
        self.mesh = fenics.Mesh(self.meshpath) # mesh in the FEniCS object format
    

    def get_brainsurface_bmesh(self):
        print("computing brainsurface boundary mesh...")
        self.brainsurface_bmesh = fenics.BoundaryMesh(self.mesh, "exterior")
        return self.brainsurface_bmesh 


    def compute_geometrical_characteristics(self):
        print("computing mesh characteristics...")
        characteristics = {}

        characteristics["n_nodes"] = self.mesh.num_vertices() 
        characteristics["coordinates"] = self.mesh.coordinates()

        characteristics["n_tets"] = self.mesh.num_cells() 
        characteristics["n_faces_Surface"] = self.brainsurface_bmesh.num_faces() 
        characteristics["n_faces_Volume"] = self.mesh.num_facets()

        maxx = max(self.mesh.coordinates()[:,0])
        minx = min(self.mesh.coordinates()[:,0])
        maxy = max(self.mesh.coordinates()[:,1])
        miny = min(self.mesh.coordinates()[:,1])
        maxz = max(self.mesh.coordinates()[:,2])
        minz = min(self.mesh.coordinates()[:,2])
        characteristics['minx'] = minx
        characteristics['maxx'] = maxx
        characteristics['miny'] = miny
        characteristics['maxy'] = maxy
        characteristics['minz'] = minz
        characteristics['maxz'] = maxz

        #data = self.mesh.data()
        #domains = self.mesh.domains()
        #topology = self.mesh.topology()

        return characteristics

    
    def compute_center_of_gravity(self, characteristics):
        print("computing center of gravity...")
        center_of_gravity_X = 0.5 * (characteristics['minx'] + characteristics['maxx'] )
        center_of_gravity_Y = 0.5 * (characteristics['miny'] + characteristics['maxy'])
        center_of_gravity_Z = 0.5 * (characteristics['minz'] + characteristics['maxz'] )    
        cog = np.array([center_of_gravity_X, center_of_gravity_Y, center_of_gravity_Z])

        return cog


    def compute_mesh_spacing(self):
        print("computing mesh spacing...")
        # For each node, calculate the closest other node and distance
        tree = cKDTree(self.mesh.coordinates())
        distance, idex_of_node_in_mesh = tree.query(self.mesh.coordinates(), k=2) # 2 closest neighbours (including the parsed node itself)
        distance_2 = np.zeros(( self.mesh.num_vertices() ), dtype=np.float64)

        for i in prange( self.mesh.num_vertices() ):
            distance_2[i] = distance[i][1]

        min_mesh_spacing = np.min(distance_2)
        max_mesh_spacing = np.max(distance_2)
        average_mesh_spacing = np.mean(distance_2)

        return min_mesh_spacing, average_mesh_spacing, max_mesh_spacing

    
    #@jit(parallel=True, forceobj=True)
    def normalize_mesh(self, characteristics, center_of_gravity):
        """
        Normalize initial mesh coordinates
        Code source: original BrainGrowth https://github.com/rousseau/BrainGrowth/blob/master/normalisation.py
        """

        #with objmode(): 
        print('normalizing mesh...')

        # Compute maximum distance to barycenter 
        maxd = max(max(max(abs(characteristics['maxx']-center_of_gravity[0]), 
                           abs(characteristics['minx']-center_of_gravity[0])), 
                       max(abs(characteristics['maxy']-center_of_gravity[1]), 
                           abs(characteristics['miny']-center_of_gravity[1]))), 
                   max(abs(characteristics['maxz']-center_of_gravity[2]), abs(characteristics['minz']-center_of_gravity[2])))
        
        # Normalize coordinates: change referential to the COG one and normalize coordinates with maximum distance to barycenter 
        self.mesh.coordinates()[:,0] = -(self.mesh.coordinates()[:,0] - center_of_gravity[0])/maxd 
        self.mesh.coordinates()[:,1] = (self.mesh.coordinates()[:,1] - center_of_gravity[1])/maxd
        self.mesh.coordinates()[:,2] = -(self.mesh.coordinates()[:,2] - center_of_gravity[2])/maxd

        print('normalized minx is {}, normalized maxx is {}'.format(min(self.mesh.coordinates()[:,0]), max(self.mesh.coordinates()[:,0])))
        print('normalized miny is {}, normalized maxy is {}'.format(min(self.mesh.coordinates()[:,1]), max(self.mesh.coordinates()[:,1])))
        print('normalized minz is {}, normalized maxz is {}'.format(min(self.mesh.coordinates()[:,2]), max(self.mesh.coordinates()[:,2])))

        # Update brainsurface boundary mesh
        self.brainsurface_bmesh = self.get_brainsurface_bmesh()
