"""Create input geometry meshes to be used by FEniCS"""

import fenics 
import numpy as np
from scipy.spatial import cKDTree
from numba import prange


class Mesh:
    

    def __init__(self, meshpath, geometry_type):
    
        self.meshpath = meshpath # FEniCS-format mesh (Mesh())
        self.get_FEniCS_mesh()
        #self.dimension = self.mesh.geometry().dim()
        self.geometry_type = geometry_type

        self.get_brainsurface_bmesh()

        self.compute_geometrical_characteristics()
        self.compute_center_of_gravity()
        self.compute_mesh_spacing()


    def get_FEniCS_mesh(self):
        print("loading mesh...")
        self.mesh = fenics.Mesh(self.meshpath) # mesh in the FEniCS object format
    

    def get_brainsurface_bmesh(self):
        self.brainsurface_bmesh = fenics.BoundaryMesh(self.mesh, "exterior")
        return self.brainsurface_bmesh 


    def compute_geometrical_characteristics(self):
        self.characteristics = {}

        self.characteristics["n_nodes"] = self.mesh.num_vertices() 
        self.characteristics["coordinates"] = self.mesh.coordinates()

        self.characteristics["n_tets"] = self.mesh.num_cells() 
        self.characteristics["n_faces_Surface"] = self.brainsurface_bmesh.num_faces() 
        self.characteristics["n_faces_Volume"] = self.mesh.num_facets()

        #data = self.mesh.data()
        #domains = self.mesh.domains()
        #topology = self.mesh.topology()

        print('n_nodes = {}'.format(self.characteristics["n_nodes"] ))
        print('n_tets = {}'.format(self.characteristics["n_tets"] ))

    
    def compute_center_of_gravity(self):
        maxx = max(self.mesh.coordinates()[:,0])
        minx = min(self.mesh.coordinates()[:,0])
        maxy = max(self.mesh.coordinates()[:,1])
        miny = min(self.mesh.coordinates()[:,1])
        maxz = max(self.mesh.coordinates()[:,2])
        minz = min(self.mesh.coordinates()[:,2])
        print('minx is {}, maxx is {}'.format(minx, maxx))
        print('miny is {}, maxy is {}'.format(miny, maxy))
        print('minz is {}, maxz is {}'.format(minz, maxz))
        self.characteristics['minx'] = minx
        self.characteristics['maxx'] = maxx
        self.characteristics['miny'] = miny
        self.characteristics['maxy'] = maxy
        self.characteristics['minz'] = minz
        self.characteristics['maxz'] = maxz

        center_of_gravity_X = 0.5 * (minx + maxx)
        center_of_gravity_Y = 0.5 * (miny + maxy)
        center_of_gravity_Z = 0.5 * (minz + maxz)    
        self.cog = np.array([center_of_gravity_X, center_of_gravity_Y, center_of_gravity_Z])
        print('COG = [xG:{}, yG:{}, zG:{}]'.format(center_of_gravity_X, center_of_gravity_Y, center_of_gravity_Z))


    def compute_mesh_spacing(self):
        """ 
        Compute input mesh spacing from nodes distances
        Arguments:
            normalized nodes coordinates: np.array(n_nodes,3)
            n_nodes: int
            min_or_ave: str. "min" or "average"
        Returns:
            min or average mesh spacing
        """

        # For each node, calculate the closest other node and distance
        tree = cKDTree(self.mesh.coordinates())
        distance, idex_of_node_in_mesh = tree.query(self.mesh.coordinates(), k=2) # 2 closest neighbours (including the parsed node itself)
        distance_2 = np.zeros(( self.mesh.num_vertices() ), dtype=np.float64)

        for i in prange( self.mesh.num_vertices() ):
            distance_2[i] = distance[i][1]

        self.min_mesh_spacing = np.min(distance_2)
        self.max_mesh_spacing = np.max(distance_2)
        self.average_mesh_spacing = np.mean(distance_2)

        #print("\n{} mesh spacing value for normalized mesh is {:.3f} mm\n".format(min_or_ave, mesh_spacing))