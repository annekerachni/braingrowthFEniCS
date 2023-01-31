import fenics
import os
import vedo.dolfin
import sys
sys.path.append(".")

from . create_xml2Dmesh_abstractclass import CreatedMesh2D


class MeshFromFEniCS(CreatedMesh2D):


    def __init__(self, meshfolderpath, geometry_name, **mesh_parameters): 
        super().__init__(meshfolderpath, geometry_name, **mesh_parameters)


    def define_meshpath(self):
        """Define outputpath for .xml mesh file writing"""
        # if meshfolderpath does not exist, create it
        try:
            if not os.path.exists(self.meshfolderpath):
                os.makedirs(self.meshfolderpath)
        except OSError:
            print ('Error: Creating directory. ' + self.meshfolderpath)
        
        # Define the output path where to write the mesh
        path = str(self.meshfolderpath + self.geometry_name + '.xml') # e.g. './data/created_meshes/rectangle.xml'

        return path


class RectangleMesh(MeshFromFEniCS):
    """Create a rectangle mesh directly from FEniCS function"""


    def __init__(self, meshfolderpath, geometry_name='rectangle', **mesh_parameters): 
        super().__init__(meshfolderpath, geometry_name, **mesh_parameters)
        self.path = self.define_meshpath() # Generates self.path (xml mesh file) to be used in simulation_xxx.py    
        self.create_mesh()


    def create_mesh(self):
        self.mesh = fenics.RectangleMesh(fenics.Point(self.mesh_parameters['xmin'], self.mesh_parameters['ymin']), \
                                         fenics.Point(self.mesh_parameters['xmax'], self.mesh_parameters['ymax']), \
                                         self.mesh_parameters['num_cell_x'], self.mesh_parameters['num_cell_y'], \
                                         "right/left")
        fenics.File(self.path) << self.mesh
        print("\nrectangle mesh created and written down: {}".format(self.path))