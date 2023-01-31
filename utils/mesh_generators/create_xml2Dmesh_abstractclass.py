from abc import ABC, abstractmethod


class CreatedMesh2D(ABC): # abstract class

    
    def __init__(self, meshfolderpath, geometry_name, **mesh_parameters):
        self.meshfolderpath = meshfolderpath
        self.geometry_name = geometry_name
        self.mesh_parameters = mesh_parameters 

        return None 
    

    @abstractmethod
    def define_meshpath(self):
        pass


    @abstractmethod
    def create_mesh(self):
        pass

