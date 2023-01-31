import fenics
import meshio
import os
import gmsh
import vedo.dolfin
import sys
sys.path.append(".")

from .. converters import mesh_format_converters_2D
from . create_xml2Dmesh_abstractclass import CreatedMesh2D


class MeshFromGMSH(CreatedMesh2D): 
    """Create a disk-based mesh with GMSH (not possbile with FEniCS)"""

    def __init__(self, meshfolderpath, geometry_name, **mesh_parameters):
        super().__init__(meshfolderpath, geometry_name, **mesh_parameters)
   

    def define_meshpath(self):
        """Define outputpath for .msh (GMSH) mesh file writing"""
        # if meshfolderpath does not exist, create it
        try:
            if not os.path.exists(self.meshfolderpath):
                os.makedirs(self.meshfolderpath)
        except OSError:
            print ('Error: Creating directory. ' + self.meshfolderpath)

        # Define the output path where to write the mesh
        meshfilepath_gmsh = str(self.meshfolderpath + self.geometry_name + '.msh') # e.g. './data/created_meshes/disk.msh'

        self.meshfilepath_gmsh = meshfilepath_gmsh
    
    
    def create_mesh(self):
        """create 2D .xml mesh from 3D .msh mesh"""
        # Transform 3D mesh into 2D mesh and write 2D mesh as a .msh
        mesh3D = meshio.read(self.meshfilepath_gmsh) # read disk.msh 3D mesh
        mesh2D = mesh_format_converters_2D.mesh3D_to_mesh2D(mesh3D, "triangle", prune_z=True)
        meshfilepath_gmsh_2D = self.meshfilepath_gmsh
        meshio.write(meshfilepath_gmsh_2D, mesh2D)

        # Transform 2D .msh mesh into FEniCS readable mesh format (xml and Mesh())
        self.meshfilepath_xml_2D = mesh_format_converters_2D.convert_msh_to_xml(meshfilepath_gmsh_2D)
        self.mesh = fenics.Mesh(self.meshfilepath_xml_2D) # convert into readable format for FEniCS


class DiskMesh(MeshFromGMSH):


    def __init__(self, meshfolderpath, geometry_name='disk', **mesh_parameters):
        super().__init__(meshfolderpath, geometry_name, **mesh_parameters)
        self.define_meshpath()
        self.build_gmsh_disk3D()
        self.create_mesh() # Generates self.meshfilepath_xml_2D to be used in simulation_xxx.py
        print("\ndisk mesh created and written down: {}".format(self.meshfilepath_xml_2D))


    def build_gmsh_disk3D(self):
        # Inspired from: https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/examples/api/boolean.py
        gmsh.initialize()
        gmsh.model.add("disk")

        gmsh.option.setNumber("Mesh.Algorithm", 1) # 2D mesh algorithm (1=MeshAdapt+Delaunay, 4=MeshAdapt, 5=Delaunay, 6=Frontal) --> 1 or 6
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_parameters['elementsize']) # min size of tetrahedrons
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_parameters['elementsize']) # max size of tetrahedrons

        #R = 1.4; Rs = R *.7; Rt = R *1.25
        #gmsh.model.occ.addCircle(0., 0., 0., 1., 1, angle1=0., angle2=2*pi) # n=?
        gmsh.model.occ.addDisk(self.mesh_parameters['diskcenter'][0], 
                                self.mesh_parameters['diskcenter'][1], 
                                self.mesh_parameters['diskcenter'][2],  
                                self.mesh_parameters['radius'], 
                                self.mesh_parameters['radius']) # addCircle(center[0], center[1], center[2], radius_x, radius_y)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2) # generates 3D mesh
        gmsh.write(self.meshfilepath_gmsh) # 'halfdisk.msh'

        #gmsh.fltk.run() # plot mesh

        gmsh.finalize()

        mesh = meshio.read(self.meshfilepath_gmsh)

        print(mesh) # print mesh information
        """ 
        #<meshio mesh object>
        #  Number of points: xx
        #  Number of cells:
        #    triangle: xx
        #  Cell sets: gmsh:bounding_entities
        #  Point data: gmsh:dim_tags
        #  Cell data: gmsh:geometrical 
        """

    
class HalfDiskMesh(MeshFromGMSH):

    def __init__(self, meshfolderpath, geometry_name='halfdisk', **mesh_parameters):
        #TODO: add error "mesh_parameters sould contain: 'elementsize' and 'radius'"
        super().__init__(meshfolderpath, geometry_name, **mesh_parameters)
        self.define_meshpath()
        self.build_gmsh_halfdisk3D()
        self.create_mesh() # Generates self.meshfilepath_xml_2D to be used in braingrowth_main.py
        print("\nhalfdisk mesh created and written down: {}".format(self.meshfilepath_xml_2D))


    def build_gmsh_halfdisk3D(self):
        # Inspired from: https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/examples/api/boolean.py
        gmsh.initialize()
        gmsh.model.add("halfdisk")

        gmsh.option.setNumber("Mesh.Algorithm", 5) # 2D mesh algorithm (1=MeshAdapt+Delaunay, 4=MeshAdapt, 5=Delaunay, 6=Frontal) --> 5 or 6
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_parameters['elementsize']) # min size of tetrahedrons
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_parameters['elementsize']) # max size of tetrahedrons

        #R = 1.4; Rs = R *.7; Rt = R *1.25
        #gmsh.model.occ.addCircle(0., 0., 0., 1., 1, angle1=0., angle2=2*pi) # n=?
        hd = [
            gmsh.model.occ.addDisk(self.mesh_parameters['diskcenter'][0], 
                                   self.mesh_parameters['diskcenter'][1], 
                                   self.mesh_parameters['diskcenter'][2], 
                                   self.mesh_parameters['radius'], 
                                   self.mesh_parameters['radius']), # addCircle(center[0], center[1], center[2], radius_x, radius_y)

            gmsh.model.occ.addRectangle(-self.mesh_parameters['radius'], 
                                        self.mesh_parameters['diskcenter'][1], 
                                        self.mesh_parameters['diskcenter'][2], 
                                        2*self.mesh_parameters['radius'], 
                                        self.mesh_parameters['radius']) # addRectangle(xmin, ymin, zmin, L, H)
            ]
        gmsh.model.occ.intersect([(2, hd[0])], [(2, hd[1])])
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2) # generates 3D mesh
        gmsh.write(self.meshfilepath_gmsh) # 'halfdisk.msh'

        #gmsh.fltk.run() # plot mesh

        gmsh.finalize()

        mesh = meshio.read(self.meshfilepath_gmsh)

        print(mesh) # print mesh information
        """ 
        #<meshio mesh object>
        #  Number of points: xx
        #  Number of cells:
        #    triangle: xx
        #  Cell sets: gmsh:bounding_entities
        #  Point data: gmsh:dim_tags
        #  Cell data: gmsh:geometrical 
        """


class QuarterDiskMesh(MeshFromGMSH):


    def __init__(self, meshfolderpath, geometry_name='quarterdisk', **mesh_parameters):
        #TODO: add error "mesh_parameters sould contain 'elementsize' and 'radius'"super().__init__
        super().__init__(meshfolderpath, geometry_name, **mesh_parameters)
        self.define_meshpath()
        self.build_gmsh_quarterdisk3D()
        self.create_mesh() # Generates self.meshfilepath_xml_2D to be used in braingrowth_main.py
        print("\nquarterdisk mesh created and written down: {}".format(self.meshfilepath_xml_2D))


    def build_gmsh_quarterdisk3D(self):
        gmsh.initialize()
        gmsh.model.add("quarterdisk")

        gmsh.option.setNumber("Mesh.Algorithm", 5) # 2D mesh algorithm (1=MeshAdapt+Delaunay, 4=MeshAdapt, 5=Delaunay, 6=Frontal)
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_parameters['elementsize']) # min size of tetrahedrons
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_parameters['elementsize']) # max size of tetrahedrons

        qd = [
             gmsh.model.occ.addDisk(self.mesh_parameters['diskcenter'][0], 
                                    self.mesh_parameters['diskcenter'][1], 
                                    self.mesh_parameters['diskcenter'][2], 
                                    self.mesh_parameters['radius'], 
                                    self.mesh_parameters['radius']), # addCircle(center[0], center[1], center[2], radius_x, radius_y)

             gmsh.model.occ.addRectangle(-self.mesh_parameters['radius'], 
                                         self.mesh_parameters['diskcenter'][1], 
                                         self.mesh_parameters['diskcenter'][2], 
                                         2*self.mesh_parameters['radius'], 
                                         self.mesh_parameters['radius']), # addRectangle(xmin, ymin, zmin, L, H)
                                            
             gmsh.model.occ.addRectangle(self.mesh_parameters['diskcenter'][0], 
                                         self.mesh_parameters['diskcenter'][1], 
                                         self.mesh_parameters['diskcenter'][2],
                                         self.mesh_parameters['radius'], 
                                         self.mesh_parameters['radius'])
            ]
        hd, _ = gmsh.model.occ.intersect([(2, qd[0])], [(2, qd[1])])
        gmsh.model.occ.intersect(hd, [(2, qd[2])])
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2) # generates 3D mesh
        gmsh.write(self.meshfilepath_gmsh) # 'quarterdisk.msh'

        #gmsh.fltk.run() # plot mesh

        gmsh.finalize()

        mesh = meshio.read(self.meshfilepath_gmsh)

        print(mesh) 
        """ 
        #<meshio mesh object>
        #  Number of points: xx
        #  Number of cells:
        #    triangle: xx
        #  Cell sets: gmsh:bounding_entities
        #  Point data: gmsh:dim_tags
        #  Cell data: gmsh:geometrical 
        """

    

