import fenics
import meshio
import os
import gmsh
import vedo.dolfin
import sys
sys.path.append(".")
import numpy as np
from abc import ABC, abstractmethod
import json

from utils import convert_mesh_to_xml


def mesh3D_to_mesh2D(mesh, cell_type, prune_z=False): 
    """convert 3D gmsh mesh into 2D gmsh mesh"""
    # Code source: https://fenicsproject.discourse.group/t/how-to-use-meshvaluecollection/5106
    cells = np.vstack([cell.data for cell in mesh.cells if cell.type==cell_type])
    cell_data = np.hstack([mesh.cell_data_dict["gmsh:geometrical"][key]
                        for key in mesh.cell_data_dict["gmsh:geometrical"].keys() if key==cell_type])
    # Remove z-coordinates from mesh if we have a 2D cell and all points have the same third coordinate
    points= mesh.points
    if prune_z:
        points = points[:,:2] # remove z
    mesh_new = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})

    return mesh_new


class CreatedMesh2D(ABC): 

    
    def __init__(self, output_filepath_xml, **mesh_parameters):
        self.output_filepath_xml = output_filepath_xml
        self.mesh_parameters = mesh_parameters 

        return None 


    @abstractmethod
    def create_mesh(self):
        pass


class MeshFromGMSH(CreatedMesh2D): 
    """Create a disk-based mesh with GMSH (not possbile with FEniCS)"""

    def __init__(self, output_filepath_xml, **mesh_parameters):
        super().__init__(output_filepath_xml, **mesh_parameters)
        self.input_filepath_msh3D = str(output_filepath_xml.split('.')[0] + '_3D.msh') 
        self.input_filepath_msh2D = str(output_filepath_xml.split('.')[0] + '_2D.msh') 
    
    
    def create_mesh(self):
        """create 2D .xml mesh from 3D .msh mesh"""
        # Transform 3D mesh into 2D mesh and write 2D mesh as a .msh
        mesh3D = meshio.read(self.input_filepath_msh3D) 
        mesh2D = mesh3D_to_mesh2D(mesh3D, "triangle", prune_z=True)
        meshio.write(self.input_filepath_msh2D, mesh2D) # .msh

        # Transform 2D .msh mesh into FEniCS readable mesh format (xml and Mesh())
        convert_mesh_to_xml.msh_to_xml_2D(self.input_filepath_msh2D, self.output_filepath_xml) 


class DiskMesh(MeshFromGMSH):


    def __init__(self, output_filepath_xml, **mesh_parameters):
        super().__init__(output_filepath_xml, **mesh_parameters)
        self.build_gmsh_disk3D()
        self.create_mesh() 
        print("\ndisk mesh created and written down: {}".format(self.output_filepath_xml))


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
        gmsh.write(self.input_filepath_msh3D) # 'halfdisk.msh'

        #gmsh.fltk.run() # plot mesh

        gmsh.finalize()

        mesh = meshio.read(self.input_filepath_msh3D)

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

    def __init__(self, output_filepath_xml, **mesh_parameters):
        #TODO: add error "mesh_parameters sould contain: 'elementsize' and 'radius'"
        super().__init__(output_filepath_xml, **mesh_parameters)
        self.build_gmsh_halfdisk3D()
        self.create_mesh() # Generates self.meshfilepath_xml_2D to be used in braingrowth_main.py
        print("\nhalfdisk mesh created and written down: {}".format(self.output_filepath_xml))


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
        gmsh.write(self.input_filepath_msh3D) # 'halfdisk.msh'

        #gmsh.fltk.run() # plot mesh

        gmsh.finalize()

        mesh = meshio.read(self.input_filepath_msh3D)

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


    def __init__(self, output_filepath_xml, **mesh_parameters):
        #TODO: add error "mesh_parameters sould contain 'elementsize' and 'radius'"super().__init__
        super().__init__(output_filepath_xml, **mesh_parameters)
        self.build_gmsh_quarterdisk3D()
        self.create_mesh() # Generates self.meshfilepath_xml_2D to be used in braingrowth_main.py
        print("\nquarterdisk mesh created and written down: {}".format(self.output_filepath_xml))


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
        gmsh.write(self.input_filepath_msh3D) # 'quarterdisk.msh'

        #gmsh.fltk.run() # plot mesh

        gmsh.finalize()

        mesh = meshio.read(self.input_filepath_msh3D)

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


class MeshFromFEniCS(CreatedMesh2D):


    def __init__(self, output_filepath_xml, **mesh_parameters): 
        super().__init__(output_filepath_xml, **mesh_parameters)


class RectangleMesh(MeshFromFEniCS):
    """Create a rectangle mesh directly from FEniCS function"""


    def __init__(self, output_filepath_xml, **mesh_parameters): 
        super().__init__(output_filepath_xml, **mesh_parameters)
        self.create_mesh()


    def create_mesh(self):
        self.mesh = fenics.RectangleMesh(fenics.Point(self.mesh_parameters['xmin'], self.mesh_parameters['ymin']), \
                                         fenics.Point(self.mesh_parameters['xmax'], self.mesh_parameters['ymax']), \
                                         self.mesh_parameters['num_cell_x'], self.mesh_parameters['num_cell_y'], \
                                         "right/left")
        fenics.File(self.output_filepath_xml) << self.mesh
        print("\nrectangle mesh created and written down: {}".format(self.output_filepath_xml))

    
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate XML 2D mesh (readable by FEniCS)')
    parser.add_argument('-g', '--geometry2d', help='Wished 2D geometry ("disk", "halfdisk", "quarterdisk", "rectangle") ', type=str, required=True)
    parser.add_argument('-prm', '--parameters', help='Parameters for mesh creation: \
                        e.g. if disk-based geometry: {"diskcenter": [0.0, 0.0, 0.0], "radius": 1.0, "elementsize": 0.03} or \
                        e.g. if rectangle geometry:  {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 2.0, "num_cell_x": 60, "num_cell_y": 60} to put between single apostrophe', \
                        type=json.loads, required=True)
    parser.add_argument('-o', '--output', help='Output XML mesh path', type=str, required=True)

    args = parser.parse_args()

    wished_geometry2d = args.geometry2d # e.g. 'disk'; 'rectangle'
    mesh_parameters = args.parameters # e.g. '{"diskcenter": [0.0, 0.0, 0.0], "radius": 1.0, "elementsize": 0.02}'; '{"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 2.0, "num_cell_x": 60, "num_cell_y": 60}'
    output_filepath_xml = args.output # e.g. './data/disk1.xml'; './data/rectangle1.xml'

    if wished_geometry2d == 'disk':
        DiskMesh(output_filepath_xml, **mesh_parameters)

    elif wished_geometry2d == 'halfdisk':
        HalfDiskMesh(output_filepath_xml, **mesh_parameters)
    
    elif wished_geometry2d == 'quarterdisk':
        QuarterDiskMesh(output_filepath_xml, **mesh_parameters)

    elif wished_geometry2d == 'rectangle':
        RectangleMesh(output_filepath_xml, **mesh_parameters)

    mesh = fenics.Mesh(output_filepath_xml) 
    vedo.dolfin.plot(mesh, wireframe=False, text='created 2D mesh', style='paraview', axes=4).close()