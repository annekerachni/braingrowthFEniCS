import meshio
import gmsh

# gmsh (ellipsoid 3D --> .msh format) 
# https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/examples/api/boolean.py

def create_MSH_halfellipsoid_mesh(halfsphere_output_path, 
                              center_x, center_y, center_z, 
                              radius, 
                              dilate_coef_x, dilate_coef_y, dilate_coef_z, 
                              tetra_size_min, tetra_size_max, 
                              box_X, box_Y, box_Z, box_dX, box_dY, box_dZ,
                              meshing_algorithm_number,
                              visualisation):

    gmsh.initialize()

    gmsh.model.add("halfsphere")

    # from http://en.wikipedia.org/wiki/Constructive_solid_geometry 
    # http://www.manpagez.com/info/gmsh/gmsh-2.2.6/gmsh_38.php
    # https://www.gmsh.info/doc/texinfo/gmsh.html#t11 
    # https://www.gmsh.info/doc/texinfo/gmsh.html#Mesh-options 
    gmsh.option.setNumber("Mesh.Algorithm3D", meshing_algorithm_number) # 3D mesh algorithm (1=Delaunay, 2=New Delaunay, 4=Frontal, 5=Frontal Delaunay, 6=Frontal Hex, 7=MMG3D, 9=R-tree)
    #gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
    #gmsh.option.setNumber("Mesh.MeshSizeMin", tetra_size_min) # min size of tetrahedrons
    gmsh.option.setNumber("Mesh.MeshSizeMax", tetra_size_max) # max size of tetrahedrons
    #gmsh.option.setNumber("Mesh.Optimize", 1) # "Optimize the mesh to improve the quality of tetrahedral elements"
    #gmsh.option.setNumber("Mesh.AnisoMax", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1) # "Optimize the mesh using Netgen to improve the quality of tetrahedral elements"
    #gmsh.option.setNumber("Mesh.HighOrderOptimize", 2) # TODO remove
    #gmsh.option.setNumber("Mesh.Smoothing", 100) # TODO: gmsh.option.setNumber("Mesh.Smoothing", 1)

    #gmsh.model.occ.addCircle(0.0, 0.0, 0.0, radius, 1, angle1=0., angle2=2*fe.pi)
    sphere = gmsh.model.occ.addSphere(center_x, center_y, center_z, radius)
    gmsh.model.occ.dilate([(3, sphere)], center_x, center_y, center_z, dilate_coef_x, dilate_coef_y, dilate_coef_z)

    box = gmsh.model.occ.addBox(box_X, box_Y, box_Z, box_dX, box_dY, box_dZ) # -1, -1, 0, 2, 2, -2
    
    gmsh.model.occ.cut([(3, sphere)], [(3, box)])
    
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(3)
    #gmsh.model.mesh.refine()
    #gmsh.model.mesh.setOrder(2)
    #gmsh.model.mesh.partition(4)

    gmsh.write(halfsphere_output_path) 

    if visualisation == True:
        gmsh.fltk.run() # plot halfsphere

    gmsh.finalize()

    mesh = meshio.read(halfsphere_output_path)

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

    return mesh

def create_MSH_halfellipsoid_emptiedVentricularZone_mesh(halfsphere_output_path, 
                                                        center_x, center_y, center_z, 
                                                        radius, 
                                                        dilate_coef_x, dilate_coef_y, dilate_coef_z, 
                                                        tetra_size_min, tetra_size_max, 
                                                        box_X, box_Y, box_Z, box_dX, box_dY, box_dZ,
                                                        meshing_algorithm_number,
                                                        visualisation):

    gmsh.initialize()

    gmsh.model.add("halfsphere")

    # from http://en.wikipedia.org/wiki/Constructive_solid_geometry 
    # http://www.manpagez.com/info/gmsh/gmsh-2.2.6/gmsh_38.php
    # https://www.gmsh.info/doc/texinfo/gmsh.html#t11 
    # https://www.gmsh.info/doc/texinfo/gmsh.html#Mesh-options 
    gmsh.option.setNumber("Mesh.Algorithm3D", meshing_algorithm_number) # 3D mesh algorithm (1=Delaunay, 2=New Delaunay, 4=Frontal, 5=Frontal Delaunay, 6=Frontal Hex, 7=MMG3D, 9=R-tree)
    #gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
    #gmsh.option.setNumber("Mesh.MeshSizeMin", tetra_size_min) # min size of tetrahedrons
    gmsh.option.setNumber("Mesh.MeshSizeMax", tetra_size_max) # max size of tetrahedrons
    #gmsh.option.setNumber("Mesh.Optimize", 1) # "Optimize the mesh to improve the quality of tetrahedral elements"
    #gmsh.option.setNumber("Mesh.AnisoMax", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1) # "Optimize the mesh using Netgen to improve the quality of tetrahedral elements"
    #gmsh.option.setNumber("Mesh.HighOrderOptimize", 2) # TODO remove
    #gmsh.option.setNumber("Mesh.Smoothing", 100) # TODO: gmsh.option.setNumber("Mesh.Smoothing", 1)

    #gmsh.model.occ.addCircle(0.0, 0.0, 0.0, radius, 1, angle1=0., angle2=2*fe.pi)
    sphere = gmsh.model.occ.addSphere(center_x, center_y, center_z, radius)
    
    inner_sphere_for_Dirichlet_BCs = gmsh.model.occ.addSphere(center_x, center_y, center_z, radius/5)
    box = gmsh.model.occ.addBox(box_X, box_Y, box_Z, box_dX, box_dY, box_dZ) # -1, -1, 0, 2, 2, -2
    
    halfsphere_emptiedVZ = gmsh.model.occ.cut([(3, sphere)], [(3, inner_sphere_for_Dirichlet_BCs), (3, box)])
    
    #box = gmsh.model.occ.addBox(box_X, box_Y, box_Z, box_dX, box_dY, box_dZ) # -1, -1, 0, 2, 2, -2
    #halfellipsoid_extruded = gmsh.model.occ.cut([(3, ellipsoid_extruded[0][0][1])], [(3, box)])
    
    gmsh.model.occ.dilate([(3, halfsphere_emptiedVZ[0][0][1])], center_x, center_y, center_z, dilate_coef_x, dilate_coef_y, dilate_coef_z)
    
    #gmsh.model.occ.cut([(3, sphere)], [(3, inner_sphere_for_Dirichlet_BCs)])
    #half_ellipsoid_extruded = gmsh.model.occ.cut([(3, half_ellipsoid[0][0][1])], [(3, inner_sphere_for_Dirichlet_BCs)])
    
    #halsphere = gmsh.model.occ.fragment([(3, sphere)], [(3, box), (3, inner_sphere_for_Dirichlet_BCs)])

    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(3)
    #gmsh.model.mesh.refine()
    #gmsh.model.mesh.setOrder(2)
    #gmsh.model.mesh.partition(4)

    gmsh.write(halfsphere_output_path) 

    if visualisation == True:
        gmsh.fltk.run() # plot halfsphere

    gmsh.finalize()

    mesh = meshio.read(halfsphere_output_path)

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

    return mesh



import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate halfsphere mesh in MSH format')

    parser.add_argument('-r', '--sphereradius', help='Basis-sphere radius', type=float, required=False, default=1.)

    parser.add_argument('-c', '--spherecenter', help='Basis-sphere center C coordinates: (Cx, Cy, Cz)', type=tuple, required=False, default=(0.0, 0.0, 0.0))
    
    parser.add_argument('-dc', '--dilatationcoefficients', help='ellipsoid dilatation coefficients: (coef_x, coef_y, coef_z). e.g. sphere: (1.0, 1.0, 1.0); ellipsoid: (0.9, 1.0, 0.7)', type=tuple, required=False, default=(1.0, 1.0, 1.0)) # sphere: (1.0, 1.0, 1.0), BrainGrowth ellipsoid: (0.9, 1.0, 0.7)
    
    parser.add_argument('-e', '--elementsize', help='Element (tetrahedron) size: (min, max)', type=float, required=False, default=(0.05, 0.05))
    
    parser.add_argument('-o', '--output', help='Output path to write 3D sphere/ellipsoid (Gmsh object: .msh)', type=str, required=False, 
                        default='./data/halfsphere/halfsphere_radius1meter.msh' )
    
    parser.add_argument('-ma', '--meshingalgorithm', help='Meshing algorithm number', type=int, required=False, default=1) # 3D mesh algorithm (1=Delaunay, 2=New Delaunay, 4=Frontal, 5=Frontal Delaunay, 6=Frontal Hex, 7=MMG3D, 9=R-tree)
    
    parser.add_argument('-v', '--visualization', help='Plot creating mesh', type=bool, required=False, default=True)

    args = parser.parse_args()

    args = parser.parse_args()

    output_path = args.output 

    sphere_radius = args.sphereradius # radius of the reference sphere from which ellipsoid is built (will correspond to the size of the y direction) --> should be 1.
    center_x = args.spherecenter[0]
    center_y = args.spherecenter[1]
    center_z = args.spherecenter[2]
    coef_x = args.dilatationcoefficients[0]
    coef_y = args.dilatationcoefficients[1]
    coef_z = args.dilatationcoefficients[2]
    tetra_size_min = args.elementsize[0]
    tetra_size_max = args.elementsize[1]
    meshing_algo_number = args.meshingalgorithm
    visualization = args.visualization

    box_X, box_Y, box_Z, box_dX, box_dY, box_dZ = -1, -1, 0, 2, 2, -2
    
    mesh = create_MSH_halfellipsoid_mesh(output_path, 
                                        center_x, center_y, center_z, 
                                        sphere_radius, 
                                        coef_x, coef_y, coef_z, 
                                        tetra_size_min, tetra_size_max, 
                                        box_X, box_Y, box_Z, box_dX, box_dY, box_dZ,
                                        meshing_algo_number,
                                        visualization)
