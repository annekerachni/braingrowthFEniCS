import fenics
import vedo.dolfin
import matplotlib.pyplot as plt
import argparse
import os, sys

sys.path.append(os.path.dirname(sys.path[0]))  # braingrowthFEniCS
from braingrowth_3D.phenomenological_dynamic_3D.FEM_biomechanical_model.preprocessing import preprocessing

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize .xml or .xdmf input mesh and Compute characteristics')

    parser.add_argument('-i', '--inputmeshpath', help='Input mesh path (.xml format)', type=str, required=False, 
                        default='./data/dHCP_raw/dhcpRight21GW_masked_10000faces_48000tets_refinedWidthCoef15.xdmf')

    parser.add_argument('-v', '--visualization', help='Visualization during simulation', type=bool, required=False, default=False)

    args = parser.parse_args()

    # Get FEniCS input mesh object
    # ----------------------------
    inputmesh_path = args.inputmeshpath
    inputmesh_format = inputmesh_path.split('.')[-1]

    if inputmesh_format == "xml":
        mesh0 = fenics.Mesh(inputmesh_path)

    elif inputmesh_format == "xdmf":
        mesh0 = fenics.Mesh()
        with fenics.XDMFFile(inputmesh_path) as infile:
            infile.read(mesh0)
            
    bmesh0 = fenics.BoundaryMesh(mesh0, "exterior") 

    # mesh characteristics
    ######################
    print('\nmesh characteristics')
    print('********************')
    characteristics0 = preprocessing.compute_geometrical_characteristics(mesh0, bmesh0) # n_nodes, coordinates, n_tets, n_faces_Surface, n_faces_Volume, minx, maxx, miny, maxy, minz, maxz 
    print('n_nodes: {}'.format( characteristics0["n_nodes"] ))
    print('\nn_tets: {}'.format( characteristics0["n_tets"] ))
    print('\nn_faces_Volume: {}'.format( characteristics0["n_faces_Volume"] ))

    print('\nn_faces_Surface: {}\n'.format( characteristics0["n_faces_Surface"] ))

    # initial input mesh 
    ####################
    print('\ninitial input mesh')
    print('******************')

    # Coordinates
    # -----------
    #print('\ncoordinates: \n{}'.format( characteristics0["coordinates"] ))
    print('minX_0: {}, maxX_0: {}'.format( characteristics0['minx'], characteristics0['maxx'] ))
    print('minY_0: {}, maxY_0: {}'.format( characteristics0['miny'], characteristics0['maxy'] ))
    print('minZ_0: {}, maxZ_0: {}'.format( characteristics0['minz'], characteristics0['maxz'] ))

    # Compute input mesh COG
    # ----------------------
    center_of_gravity_0 = preprocessing.compute_center_of_gravity(characteristics0) 
    print('\nCOG_0 = [xG0:{}, yG0:{}, zG0:{}]'.format(center_of_gravity_0[0], center_of_gravity_0[1], center_of_gravity_0[2]))

    # Mesh spacings
    # -------------
    min_mesh_spacing, average_mesh_spacing, max_mesh_spacing = preprocessing.compute_mesh_spacing(mesh0)
    print("\ninitial min mesh_spacing: {:.3f} mm".format(min_mesh_spacing))
    print("initial max mesh_spacing: {:.3f} mm".format(max_mesh_spacing))
    print("initial mean mesh_spacing: {:.3f} mm\n".format(average_mesh_spacing))

    # normalized input mesh 
    #######################
    # Normalize mesh
    print('normalized input mesh')
    print('*********************')
    mesh = preprocessing.normalize_mesh(mesh0, characteristics0, center_of_gravity_0)
    bmesh = fenics.BoundaryMesh(mesh, "exterior") # update bmesh

    # Coordinates
    # -----------
    characteristics = preprocessing.compute_geometrical_characteristics(mesh, bmesh) # n_nodes, coordinates, n_tets, n_faces_Surface, n_faces_Volume, minx, maxx, miny, maxy, minz, maxz 
    """ #print('\ncoordinates: \n{}'.format( characteristics["coordinates"] ))
    print('\nminX: {}, maxX: {}'.format( characteristics['minx'], characteristics['maxx'] ))
    print('minY: {}, maxY: {}'.format( characteristics['miny'], characteristics['maxy'] ))
    print('minZ: {}, maxZ: {}'.format( characteristics['minz'], characteristics['maxz'] )) """

    # Compute input mesh COG
    # ----------------------
    center_of_gravity = preprocessing.compute_center_of_gravity(characteristics) 
    print('\nCOG = [xG:{}, yG:{}, zG:{}]'.format(center_of_gravity[0], center_of_gravity[1], center_of_gravity[2]))

    # Mesh spacings
    # -------------
    min_mesh_spacing, average_mesh_spacing, max_mesh_spacing = preprocessing.compute_mesh_spacing(mesh0)
    print("\nnormalized min mesh_spacing: {:.3f} mm".format(min_mesh_spacing))
    print("normalized max mesh_spacing: {:.3f} mm".format(max_mesh_spacing))
    print("normalized mean mesh_spacing: {:.3f} mm\n".format(average_mesh_spacing))

    # Display normalized mesh
    #########################
    if args.visualization == True:

        fenics.plot(mesh) 
        plt.title("input mesh")
        plt.show()  


        """
        vedo.dolfin.plot(bmesh, 
                        mode='mesh', 
                        text="input mesh", 
                        style='paraview', 
                        axes=4, 
                        interactive=True).clear()
        """

        # visualize the 3D mesh inner elements
        """
        plot = vedo.dolfin.plot(mesh, 
                                mode='mesh', 
                                text="whole mesh", 
                                style='paraview',
                                axes=4, 
                                interactive=False)

        
        # See https://github.com/marcomusy/vedo/issues/133

        mesh_actor = plot.actors[0].lineWidth(0)
        mesh_actor.cutWithPlane(origin=(0., 0., 0.), normal=(1., 0., 0.))

        vedo.dolfin.plot(mesh_actor, # cut_mesh.mesh
                        #wireframe=True,
                        style='paraview',
                        interactive=True)
        """

        # submesh
        class Halfbrain(fenics.SubDomain):
            def inside(self, x, on_boundary):
                return x[1] > 0.0 # TO MODIFY: x[0] or x[1] to visualize the correct orientation of the cut
            
        halfbrain = Halfbrain()
        
        regions = fenics.MeshFunction('size_t', mesh, mesh.topology().dim())
        regions.set_all(0)
        halfbrain.mark(regions, 1)

        submesh = fenics.SubMesh(mesh, regions, 1)

        vedo.dolfin.plot(submesh,
                        #mode='cut mesh',
                        #style='paraview',
                        camera={'pos':(0, -4, 0)}, # TO MODIFY
                        azimuth = 90, # rotation of the scene
                        interactive=True).clear() 
        
        #fenics.plot(submesh) 