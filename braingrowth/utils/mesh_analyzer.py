import fenics
import vedo.dolfin
import matplotlib.pyplot as plt
import argparse

import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import preprocessing

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Visualize .xml input mesh and Compute characteristics')

    parser.add_argument('-i', '--inputmeshpath', help='Input mesh path (.xml format)', type=str, required=False, 
                        default='./data/sphere_algoDelaunay1_tets005_refinedcoef5.xml')

    parser.add_argument('-v', '--visualization', help='Visualization during simulation', type=bool, required=False, default=False)

    args = parser.parse_args()

    # Get FEniCS input mesh object
    # ----------------------------
    mesh = fenics.Mesh(args.inputmeshpath)
    bmesh = fenics.BoundaryMesh(mesh, "exterior")

    # Display input mesh
    # ------------------
    if args.visualization == True:
        """ 
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
        vedo.dolfin.plot(bmesh, 
                        mode='mesh', 
                        wireframe=True,
                        text="input mesh", 
                        style='paraview', 
                        axes=4).clear()
        """


    # Compute input mesh characteristics
    # ----------------------------------
    characteristics = preprocessing.compute_geometrical_characteristics(mesh, bmesh) # n_nodes, coordinates, n_tets, n_faces_Surface, n_faces_Volume, minx, maxx, miny, maxy, minz, maxz 

    print('\nn_nodes: {}'.format( characteristics["n_nodes"] ))
    #print('\ncoordinates: \n{}'.format( characteristics["coordinates"] ))
    print('\nminX: {}, maxX: {}'.format( characteristics['minx'], characteristics['maxx'] ))
    print('minY: {}, maxY: {}'.format( characteristics['miny'], characteristics['maxy'] ))
    print('minZ: {}, maxZ: {}'.format( characteristics['minz'], characteristics['maxz'] ))

    print('\nn_tets: {}'.format( characteristics["n_tets"] ))
    print('\nn_faces_Volume: {}'.format( characteristics["n_faces_Volume"] ))

    print('\nn_faces_Surface: {}\n'.format( characteristics["n_faces_Surface"] ))


    # Compute input mesh characteristics
    # ----------------------------------
    center_of_gravity = preprocessing.compute_center_of_gravity(characteristics) 
    print('\nCOG = [xG0:{}, yG0:{}, zG0:{}]\n'.format(center_of_gravity[0], center_of_gravity[1], center_of_gravity[2]))

    # Mesh spacings
    # -------------
    min_mesh_spacing, average_mesh_spacing, max_mesh_spacing = preprocessing.compute_mesh_spacing(mesh)
    print("\nmin mesh_spacing: {:.3f} mm".format(min_mesh_spacing))
    print("max mesh_spacing: {:.3f} mm".format(max_mesh_spacing))
    print("mean mesh_spacing: {:.3f} mm\n".format(average_mesh_spacing))