import fenics
import meshio
import numpy as np
import vedo.dolfin
import matplotlib.pyplot as plt

def rescale_folded_mesh_to_initial_smooth_mesh(initial_smooth_mesh, folded_mesh):
    """
    Scaling down the folded mesh to the size of the initial smooth mesh, taken as the convex hull.
    Args:
    - initial_smooth_mesh: FEniCS mesh
    - folded_mesh: FEniCS mesh
    Output:
    - (rescaled) folded_mesh: FEniCS mesh
    """
    
    L1 = (max(initial_smooth_mesh.coordinates()[:,0]) - min(initial_smooth_mesh.coordinates()[:,0])) / (max(folded_mesh.coordinates()[:,0]) - min(folded_mesh.coordinates()[:,0]))
    L2 = (max(initial_smooth_mesh.coordinates()[:,1]) - min(initial_smooth_mesh.coordinates()[:,1])) / (max(folded_mesh.coordinates()[:,1]) - min(folded_mesh.coordinates()[:,1]))
    L3 = (max(initial_smooth_mesh.coordinates()[:,2]) - min(initial_smooth_mesh.coordinates()[:,2])) / (max(folded_mesh.coordinates()[:,2]) - min(folded_mesh.coordinates()[:,2]))
    folded_mesh.coordinates()[:,0] = L1 * folded_mesh.coordinates()[:,0]
    folded_mesh.coordinates()[:,1] = L2 * folded_mesh.coordinates()[:,1]
    folded_mesh.coordinates()[:,2] = L3 * folded_mesh.coordinates()[:,2]
    
    return folded_mesh

def compute_gyrification_index(initial_smooth_bmesh, rescaled_folded_bmesh):
    """
    Args:
    - initial_smooth_mesh: FEniCS boundary mesh
    - folded_mesh: FEniCS mesh
    Output:
    - GI: gyrification index (int)
    """
    ### Convex hull
    Area_convex_hull = 0.0 # = area of the initial unfolded mesh
    """
    Ntmp = np.cross(initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,1]] - initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,0]], 
                    initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,2]] - initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,0]])
    """
    for face in initial_smooth_bmesh.cells(): # e.g. triangle=[0, 2, 4], with 0, 2, 4 node indices
        Ntmp = np.cross(initial_smooth_bmesh.coordinates()[face[1]] - initial_smooth_bmesh.coordinates()[face[0]], 
                        initial_smooth_bmesh.coordinates()[face[2]] - initial_smooth_bmesh.coordinates()[face[0]])
        Area_convex_hull += 0.5*np.linalg.norm(Ntmp)
    
    ### Folded mesh
    Area_rescaled_folded_mesh = 0.0
    """
    Ntmp_2 = np.cross(rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,1]] - rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,0]], 
                      rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,2]] - rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,0]])
    """
    for face in initial_smooth_bmesh.cells():
        Ntmp_2 = np.cross(rescaled_folded_bmesh.coordinates()[face[1]] - rescaled_folded_bmesh.coordinates()[face[0]], 
                          rescaled_folded_bmesh.coordinates()[face[2]] - rescaled_folded_bmesh.coordinates()[face[0]])
        Area_rescaled_folded_mesh += 0.5*np.linalg.norm(Ntmp_2)
        
    GI = Area_rescaled_folded_mesh/Area_convex_hull
    
    return GI


if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Export gyrification index (GI) of the folded mesh (by braingrowthFEniCS)') 
    
    parser.add_argument('-i', '--initialsmoothmesh', help='Initial smooth (unfolded) mesh path (.stl)', type=str, required=False, 
    default='./data/surface_meshes/initial_sphere005refinedcoef5.stl') # taken as the convex hull mesh
    
    parser.add_argument('-t', '--numericaltimes', help='numerical time (between 0. and 1.)', type=int, nargs='+', required=False, 
    default=[0.4, 0.6, 0.8, 0.85, 1.0]) 
    
    parser.add_argument('-n', '--nsteps', help='nsteps goal', type=int, required=False, 
    default=100) 
    
    parser.add_argument('-v', '--visualizationmode', help='Visualization during simulation', type=bool, required=False, default=False)
    
    args = parser.parse_args()
    ##
    
    # Load input meshes
    initial_smooth_mesh_path = args.initialsmoothmesh
    initial_smooth_mesh =  meshio.read(initial_smooth_mesh_path)
    
    for numerical_time in args.numericaltimes:
        
        folded_mesh_path_STL = './simulations/references_Fg0/sphere_Ptot_alphaTAN3_nsteps{}_newtonabs3rel2relax1_gmres_sor/sphere_Ptot_alpha3_nsteps{}_newtonabs3rel2relax1_gmres_sor_time{}.stl'.format(args.nsteps, args.nsteps, str(numerical_time).split('.')[0] + "_" + str(numerical_time).split('.')[-1])
        folded_mesh = meshio.read(folded_mesh_path_STL)
        
        if args.visualizationmode == True:
            # plot folded mesh
            folded_mesh_XMLpath = './simulations/references_Fg0/sphere_Ptot_alphaTAN3_nsteps{}_newtonabs3rel2relax1_gmres_sor/sphere_Ptot_alpha3_nsteps{}_newtonabs3rel2relax1_gmres_sor_time{}_volume.xml'.format(args.nsteps, args.nsteps, str(numerical_time).split('.')[0] + "_" + str(numerical_time).split('.')[-1])
            folded_mesh_FEniCS = fenics.Mesh(folded_mesh_XMLpath)

            fenics.plot(folded_mesh_FEniCS) 
            plt.title("folded mesh")
            plt.show() 

        # Scale down the folded mesh to the size of the initial unfolded mesh
        rescaled_folded_mesh = rescale_folded_mesh_to_initial_smooth_mesh(initial_smooth_mesh, folded_mesh)
        
        if args.visualizationmode == True:
            # write rescaled folded mesh as an .xml file 
            rescaled_folded_mesh_XMLpath = './simulations/references_Fg0/sphere_Ptot_alphaTAN3_nsteps{}_newtonabs3rel2relax1_gmres_sor/sphere_Ptot_alpha3_nsteps{}_newtonabs3rel2relax1_gmres_sor_time{}_volume_RESCALED.xml'.format(args.nsteps, args.nsteps, str(numerical_time).split('.')[0] + "_" + str(numerical_time).split('.')[-1])
            meshio.write(rescaled_folded_mesh_XMLpath, meshio.Mesh(points=rescaled_folded_mesh.points, cells={'triangle': rescaled_folded_mesh.cells_dict['triangle']}))
            
            # plot rescaled folded mesh
            rescaled_folded_mesh_FEniCS = fenics.Mesh(rescaled_folded_mesh_XMLpath)
            
            fenics.plot(rescaled_folded_mesh_FEniCS) 
            plt.title("folded mesh scaled down to the size of the initial unfolded mesh")
            plt.show() 
        
        #vedo.dolfin.plot(rescaled_folded_mesh, wireframe=False, text='folded mesh scaled down to the size of the initial smooth mesh', style='paraview', axes=4).close()
        
        
        # Computing GI
        gi_index = compute_gyrification_index(initial_smooth_mesh, rescaled_folded_mesh)
        
        print("\nfolded mesh '{}':\n--> gyrification index (GI) = {}\n***".format(folded_mesh_path_STL, gi_index))