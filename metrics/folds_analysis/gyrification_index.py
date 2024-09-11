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
    for face in rescaled_folded_bmesh.cells():
        Ntmp_2 = np.cross(rescaled_folded_bmesh.coordinates()[face[1]] - rescaled_folded_bmesh.coordinates()[face[0]], 
                          rescaled_folded_bmesh.coordinates()[face[2]] - rescaled_folded_bmesh.coordinates()[face[0]])
        Area_rescaled_folded_mesh += 0.5*np.linalg.norm(Ntmp_2)
        
    GI = Area_rescaled_folded_mesh/Area_convex_hull
    
    return GI


if __name__ == '__main__':     
    
    import os

    # simulation folded mesh folder
    folded_meshes_folder = './results/simulations_STL_22_30GW/'

    gyrification_indices_simulations_STL_21_30GW = {}
    
    # initial smooth mesh
    initial_smooth_mesh_path =  folded_meshes_folder + "brain_21GW.stl"
    initial_smooth_mesh = meshio.read(initial_smooth_mesh_path)

    for folded_mesh_file in os.listdir(folded_meshes_folder):
        
        tGW = int(folded_mesh_file.split(".stl")[0].split("_")[-1].split("GW")[0])
        
        # simulation folded mesh 
        #numerical_time = 28
        folded_mesh_path = folded_meshes_folder + folded_mesh_file    
        folded_mesh = meshio.read(folded_mesh_path)
        
        # Scale down the folded mesh to the size of the initial unfolded mesh
        rescaled_folded_mesh = rescale_folded_mesh_to_initial_smooth_mesh(initial_smooth_mesh, folded_mesh)               
        
        # Computing GI
        gi_index = compute_gyrification_index(initial_smooth_mesh, rescaled_folded_mesh)
        gyrification_indices_simulations_STL_21_30GW[tGW] = gi_index

        print("{}: {}".format(tGW, gi_index))
        
        #print("\nfolded mesh: '{}':\n--> gyrification index (GI) = {}\n***".format(folded_mesh_path, gi_index))

    gyrification_indices_simulations_STL_21_30GW = dict(sorted(gyrification_indices_simulations_STL_21_30GW.items()))

    tGW_list = list(gyrification_indices_simulations_STL_21_30GW.keys())
    GI_simulations_list = list(gyrification_indices_simulations_STL_21_30GW.values())