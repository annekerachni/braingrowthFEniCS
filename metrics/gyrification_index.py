import numpy as np
# source: https://github.com/rousseau/BrainGrowth/blob/master/impact_mesh_density_GI.py

def rescale_folded_mesh_to_initial_smooth_mesh(initial_smooth_mesh, folded_mesh):
    """scaling down the folded mesh to the size of the initial smooth mesh, taken as the convex hull"""
    
    L1 = (max(initial_smooth_mesh.points[:,0]) - min(initial_smooth_mesh.points[:,0])) / (max(folded_mesh.points[:,0]) - min(folded_mesh.points[:,0]))
    L2 = (max(initial_smooth_mesh.points[:,1]) - min(initial_smooth_mesh.points[:,1])) / (max(folded_mesh.points[:,1]) - min(folded_mesh.points[:,1]))
    L3 = (max(initial_smooth_mesh.points[:,2]) - min(initial_smooth_mesh.points[:,2])) / (max(folded_mesh.points[:,2]) - min(folded_mesh.points[:,2]))
    folded_mesh.points[:,0] = L1 * folded_mesh.points[:,0]
    folded_mesh.points[:,1] = L2 * folded_mesh.points[:,1]
    folded_mesh.points[:,2] = L3 * folded_mesh.points[:,2]
    
    return folded_mesh

def compute_gyrification_index(initial_smooth_mesh, rescaled_folded_mesh):

    Area_convex_hull = 0.0 # = area of the initial unfolded mesh
    Ntmp = np.cross(initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,1]] - initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,0]], initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,2]] - initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,0]])
    Area_convex_hull += 0.5*np.linalg.norm(Ntmp)
    
    Area_rescaled_folded_mesh = 0.0
    Ntmp_2 = np.cross(rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,1]] - rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,0]], rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,2]] - rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,0]])
    Area_rescaled_folded_mesh += 0.5*np.linalg.norm(Ntmp_2)
        
    GI = Area_rescaled_folded_mesh/Area_convex_hull
    
    return GI