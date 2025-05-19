import fenics
import meshio
import numpy as np
import vedo.dolfin
import matplotlib.pyplot as plt

# If input mesh is .xdmf (FEniCS)
#################################
def rescale_initial_smooth_mesh_to_folded_mesh_XDMF(initial_smooth_mesh, folded_mesh):
    """
    Scaling up the initial smooth mesh, taken as the convex hull, to the size of the folded mesh.
    Args:
    - initial_smooth_mesh: .stl mesh
    - folded_mesh: .stl mesh
    Output:
    - (rescaled) initial_smooth_mesh: .stl mesh
    """
    
    L1 = (max(folded_mesh.coordinates()[:,0]) - min(folded_mesh.coordinates()[:,0])) / (max(initial_smooth_mesh.coordinates()[:,0]) - min(initial_smooth_mesh.coordinates()[:,0]))
    L2 = (max(folded_mesh.coordinates()[:,1]) - min(folded_mesh.coordinates()[:,1])) / (max(initial_smooth_mesh.coordinates()[:,1]) - min(initial_smooth_mesh.coordinates()[:,1]))
    L3 = (max(folded_mesh.coordinates()[:,2]) - min(folded_mesh.coordinates()[:,2])) / (max(initial_smooth_mesh.coordinates()[:,2]) - min(initial_smooth_mesh.coordinates()[:,2]))
    initial_smooth_mesh.coordinates()[:,0] = L1 * initial_smooth_mesh.coordinates()[:,0]
    initial_smooth_mesh.coordinates()[:,1] = L2 * initial_smooth_mesh.coordinates()[:,1]
    initial_smooth_mesh.coordinates()[:,2] = L3 * initial_smooth_mesh.coordinates()[:,2]
    
    return initial_smooth_mesh

def compute_gyrification_index_XDMF(rescaled_initial_smooth_bmesh, folded_bmesh):
    """
    Args:
    - initial_rescaled_smooth_mesh: FEniCS boundary mesh
    - folded_mesh: FEniCS mesh
    Output:
    - GI: gyrification index (int)
    """
    ### Convex hull
    Area_convex_hull = 0.0 # = area of the initial unfolded mesh

    for face in rescaled_initial_smooth_bmesh.cells(): # e.g. triangle=[0, 2, 4], with 0, 2, 4 node indices
        Ntmp = np.cross(rescaled_initial_smooth_bmesh.coordinates()[face[1]] - rescaled_initial_smooth_bmesh.coordinates()[face[0]], 
                        rescaled_initial_smooth_bmesh.coordinates()[face[2]] - rescaled_initial_smooth_bmesh.coordinates()[face[0]])
        Area_convex_hull += 0.5*np.linalg.norm(Ntmp)
    
    ### Folded mesh
    Area_folded_mesh = 0.0

    for face in folded_bmesh.cells():
        Ntmp_2 = np.cross(folded_bmesh.coordinates()[face[1]] - folded_bmesh.coordinates()[face[0]], 
                          folded_bmesh.coordinates()[face[2]] - folded_bmesh.coordinates()[face[0]])
        Area_folded_mesh += 0.5*np.linalg.norm(Ntmp_2)
        
    GI = Area_folded_mesh/Area_convex_hull
    
    return GI
    
    return GI

# If input mesh is .stl
#######################
def rescale_initial_smooth_mesh_to_folded_mesh(initial_smooth_mesh, folded_mesh):
    """
    Scaling up the initial smooth mesh, taken as the convex hull, to the size of the folded mesh.
    Args:
    - initial_smooth_mesh: .stl mesh
    - folded_mesh: .stl mesh
    Output:
    - (rescaled) initial_smooth_mesh: .stl mesh
    """
    
    L1 = (max(folded_mesh.points[:,0]) - min(folded_mesh.points[:,0])) / (max(initial_smooth_mesh.points[:,0]) - min(initial_smooth_mesh.points[:,0]))
    L2 = (max(folded_mesh.points[:,1]) - min(folded_mesh.points[:,1])) / (max(initial_smooth_mesh.points[:,1]) - min(initial_smooth_mesh.points[:,1]))
    L3 = (max(folded_mesh.points[:,2]) - min(folded_mesh.points[:,2])) / (max(initial_smooth_mesh.points[:,2]) - min(initial_smooth_mesh.points[:,2]))
    initial_smooth_mesh.points[:,0] = L1 * initial_smooth_mesh.points[:,0]
    initial_smooth_mesh.points[:,1] = L2 * initial_smooth_mesh.points[:,1]
    initial_smooth_mesh.points[:,2] = L3 * initial_smooth_mesh.points[:,2]
    
    return initial_smooth_mesh

def compute_gyrification_index(rescaled_initial_smooth_bmesh, folded_bmesh):
    """
    Args:
    - initial_smooth_mesh: .stl boundary mesh
    - folded_mesh: .stl mesh
    Output:
    - GI: gyrification index (int)
    """
    ### Convex hull
    Area_convex_hull = 0.0 # = area of the initial unfolded mesh
    """
    Ntmp = np.cross(initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,1]] - initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,0]], 
                    initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,2]] - initial_smooth_mesh.points[initial_smooth_mesh.cells_dict['triangle'][:,0]])
    """
    for face in rescaled_initial_smooth_bmesh.cells_dict["triangle"]: # e.g. triangle=[0, 2, 4], with 0, 2, 4 node indices
        Ntmp = np.cross(rescaled_initial_smooth_bmesh.points[face[1]] - rescaled_initial_smooth_bmesh.points[face[0]], 
                        rescaled_initial_smooth_bmesh.points[face[2]] - rescaled_initial_smooth_bmesh.points[face[0]])
        Area_convex_hull += 0.5*np.linalg.norm(Ntmp)
    
    ### Folded mesh
    Area_folded_mesh = 0.0
    """
    Ntmp_2 = np.cross(rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,1]] - rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,0]], 
                      rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,2]] - rescaled_folded_mesh.points[rescaled_folded_mesh.cells_dict['triangle'][:,0]])
    """
    for face in folded_bmesh.cells_dict["triangle"]:
        Ntmp_2 = np.cross(folded_bmesh.points[face[1]] - folded_bmesh.points[face[0]], 
                          folded_bmesh.points[face[2]] - folded_bmesh.points[face[0]])
        Area_folded_mesh += 0.5*np.linalg.norm(Ntmp_2)
        
    GI = Area_folded_mesh/Area_convex_hull
    
    return GI


if __name__ == '__main__':   

    # paths to smooth and folded brain meshes (.stl)
    initial_smooth_mesh_path = "initial_smooth_mesh.stl" # e.g. fetal brain mesh at 21 GW
    folded_mesh_path = "folded_mesh.stl" # e.g. simulated folded brain mesh at 28 GW

    # load smooth and folded brain meshes
    initial_smooth_mesh = meshio.read(initial_smooth_mesh_path)
    folded_mesh = meshio.read(folded_mesh_path) 

    # rescale initial smooth brain mesh onto the folded brain mesh
    rescaled_initial_smooth_mesh = rescale_initial_smooth_mesh_to_folded_mesh(initial_smooth_mesh, folded_mesh)

    # compute gyrification index
    GI = compute_gyrification_index(rescaled_initial_smooth_mesh, folded_mesh)
