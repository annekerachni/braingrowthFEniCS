import numpy as np

def compute_mesh_external_surface(bmesh):
    """mesh external surface"""
    #Code source: https://github.com/rousseau/BrainGrowth/blob/master/geometry.py#L165

    print("Computing the external area of mesh (Cortex folded surface)...")

    coordinates = bmesh.coordinates()
    faces = bmesh.cells()

    cortex_area = 0.0
    for i in range(len(faces)):
        Ntmp = np.cross(coordinates[faces[i,1]] - coordinates[faces[i,0]], 
                        coordinates[faces[i,2]] - coordinates[faces[i,0]])
        cortex_area += 0.5*np.linalg.norm(Ntmp)

    return cortex_area