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


if __name__ == '__main__':
    
    import sys, os
    sys.path.append(sys.path[0]) 
    sys.path.append(os.path.dirname(os.path.dirname(sys.path[0]))) # braingrowthFEniCS

    import fenics
    from FEM_biomechanical_model import preprocessing
    import json
    
    # mesh
    inputmeshes_dict = {21: "./data/21_28_36GW/transformed_niftis_meshes/transformed_dhcp21GW_isotropic_smoothed.xdmf",
                        28: "./data/21_28_36GW/transformed_niftis_meshes/transformed_dhcp28GW_isotropic_smoothed.xdmf",
                        36: "./data/21_28_36GW/transformed_niftis_meshes/transformed_dhcp36GW_isotropic_sigma_0_3.xdmf"
                        }
    
    # output
    cortical_areas = {}
    
    # compute cortical area for each tGW
    for tGW, inputmesh_path in inputmeshes_dict.items():
        inputmesh_format = inputmesh_path.split('.')[-1]

        if inputmesh_format == "xml":
            mesh0 = fenics.Mesh(inputmesh_path)
            mesh = fenics.Mesh(inputmesh_path)

        elif inputmesh_format == "xdmf":
            mesh0 = fenics.Mesh()
            with fenics.XDMFFile(inputmesh_path) as infile:
                infile.read(mesh0)

            mesh = fenics.Mesh()
            with fenics.XDMFFile(inputmesh_path) as infile:
                infile.read(mesh)
                
        # convert mesh from mm into m (SI)
        mesh = preprocessing.converting_mesh_from_millimeters_into_meters(mesh)
                
        # bmesh
        bmesh = fenics.BoundaryMesh(mesh, "exterior")
        
        # compute cortical surface
        area = compute_mesh_external_surface(bmesh)
        
        cortical_areas[tGW] = area
        #print('cortical area at {}GW = {} m'.format(tGW, area))  
    
    # export json file 
    output_path = './metrics/cortex_areas_21_28_36GW.json'
    
    with open(output_path, 'w') as cortex_areas_json_file:  
        json.dump(cortical_areas, cortex_areas_json_file)

    """
    with open(output_path, 'r') as cortex_areas_json_file:  
        cortex_areas_json_str = cortex_areas_json_file.read()
        cortex_areas_values = json.loads(cortex_areas_json_str)
    """ 
       