import nibabel as nib
import meshio

def stl_to_gii(input_file_stl, output_file_gii):
    
    # read .stl
    # ---------
    triangle_mesh = meshio.read(input_file_stl)

    # vertices coords
    vertices = triangle_mesh.points

    # faces
    faces = triangle_mesh.cells_dict["triangle"]
    print("number of faces: {}".format(len(triangle_mesh.cells_dict["triangle"])))
    
    #write .gii
    # ---------
     #meshio.write(output_file_gii, meshio.Mesh(points=mesh.points, cells={'triangle': mesh.cells_dict['triangle']}))

    # https://netneurolab.github.io/neuromaps/_modules/neuromaps/images.html

    # Prepare img
    vert = nib.gifti.GiftiDataArray(vertices, 'NIFTI_INTENT_POINTSET',
                                    'NIFTI_TYPE_FLOAT32',
                                    coordsys=nib.gifti.GiftiCoordSystem(3, 3))
    
    tri = nib.gifti.GiftiDataArray(faces, 'NIFTI_INTENT_TRIANGLE',
                                   'NIFTI_TYPE_INT32')
    
    img = nib.GiftiImage(darrays=[vert, tri])

    # Save .gii
    """
    fn = Path(output_file_gii)
    img = nib.load(fn)
    for attr in ('dataspace', 'xformspace'):
        setattr(img.darrays[0].coordsys, attr, val=3)"""
    nib.save(img, output_file_gii)
    
    print('\nmesh file was well converted from "stl" to "gii" format\n')
    
    return 

if __name__ == '__main__':
    files_folder = '/home/administrateur/Bureau/sphere_muCortex/STL/'

    import os
    for surface_file in os.listdir(files_folder):
        #with open(os.path.join(files_folder, surface_file), 'r') as f: 
        stl_to_gii(files_folder + surface_file, (files_folder + surface_file).split('.')[0] + '.gii')