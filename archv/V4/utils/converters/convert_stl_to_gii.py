# # Convert surface mesh structure (from braingrowthFEniCS simulation) to .gii format file

import meshio
import nibabel as nib


def read_stl_mesh(input_file_stl):

    triangle_mesh = meshio.read(input_file_stl)

    # vertices coords
    vertices = triangle_mesh.points

    # faces
    faces = triangle_mesh.cells_dict["triangle"]
    print("number of faces: {}".format(len(triangle_mesh.cells_dict["triangle"])))

    return triangle_mesh, faces, vertices


def write_gii_mesh(vertices, faces, output_file_gii):

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
    print(".gii mesh was well written from brain growth simulation result mesh file")

    return



import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert Mesh from STL to GII')

    parser.add_argument('-i', '--inputmesh', help='Input mesh path (.stl)', type=str, required=False, 
    default='./data/surfaces/initial_sphere005refinedcoef5.stl' ) 

    parser.add_argument('-o', '--outputmesh', help='Output mesh path (.gii)', type=str, required=False, 
    default='./data/surfaces/initial_sphere005refinedcoef5.gii') 
    
    args = parser.parse_args()


    # pre-requisites: extract surface mesh at a given time from 4D results (3D + time)
    # --------------------------------------------------------------------------------
    # - load .xmdf result file in Paraview
    # - Apply filter "ExtractRegionSurface"
    # - Choose time of interest
    # - File/Save Data (choose .stl)

    # read 2D mesh (from braingrowth FEniCS simulation)
    # -------------------------------------------------
    triangle_mesh, faces, vertices = read_stl_mesh(args.inputmesh)

    # convert into .gii format (input for Spangy analysis)
    # ----------------------------------------------------
    write_gii_mesh(vertices, faces, args.outputmesh)

