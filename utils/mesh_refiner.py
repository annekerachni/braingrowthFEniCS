import fenics
import vedo.dolfin
import sys
sys.path.append(".")
import numpy as np

from FEM_biomechanical_model import preprocessing

def refine_mesh(mesh_to_refine_XMLpath, refined_mesh_XMLpath, visualization_mode):

    # Load input mesh
    # ---------------
    input_path = mesh_to_refine_XMLpath 
    mesh = fenics.Mesh(input_path) 
    # n_nodes = mesh.num_vertices()
    # n_tets = mesh.num_cells()
    # tets = mesh.cells()

    if visualization_mode == True:
        vedo.dolfin.plot(mesh, wireframe=False, text='mesh to refine', style='paraview', axes=4).close()

    bmesh = fenics.BoundaryMesh(mesh, "exterior")
    # n_surf_nodes = bmesh.num_vertices()
    # n_faces = bmesh.num_cells()
    # faces = bmesh.cells()

    # Refine mesh
    # -----------
    mesh2 = fenics.refine(mesh) 

    # Export refined meshes
    # ---------------------
    file = fenics.File(refined_mesh_XMLpath) # './data/ellipsoid/sphere5_refined_284Ktets.xml'
    file << mesh2

    # Plot meshes
    # -----------
    if visualization_mode == True:
        vedo.dolfin.plot(mesh2, wireframe=False, text='refined mesh', style='paraview', axes=4).close()

    """ fig = plt.figure()
    fig.add_subplot(projection='3d')
    fenics.plot(mesh) 
    plt.title("input mesh")
    plt.show() """

    return


def refine_mesh_on_brainsurface_boundary(mesh_to_refine, brainsurface_bmesh_bbtree, min_mesh_spacing, refinement_width_coef): 
        """ 
        Refine mesh near the brain surface boundary.
        Refined width from boundary = refinement_width_coef * min_mesh_spacing (refinement_width_coef: number of time the minmeshspacing)
        """

        print("refining mesh on the brainsurface boundary...")

        cells_to_refine = fenics.MeshFunction("bool", mesh_to_refine, mesh_to_refine.topology().dim())
        cells_to_refine.set_all(False)
        for cell in fenics.cells(mesh_to_refine):
            p = cell.midpoint()
            _, distance = brainsurface_bmesh_bbtree.compute_closest_entity(p) # compute distance of p to boundingbox
            if distance < refinement_width_coef * min_mesh_spacing: 
                cells_to_refine[cell] = True

        refined_mesh = fenics.refine(mesh_to_refine, cells_to_refine) 

        return refined_mesh


if __name__ == '__main__':

    # REFINE MESH
    # -----------
    """
    mesh_to_refine_XMLpath = './data/ellipsoid.xml'
    refined_mesh_XMLpath = './data/ellipsoid.xml'
    visualization_mode = True
    refine_mesh(mesh_to_refine_XMLpath, refined_mesh_XMLpath, visualization_mode)
    """

    # REFINE MESH NEAR BY BOUNDARY
    # ----------------------------
    import argparse
    from mpi4py import MPI
    import sys, os
    
    sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'FEM_biomechanical_model'))
    from FEM_biomechanical_model import preprocessing
    
    parser = argparse.ArgumentParser(description='Refine mesh near by surface boundary')

    parser.add_argument('-i', '--inputmesh', help='Path to mesh to refine (.xml, .xdmf)', type=str, required=True, 
                        default='./data/sphere.xdmf') 
    
    #parser.add_argument('-o', '--outputmesh', help='Path to output folder', type=str, required=True, 
    #                    default='./data/') 

    parser.add_argument('-rfc', '--refinementwidthcoef', help='refinement width coef (integer > 0)', type=int, required=False, 
                        default=5)  
    # if 0: no refinement 
    # refinement for Points located (refinement_width_coef*min mesh spacing)mm away from the brainsurface boundary

    args = parser.parse_args()

    input_file_path = args.inputmesh
    inputmesh_format = input_file_path.split('.')[-1]
    
    output_file_path = os.path.dirname(input_file_path)
    
    refinement_width_coef = args.refinementwidthcoef 
    
            
    # read FEniCS mesh 
    if inputmesh_format == "xml":
        FEniCSmesh_to_refine = fenics.Mesh(input_file_path)
    
    elif inputmesh_format == 'xdmf':
        FEniCSmesh_to_refine = fenics.Mesh()
        with fenics.XDMFFile(input_file_path) as infile:
            infile.read(FEniCSmesh_to_refine)
    
    vedo.dolfin.plot(FEniCSmesh_to_refine, wireframe=False, text='mesh to refine', style='paraview', axes=4).close()

    # min mesh spacing
    min_mesh_spacing, average_mesh_spacing, max_mesh_spacing = preprocessing.compute_min_mesh_spacing(FEniCSmesh_to_refine)

    # get required args for refinement function
    brainsurface_bmesh = fenics.BoundaryMesh(FEniCSmesh_to_refine, "exterior")

    brainsurface_bmesh_bbtree = fenics.BoundingBoxTree()
    brainsurface_bmesh_bbtree.build(brainsurface_bmesh)  

    # refined mesh
    refined_FEniCSmesh = refine_mesh_on_brainsurface_boundary(FEniCSmesh_to_refine, 
                                                                brainsurface_bmesh_bbtree, 
                                                                min_mesh_spacing, 
                                                                refinement_width_coef)
    
    # generate XDMF refined mesh
    # --------------------------
    if inputmesh_format == "xml":
        output_path = os.path.join(input_file_path.split('.')[0], '_refined.xml')
        fenics.File(output_path) << refined_FEniCSmesh
        
    elif inputmesh_format == 'xdmf':
        output_path = os.path.join(input_file_path.split('.')[0], '_refined.xdmf')
        with fenics.XDMFFile(MPI.COMM_WORLD, output_path) as xdmf:
            xdmf.write(refined_FEniCSmesh)

    vedo.dolfin.plot(refined_FEniCSmesh, wireframe=False, text='refined mesh', style='paraview', axes=4).close()

