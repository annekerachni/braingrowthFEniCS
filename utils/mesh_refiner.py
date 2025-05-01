import fenics
import vedo.dolfin
import argparse
from mpi4py import MPI
import sys, os

sys.path.append(os.path.dirname(sys.path[0]))
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


def refine_mesh_on_boundary(mesh_to_refine, bmesh_bbtree, mesh_spacing, refinement_width_coef): 
        """ 
        Refine mesh near any surface boundary (identified by the tree of coordinates "bmesh_bbtree").
        Refined width from boundary = refinement_width_coef * min_mesh_spacing (refinement_width_coef: number of time the minmeshspacing)
        """

        print("refining mesh on the brainsurface boundary...")

        cells_to_refine = fenics.MeshFunction("bool", mesh_to_refine, mesh_to_refine.topology().dim())
        cells_to_refine.set_all(False)
        for cell in fenics.cells(mesh_to_refine):
            p = cell.midpoint()
            _, distance = bmesh_bbtree.compute_closest_entity(p) # compute distance of p to boundingbox
            if distance < refinement_width_coef * mesh_spacing: 
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
    parser = argparse.ArgumentParser(description='Refine mesh near by surface boundary')

    parser.add_argument('-i', '--inputmesh', help='Path to 3D mesh to refine (.xml, .xdmf)', type=str, required=False, 
                        default='./data/dHCP_surface_atlas_21GW/fetal_week21_pial_surf_32536faces_106362tets.xdmf') 
    
    #parser.add_argument('-o', '--outputmesh', help='Path to output folder', type=str, required=True, 
    #                    default='./data/') 

    parser.add_argument('-rfc', '--refinementwidthcoef', help='refinement width coef (integer > 0)', type=int, required=False, 
                        default=2)  
    # if 0: no refinement 
    # refinement for Points located (refinement_width_coef*min mesh spacing)mm away from the brainsurface boundary
    
    parser.add_argument('-v', '--visualization', help='Visualization mode (deactivated by default)', type=bool, required=False, default=False)

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
    
    if args.visualization == True:
        vedo.dolfin.plot(FEniCSmesh_to_refine, wireframe=False, text='mesh to refine', style='paraview', axes=4).close()

    # min mesh spacing
    """min_mesh_spacing, average_mesh_spacing, max_mesh_spacing = preprocessing.compute_mesh_spacing(FEniCSmesh_to_refine)"""
    hmin = FEniCSmesh_to_refine.hmin()

    # get required args for refinement function
    brainsurface_bmesh = fenics.BoundaryMesh(FEniCSmesh_to_refine, "exterior")

    brainsurface_bmesh_bbtree = fenics.BoundingBoxTree()
    brainsurface_bmesh_bbtree.build(brainsurface_bmesh)  

    # refined mesh
    refined_FEniCSmesh = refine_mesh_on_boundary(FEniCSmesh_to_refine, 
                                                 brainsurface_bmesh_bbtree, 
                                                 hmin, 
                                                 refinement_width_coef)
    
    # generate XDMF refined mesh
    # --------------------------
    direname = os.path.dirname(os.path.abspath(input_file_path))
    filename = os.path.basename(input_file_path)
    filename_without_format = filename.split('.')[0]
    
    if inputmesh_format == "xml":
        output_mesh_path = os.path.join(direname, filename_without_format + "_refinedWidthCoef{}.xml".format(refinement_width_coef)) 
        fenics.File(output_mesh_path) << refined_FEniCSmesh
        
    elif inputmesh_format == 'xdmf':
        output_mesh_path = os.path.join(direname, filename_without_format + "_refinedWidthCoef{}.xdmf".format(refinement_width_coef)) 
        with fenics.XDMFFile(MPI.COMM_WORLD, output_mesh_path) as xdmf:
            xdmf.write(refined_FEniCSmesh)

    if args.visualization == True:
        vedo.dolfin.plot(refined_FEniCSmesh, wireframe=False, text='refined mesh', style='paraview', axes=4).close()
    
    print('\nmesh was refined near by its surface: {} was well written down'.format(output_mesh_path))

