import meshio
import fenics
import numpy as np
from scipy.spatial import cKDTree
from numba import prange
from mpi4py import MPI
import vedo.dolfin


def msh_to_xdmf(input_file_msh, output_file_xdmf): 

    mesh = meshio.read(input_file_msh) 
    meshio.write(output_file_xdmf, meshio.Mesh(points=mesh.points, cells={'tetra': mesh.cells_dict['tetra']})) 
    #meshio.write(output_file_xdmf, meshio.Mesh(points=mesh.points, cells={'tetra': mesh.cells_dict['tetra'], 'triangle': mesh.cells_dict['triangle']}))
    print('mesh file was well converted from "msh" to "xdmf" format\n')
    
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

def compute_min_mesh_spacing(FEniCSmesh):
    print("computing mesh spacing...")
    # For each node, calculate the closest other node and distance
    tree = cKDTree(FEniCSmesh.coordinates())
    distance, idex_of_node_in_mesh = tree.query(FEniCSmesh.coordinates(), k=2) 
    distance_2 = np.zeros(( FEniCSmesh.num_vertices() ), dtype=np.float64)

    for i in prange( FEniCSmesh.num_vertices() ):
        distance_2[i] = distance[i][1]

    min_mesh_spacing = np.min(distance_2)
    #max_mesh_spacing = np.max(distance_2)
    #average_mesh_spacing = np.mean(distance_2)
     
    return min_mesh_spacing



import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Mesh to XML Fenics')

    parser.add_argument('-i', '--inputmesh', help='Input mesh (.msh) path', type=str, required=False, 
                        default='./data/gmsh/sphere/sphere_algoDelaunay1_tets005.msh') 
    
    parser.add_argument('-o', '--outputmesh', help='Output mesh path (.xdmf)', type=str, required=False, 
                        default='./data/gmsh/sphere/sphere_algoDelaunay1_tets005.xdmf') 

    parser.add_argument('-rfc', '--refinementwidthcoef', help='refinement width coef', type=int, required=False, 
                        default=10)  
    # if 0: no refinement 
    # refinement for Points located (refinement_width_coef*min mesh spacing)mm away from the brainsurface boundary

    parser.add_argument('-or', '--outputrefinedmesh', help='Output refined mesh path (.xdmf)', type=str, required=False, 
                        default='./data/gmsh/sphere/sphere_algoDelaunay1_tets005_refinedcoef10.xdmf') 


    args = parser.parse_args()

    msh_input_file_path = args.inputmesh
    xdmf_output_file_path = args.outputmesh
    refinement_width_coef = args.refinementwidthcoef 
    refined_xdmf_output_file_path = args.outputrefinedmesh


    # convert mesh from .msh to .xdmf
    # -------------------------------
    mesh = meshio.read(msh_input_file_path) 
    msh_to_xdmf(msh_input_file_path, xdmf_output_file_path)


    # refine .xdmf mesh
    # -----------------

    """ mesh = fenics.Mesh()
    f = fenics.XDMFFile(xdmf_output_file_path)
    f.read(mesh)

    outfile = fenics.XDMFFile("mesh_to_refine.xdmf").write(mesh) """

    # read FEniCS mesh 
    #FEniCSmesh_to_refine = fenics.Mesh(xdmf_output_file_path)
    FEniCSmesh_to_refine = fenics.Mesh()
    with fenics.XDMFFile(xdmf_output_file_path) as infile:
        infile.read(FEniCSmesh_to_refine)

    #comm = MPI.COMM_WORLD # https://fenicsproject.discourse.group/t/mesh-conversion-xdmf-in-a-mpi-environment/3499
    #size = comm.Get_size()
    #rank = comm.Get_rank()
    
    #f = fenics.XDMFFile(comm, xdmf_output_file_path)
    #f.read(FEniCSmesh_to_refine, True)
    vedo.dolfin.plot(FEniCSmesh_to_refine, wireframe=False, text='mesh to refine', style='paraview', axes=4).close()

    # min mesh spacing
    min_mesh_spacing = compute_min_mesh_spacing(FEniCSmesh_to_refine)

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
    with fenics.XDMFFile(MPI.COMM_WORLD, refined_xdmf_output_file_path) as xdmf:
        xdmf.write(refined_FEniCSmesh)

    vedo.dolfin.plot(refined_FEniCSmesh, wireframe=False, text='refined mesh', style='paraview', axes=4).close()


    


