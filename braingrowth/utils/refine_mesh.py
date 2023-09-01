import fenics
import vedo.dolfin
import sys
sys.path.append(".")

import preprocessing


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


import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Refine boundary of an XML mesh')

    parser.add_argument('-i', '--inputmesh', help='Input mesh (.msh) path', type=str, required=False, 
                        default='./data/ellipsoid/sphere5.xml') 

    parser.add_argument('-rfc', '--refinementwidthcoef', help='refinement width coef', type=int, required=False, 
                        default=20)  
    # if 0: no refinement 
    # refinement for Points located (refinement_width_coef*min mesh spacing)mm away from the brainsurface boundary

    parser.add_argument('-or', '--outputrefinedmesh', help='Output refined mesh path (.xdmf)', type=str, required=False, 
                        default='./data/ellipsoid/ellipsoid_refinedcoef20.xml') 


    args = parser.parse_args()


    # load input mesh to refine
    mesh_to_refine = fenics.Mesh(args.inputmesh)
    vedo.dolfin.plot(mesh_to_refine, wireframe=False, text='mesh mesh to refine', style='paraview', axes=4).close()

    # get required args for refinement function
    brainsurface_bmesh = fenics.BoundaryMesh(mesh_to_refine, "exterior")

    brainsurface_bmesh_bbtree = fenics.BoundingBoxTree()
    brainsurface_bmesh_bbtree.build(brainsurface_bmesh)   

    min_mesh_spacing, average_mesh_spacing, max_mesh_spacing = preprocessing.compute_mesh_spacing(mesh_to_refine)

    # choose refinement coef
    refinement_width_coef = args.refinementwidthcoef # refinement for Points located (refinement_width_coef*min mesh spacing)mm away from the brainsurface boundary

    # refined mesh
    refined_mesh = refine_mesh_on_brainsurface_boundary(mesh_to_refine, brainsurface_bmesh_bbtree, min_mesh_spacing, refinement_width_coef)

    # generate refined XML mesh
    # -------------------------
    fenics.File(args.outputrefinedmesh) << refined_mesh
    vedo.dolfin.plot(refined_mesh, wireframe=False, text='refined mesh', style='paraview', axes=4).close()