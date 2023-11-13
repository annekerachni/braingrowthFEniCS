import fenics
import vedo.dolfin
import sys
sys.path.append(".")

import numpy as np

import preprocessing

def refine_mesh(mesh_to_refine_XMLpath, refined_mesh_XMLpath, visualization_mode):

    # Load input mesh
    # ---------------
    input_path = mesh_to_refine_XMLpath # './data/ellipsoid/sphere5.xml'  './data/gmsh/sphere/sphere_algoDelaunay1_tets005.xml'
    mesh = fenics.Mesh(input_path) # 6761 nodes, 35497 tets
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
    mesh2 = fenics.refine(mesh) # 51100 nodes, 283976 tets
    """ mesh3 = fenics.refine(mesh2) # 394503 nodes, 2271808 tets """

    # Export refined meshes
    # ---------------------
    file = fenics.File(refined_mesh_XMLpath) # './data/ellipsoid/sphere5_refined_284Ktets.xml'
    file << mesh2

    """ file = fenics.File('./data/ellipsoid/sphere5_refined_2Mtets.xml')
    file << mesh3 """

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
    
    """
    python3 -i refine_mesh.py --meshtorefine './data/dhcp/dhcp_atlas/21GW/dhcp21GW_29296faces_107908tets.xml'
                              --refinementwidthcoef 20
                              --refinedmesh './data/dhcp/dhcp_atlas/21GW/dhcp21GW_29296faces_107908tets_refined20.xml'
                              --visualization True
                              --longitudinalaxis 0                           
    """

    # REFINE MESH
    # -----------
    """
    mesh_to_refine_XMLpath = './data/ellipsoid/sphere5.xml'
    refined_mesh_XMLpath = './data/ellipsoid/sphere5_refined_284Ktets.xml'
    visualization_mode = True
    refine_mesh(mesh_to_refine_XMLpath, refined_mesh_XMLpath, visualization_mode)
    """

    # REFINE MESH NEAR BY BOUNDARY
    # ----------------------------
    import argparse
    parser = argparse.ArgumentParser(description='Refine the .xml input mesh near by the cortical surface')

    parser.add_argument('-i', '--meshtorefine', help='Path to the mesh to refine (.xml format)', type=str, required=False, 
    default='./data/dhcp/dhcp_atlas/21GW/dhcp21GW_29296faces_107908tets.xml')

    parser.add_argument('-c', '--refinementwidthcoef', help='Refinement coefficient (node for which distance to surface < refinement coef * min mesh spacing, will be in the refinement zone) e.g. 5; 10: 20', type=int, required=False, 
    default=20)

    parser.add_argument('-o', '--refinedmesh', help='Path where to write the refined mesh (.xml format)', type=str, required=False, 
    default='./data/dhcp/dhcp_atlas/21GW/dhcp21GW_29296faces_107908tets_refined20.xml')

    parser.add_argument('-v', '--visualization', help='Visualization during simulation', type=bool, required=False, 
    default=True)

    parser.add_argument('-l', '--longitudinalaxis', help='Longitudinal axis of the brain mesh. e.g.: 0 for x; 1 for y; 2 for z', type=int, required=False, 
    default=0) # longitudinal axis: y


    args = parser.parse_args()

    # input mesh to refine
    mesh_to_refine_path_XML = args.meshtorefine
    mesh_to_refine = fenics.Mesh(mesh_to_refine_path_XML) # mesh_to_refine = fenics.Mesh(mesh_to_refine_path_XML)
    bmesh = fenics.BoundaryMesh(mesh_to_refine, "exterior") 

    characteristics = preprocessing.compute_geometrical_characteristics(mesh_to_refine, bmesh)
    COG = preprocessing.compute_center_of_gravity(characteristics)

    #vedo.dolfin.plot(mesh_to_refine, wireframe=False, text='mesh mesh to refine', style='paraview', axes=4).close()
    class Halfbrain(fenics.SubDomain):
            def inside(self, x, on_boundary):
                return x[args.longitudinalaxis] > COG[0] # TO MODIFY: x[0] or x[1] to visualize the correct orientation of the cut
            
    halfbrain = Halfbrain()
    
    regions = fenics.MeshFunction('size_t', mesh_to_refine, mesh_to_refine.topology().dim())
    regions.set_all(0)
    halfbrain.mark(regions, 1)

    submesh = fenics.SubMesh(mesh_to_refine, regions, 1)

    # define camera position to visualize the cut half brain mesh
    X = mesh_to_refine.coordinates()[:,0]
    Y = mesh_to_refine.coordinates()[:,1]
    Z = mesh_to_refine.coordinates()[:,2]
    min_negative_half_brain_distance = min( min(np.min(X), np.min(Y)), np.min(Z))

    if args.longitudinalaxis == 0:
        cameraposition = (COG[0] - 20*int(min_negative_half_brain_distance), COG[1], COG[2])
    elif args.longitudinalaxis == 1:
        cameraposition = (COG[0], COG[1] - 20*int(min_negative_half_brain_distance), COG[2])
    else:
        cameraposition = (COG[0], COG[1], COG[2] - 20*int(min_negative_half_brain_distance))
            

    # plot
    vedo.dolfin.plot(submesh,
                    #mode='cut mesh',
                    #style='paraview',
                    camera={'pos':cameraposition}, # TO MODIFY
                    #azimuth = 90, # rotation of the scene
                    interactive=True).close()

    # get required args for refinement function
    brainsurface_bmesh = fenics.BoundaryMesh(mesh_to_refine, "exterior") 
    
    # characteristics of the initial mesh
    characteristics0 = preprocessing.compute_geometrical_characteristics(mesh_to_refine, brainsurface_bmesh) 
    min_mesh_spacing0, average_mesh_spacing0, max_mesh_spacing0 = preprocessing.compute_mesh_spacing(mesh_to_refine)
    print("\ninitial mesh:\nn_nodes : {}\nn_tets : {}\nn_faces : {}\nmean mesh spacing : {}\n".format(characteristics0["n_nodes"], 
                                                                                                  characteristics0["n_tets"],
                                                                                                  characteristics0["n_faces_Surface"],
                                                                                                  average_mesh_spacing0))

    # choose refinement coef
    refinement_width_coef = args.refinementwidthcoef # refinement for Points located (refinement_width_coef*min mesh spacing)mm away from the brainsurface boundary

    # refined mesh
    min_mesh_spacing, average_mesh_spacing, max_mesh_spacing = preprocessing.compute_mesh_spacing(mesh_to_refine)
    brainsurface_bmesh_bbtree = fenics.BoundingBoxTree()
    brainsurface_bmesh_bbtree.build(brainsurface_bmesh)  
    
    refined_mesh = refine_mesh_on_brainsurface_boundary(mesh_to_refine, brainsurface_bmesh_bbtree, min_mesh_spacing, refinement_width_coef)

    # generate XML mesh
    # -----------------
    fenics.File(args.refinedmesh) << refined_mesh
    
    #vedo.dolfin.plot(refined_mesh, wireframe=False, text='refined mesh', style='paraview', axes=4).close()

    halfbrain2 = Halfbrain()
    
    regions2 = fenics.MeshFunction('size_t', refined_mesh, refined_mesh.topology().dim())
    regions2.set_all(0)
    halfbrain2.mark(regions2, 1)

    submesh2 = fenics.SubMesh(refined_mesh, regions2, 1)

    vedo.dolfin.plot(submesh2,
                    #mode='cut mesh',
                    #style='paraview',
                    camera={'pos':cameraposition}, # TO MODIFY
                    #azimuth = 90, # rotation of the scene
                    interactive=True).clear() 
    
    # characteristics of the refined mesh
    brainsurface_bmesh_refined = fenics.BoundaryMesh(refined_mesh, "exterior") 
    characteristics = preprocessing.compute_geometrical_characteristics(refined_mesh, brainsurface_bmesh_refined) 
    min_mesh_spacing, average_mesh_spacing, max_mesh_spacing = preprocessing.compute_mesh_spacing(refined_mesh)
    print("\nrefined mesh:\nn_nodes : {}\nn_tets : {}\nn_faces : {}\nmean mesh spacing : {}\n".format(characteristics["n_nodes"], 
                                                                                                  characteristics["n_tets"],
                                                                                                  characteristics["n_faces_Surface"],
                                                                                                  average_mesh_spacing))
    