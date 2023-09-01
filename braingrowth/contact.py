import fenics
import numpy as np
from collections import Counter
from scipy.spatial import cKDTree

"""
def unilateral_contact_modeled_by_penalization_method_Y(r12vec, Kc, epsilon): 

    if (r12vec[1] >= - epsilon):
        return 0.0
    else:
        y_neg = np.array([0., -1., 0.])
        return + Kc * abs(r12vec[1]) * y_neg
"""

def unilateral_contact_modeled_by_penalization_method_XYZ(r12vec, Kc, epsilon):

    r12 = np.linalg.norm(r12vec) 

    if (r12 <= epsilon):
        return np.array([0.0, 0.0, 0.0])
    else:
        r12hat = r12vec/np.linalg.norm(r12vec)
        return + Kc * r12 * r12hat 

def global_collisions_forces(mesh, V, K, vertex2dofs_V, vertexB_2_dofsV_mapping):
    """detect global collisions in the whole mesh and compute global contact forces"""

    # 1. Detect collision in the mesh
    # -------------------------------
    """source: https://fenicsproject.discourse.group/t/speed-up-collision-detection-bounding-box-tree-for-point-cloud/7163/3"""

    bbtree_mesh = fenics.BoundingBoxTree()
    bbtree_mesh.build(mesh, mesh.topology().dim())
    coordinates_mesh = mesh.coordinates()

    boundarymesh = fenics.BoundaryMesh(mesh, "exterior")
    bbtree_bmesh = fenics.BoundingBoxTree()
    bbtree_bmesh.build(boundarymesh, boundarymesh.topology().dim())

    cells = mesh.cells() # cell1: [node1, node2, node3]; cell1: [node1, node4, node10]
    cells_flat = cells.flatten().tolist()
    counter = Counter(cells_flat) # returns all node indices 

    number_of_collisions_experiences_by_each_vertex = [] # debug: [i for i in range(coordinates_mesh.shape[0]) if len(number_of_collisions_experiences_by_each_vertex[:][i]) != 0]
    detected_mesh_nodes_that_experience_collision_IDX = []
    detected_mesh_colliding_nodes_COORDS = []

    for i in range(coordinates_mesh.shape[0]):
        # Compute the number of collisions with cells for node i
        x = fenics.Point(coordinates_mesh[i])
        collision_cells = bbtree_mesh.compute_entity_collisions(x)
        number_of_collisions_experiences_by_each_vertex.append(collision_cells)
        num_collisions = len(collision_cells) # number of collisions experienced by that vertex with any cell. at least equals to the number of neighbour-cells to which the node belongs.

        # get the precomputed value
        num_cells_associated_to_that_vertex = counter[i] # counter[i] returns the number of times node "i" is a component of a cell (e.g. counter[315]: 7 meaning node 315 enters into the composition of 7 cells)

        if num_collisions > num_cells_associated_to_that_vertex:
            detected_mesh_nodes_that_experience_collision_IDX.append(i)
            detected_mesh_colliding_nodes_COORDS.append(mesh.coordinates()[i])
    
    """
    plot_vertices_onto_mesh3D(mesh, mesh_colliding_nodes_coords, "mesh self-collision nodes")
    plot_vertex_onto_mesh3D(mesh, coordinates_mesh[343], "node 343")
    """


    # 2. Compute contact forces to correct these collisions
    # -----------------------------------------------------

    # Reinitialize tmp forces (Function(V))
    fcontact_global = fenics.Function(V) 

    # linear penalty force parameters. 
    epsilon = fenics.DOLFIN_EPS 
    Kc = 6 * float(K) # Kc: contact stiffness; K: bulk modulus. BrainGrowth: Kc=10*K


    treeBoundaryMesh = cKDTree(boundarymesh.coordinates()) 
        
    dist_to_boundary = [] # distance of colliding node to self boundary (r12)
    vector_to_boundary = [] # r12vec
    closest_nodeIDX_on_boundary = []

    for detected_colliding_node_COORDS in detected_mesh_colliding_nodes_COORDS:
        dist2boundary, closest_boundarynode_IDX = treeBoundaryMesh.query(detected_colliding_node_COORDS) # compute closest node on self boundary to colliding surface Point
        
        if dist2boundary == 0.0: # meaning detected colliding node belonds to boundary and distance to boundary means distance to itself
            dist2boundary, closest_boundarynode_IDX = treeBoundaryMesh.query(detected_colliding_node_COORDS, k=2) # considering other boundary nodes than itself
            closest_boundarynode_COORDS = boundarymesh.coordinates()[ closest_boundarynode_IDX[1] ]
            r12vec = closest_boundarynode_COORDS - detected_colliding_node_COORDS 

            dist_to_boundary.append(dist2boundary[1]) 
            vector_to_boundary.append( r12vec ) 
            closest_nodeIDX_on_boundary.append(closest_boundarynode_IDX[1])

        else:          
            closest_boundarynode_COORDS = boundarymesh.coordinates()[closest_boundarynode_IDX]
            r12vec = closest_boundarynode_COORDS - detected_colliding_node_COORDS 

            dist_to_boundary.append(dist2boundary) 
            vector_to_boundary.append( r12vec ) 
            closest_nodeIDX_on_boundary.append(closest_boundarynode_IDX)
            
        
    # Compute anti-collision forces to apply onto colliding mesh nodes and associated projection node on boundary:
    for i, detected_colliding_mesh_node_IDX in enumerate(detected_mesh_nodes_that_experience_collision_IDX): # colliding_boundary_node_L: node idx, node coords
        f12 = unilateral_contact_modeled_by_penalization_method_XYZ(vector_to_boundary[i], Kc, epsilon) # should be a vector (the contact penalty force vector)
        fcontact_global.vector()[ vertex2dofs_V[ detected_colliding_mesh_node_IDX ] ] += f12  
        #fcontact_global.vector()[ mappings.vertex2dofs_V[ closest_nodeIDX_on_boundary[i] ] ] -= f12  # apply opposite force onto projection node on boundary
        fcontact_global.vector()[ vertexB_2_dofsV_mapping[ closest_nodeIDX_on_boundary[i] ] ] -= f12                               

    # 3. Remove velocity previous field to colliding nodes
    # ----------------------------------------------------
    """
    for i, detected_colliding_mesh_node_IDX in enumerate(detected_mesh_nodes_that_experience_collision_IDX):
        v = v_solution.vector()[ mappings.vertex2dofs_V[ detected_colliding_mesh_node_IDX ] ] # v: colliding vertex previous velocity that led to collision --> to undo
        fcontact_global.vector()[ mappings.vertex2dofs_V[ detected_colliding_mesh_node_IDX ] ] -= v
    """

    return fcontact_global