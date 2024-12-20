import fenics
import numpy as np
from collections import Counter
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


# Contact forces that correct already uccured collisions
# ------------------------------------------------------
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

    if (r12 <= epsilon): # does not taking into account "small" penetration to avoid correcting mesh close contact, that is acceptable.
        return np.array([0.0, 0.0, 0.0])
    else:
        r12hat = r12vec/np.linalg.norm(r12vec)
        return + Kc * r12 * r12hat # penalty-force formula
    

def unilateral_contact_modeled_by_penalization_method_XYZ_smallcontactcorrected(r12vec, Kc):

    r12 = np.linalg.norm(r12vec) 

    r12hat = r12vec/np.linalg.norm(r12vec)
    return + Kc * r12 * r12hat # penalty-force formula


# Contact forces that anticipate upcoming collisions
# --------------------------------------------------
def contact_force_XYZ_Tallinen(r12vec, Kc, mean_mesh_spacing, epsilon):

    r12 = np.linalg.norm(r12vec) 

    if (r12 <= epsilon):
        return np.array([0.0, 0.0, 0.0])
    else:
        r12hat = r12vec/np.linalg.norm(r12vec)

        #mean_mesh_spacing = 0.03
        repuls_skin = 0.2 * mean_mesh_spacing
        return + Kc * (r12 - repuls_skin)/ repuls_skin * mean_mesh_spacing**2 * r12hat # T.Tallinen contact force expression

##############
##############

def correct_collisions(mesh, V, K, vertex2dofs_V, vertexB_2_dofsV_mapping):
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
        
        #f12 = unilateral_contact_modeled_by_penalization_method_XYZ(vector_to_boundary[i], Kc, epsilon) # should be a vector (the contact penalty force vector)
        f12 = unilateral_contact_modeled_by_penalization_method_XYZ_smallcontactcorrected(vector_to_boundary[i], Kc)
        
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

  
def correct_collisions_2(mesh, V, K, 
                         vertex2dofs_V, vertexB_2_dofsV_mapping, 
                         mean_mesh_spacing):
    """detect global collisions in the whole mesh and compute global contact forces"""

    """source: https://fenicsproject.discourse.group/t/speed-up-collision-detection-bounding-box-tree-for-point-cloud/7163/3"""

    # mesh
    # ----
    bbtree_mesh = fenics.BoundingBoxTree()
    bbtree_mesh.build(mesh, mesh.topology().dim())
    coordinates_mesh = mesh.coordinates()
    
    c_to_v = mesh.topology()(mesh.topology().dim(), 0)
    
    cells = mesh.cells() # cell1: [node1, node2, node3]; cell1: [node1, node4, node10]
    cells_flat = cells.flatten().tolist()
    counter = Counter(cells_flat) # returns all node indices 

    # bmesh
    # -----
    boundarymesh = fenics.BoundaryMesh(mesh, "exterior")
    bbtree_boundarymesh = fenics.BoundingBoxTree()
    #treeBoundaryMesh = cKDTree(boundarymesh.coordinates()) 
    bbtree_boundarymesh.build(boundarymesh, boundarymesh.topology().dim())
    coordinates_boundarymesh = boundarymesh.coordinates()
    

    # 1. Detect collisions in the mesh
    ##################################
    
    # 1.1 Detect surface node IDX (indexation of the whole mesh) that experience "real" collision with "extra" tets of the mesh
    # -------------------------------------------------------------------------------------------------------------------------
    #detected_collisionTets_for_each_meshVertex = [] # debug: [i for i in range(coordinates_mesh.shape[0]) if len(number_of_collisions_experiences_by_each_vertex[:][i]) != 0]
    #colliding_meshVertexIDX_which_are_on_boundary = []
    collidingMeshNodes_detectedTetsIDXs = []

    # Compute the number of collisions with tets for each mesh node i
    for i in range(coordinates_mesh.shape[0]):
        if mesh.coordinates()[i] in boundarymesh.coordinates():
            x = fenics.Point(*coordinates_mesh[i])
            all_collision_tets_IDX = bbtree_mesh.compute_entity_collisions(x)  # VERTEX indexation 
            """
            https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.geometry.html
            https://fenicsproject.org/olddocs/dolfin/1.3.0/python/programmers-reference/cpp/mesh/GenericBoundingBoxTree.html
            --> compute_closest_entity(x) instead of compute_entity_collisions(x)? 
            """ 
            #c_to_v(all_collision_tets_IDX[i]), mesh.coordinates()[c_to_v(all_collision_tets_IDX[i])] # https://fenicsproject.discourse.group/t/how-to-determine-whether-a-point-belongs-to-a-mesh-cell/6857/2
            
            #cell = fenics.Cell(mesh, all_collision_tets_IDX[i]) # https://fenicsproject.discourse.group/t/evaluate-basis-gives-strange-results/2153
            #elements = V.elements()
            #coord = elements.tabulate_dof_coordinates(cell)
            
            #detected_collisionTets_for_each_meshVertex.append(collision_tets)
            num_collisions = len(all_collision_tets_IDX) # number of collisions experienced by that vertex with any cell. at least equals to the number of neighbour-cells to which the node belongs.

            # get the precomputed value
            num_cells_associated_to_that_vertex = counter[i] # counter[i] returns the number of times node "i" is a component of a cell (e.g. counter[315]: 7 meaning node 315 enters into the composition of 7 cells)

            if num_collisions > num_cells_associated_to_that_vertex: # => there exists a "extra" tetrahedron which is not a neighbour of the Point, and into which the point collides
            # keeping only colliding nodes that belong to the boundary (surface collisions)
                #colliding_meshVertexIDX_which_are_on_boundary.append(i)
                collidingMeshNodes_detectedTetsIDXs.append([i, all_collision_tets_IDX]) # including neighbour tets
                #colliding_meshVertexCOORDS_which_is_on_boundary.append(mesh.coordinates()[i]) # indexation in the whole mesh
    
    
    # 2. For colliding mesh nodes detected and which belong to boundarymesh, get the "real" collisions
    ##################################################################################################
    real_collision_tets_IDX = [] # for each colliding surface node, return all "real" colliding face IDX (that do not include neighbour faces)
    for colliding_MeshNode_thatIsOnSurface_IDX, detected_Tets_IDX in collidingMeshNodes_detectedTetsIDXs: 
        
        #colliding_MeshNode_thatIsOnSurface_COORDS = coordinates_mesh[colliding_MeshNode_thatIsOnSurface_IDX]        
        for tet_IDX in detected_Tets_IDX:
            detectedTet_4vertices_IDX = c_to_v(tet_IDX)
            #detectedTet_4vertices_IDX = faceIDX_to_verticesIDX_mapping(detected_boundaryFace_IDX) # 4 nodes indices of the tet colliding with the point          
            
            if colliding_MeshNode_thatIsOnSurface_IDX not in detectedTet_4vertices_IDX[:]:  # if  node does not belong to the tetraedron (= real collision) (= "if pt != faces[tri,0] and pt != faces[tri,1] and pt != faces[tri,2]:"")
                real_collision_tets_IDX.append([colliding_MeshNode_thatIsOnSurface_IDX, tet_IDX]) # real_collision_tets_IDX (tet_IDX) = NNLt in T.Tallinen
    
    """
    plot_vertices_and_collidingvertex_onto_mesh3D(mesh, 
                                                  coordinates_mesh[colliding_MeshNode_thatIsOnSurface_IDX], 
                                                  (coordinates_mesh[mesh.cells()[tet_IDX][:]]), 
                                                  "colliding node and associated vertices of the tet in which it collides")"""
    
    
    # 3. For colliding mesh nodes detected, compute distance vector to the tetraedron in which it "real" collides
    #############################################################################################################
    r12vec_collidingMeshNodeOnBoundarymesh_to_boundaryFaceInRealCollisionTet = [] # r12vec
    # get vector r12 (surface node to 1st colliding cell COG)
    for colliding_MeshNode_thatIsOnSurface_IDX, real_collision_associated_Tet_IDX in real_collision_tets_IDX: # may include various tets
        
        colliding_MeshNode_thatIsOnSurface_COORDS = coordinates_mesh[colliding_MeshNode_thatIsOnSurface_IDX]
        
        realCollisionTet_4vertices_indices = c_to_v(real_collision_associated_Tet_IDX)
        
        boundarymeshVertices_of_Tet = []
        for vertex in realCollisionTet_4vertices_indices: # vertex is expressed in the whole mesh indexation
            if coordinates_mesh[vertex] in coordinates_boundarymesh: # vertex belongs to boundary
                boundarymeshVertices_of_Tet.append(vertex)
        
        num_vert = len(boundarymeshVertices_of_Tet)
        if num_vert != 0: # there is at least one node of the tetraedron that belong to the boundary (collision close to the boundary)
            closest_point_inBoundaryMesh_belonging_RealCollisionTet_COORDS = np.sum(coordinates_mesh[boundarymeshVertices_of_Tet[:]], axis=0) / num_vert # compute COG of the nodes that belong to the boundary (in the colliding tetraedron)
            
            """
            closest_point_inBoundaryMesh_belonging_RealCollisionTet_COORDS, ubt, vbt, wbt = closestPoint_in_Face( colliding_MeshNode_thatIsOnSurface_COORDS, 
                                                                                                                coordinates_mesh[boundarymeshFace_of_Tet[0]], 
                                                                                                                coordinates_mesh[boundarymeshFace_of_Tet[1]],
                                                                                                                coordinates_mesh[boundarymeshFace_of_Tet[2]])
            """
                    
            """
            realCollisionTet_4vertices_indices = mesh.cells()[real_collision_associated_Tet_IDX]
            realCollisionTet_COG_COORDS = 0.25 * (coordinates_mesh[ realCollisionTet_4vertices_indices[0] ] + 
                                                coordinates_mesh[ realCollisionTet_4vertices_indices[1] ] +
                                                coordinates_mesh[ realCollisionTet_4vertices_indices[2] ] +
                                                coordinates_mesh[ realCollisionTet_4vertices_indices[3] ]
                                                )
            """
        
            r12vec = closest_point_inBoundaryMesh_belonging_RealCollisionTet_COORDS - colliding_MeshNode_thatIsOnSurface_COORDS # 1: colliding mesh vertex on boundary, 2: closest point of colliding tetraedron (cell)
            r12vec_collidingMeshNodeOnBoundarymesh_to_boundaryFaceInRealCollisionTet.append([colliding_MeshNode_thatIsOnSurface_IDX, boundarymeshVertices_of_Tet, r12vec])
        
        else: # vertices of the colliding tetraedron do not belong to boundarymesh (deeper collision)
            pass
            # compute neighbor tetraedrons to colliding tet
            
            """
            for vertex in realCollisionTet_4vertices_indices:
                closest_point_on_boundarmesh_B_IDX, distance = bbtree_boundarymesh.compute_closest_entity(fenics.Point(*coordinates_mesh[vertex]))
                r12vec = 
                r12vec_collidingMeshNodeOnBoundarymesh_to_boundaryFaceInRealCollisionTet.append([colliding_MeshNode_thatIsOnSurface_IDX, boundarymeshVertices_of_Tet, r12vec])
            """
    """
    dist_to_boundary = [] # distance of colliding node to self boundary (r12)
    vector_to_boundary = [] # r12vec
    closest_nodeIDX_on_boundary = []
    
    #for detected_colliding_surface_node_idx_V, detected_colliding_surface_node_COORDS in enumerate(colliding_meshVertexCOORDS_which_is_on_boundary): # indexation in the whole mesh
    for colliding_meshVertexIDX_which_is_on_boundary in colliding_meshVertexIDX_which_are_on_boundary:
        
        colliding_meshVertexCOORDS_which_is_on_boundary = coordinates_mesh(colliding_meshVertexIDX_which_are_on_boundary)
        
        distances2boundary, closest_boundarynodes_IDX = treeBoundaryMesh.query(colliding_meshVertexCOORDS_which_is_on_boundary, k=10) # compute closest nodes on boundary to colliding surface Point
        # TODO: problem with k=10?
        
        for i in range(len(distances2boundary)): # =k=10
            neighbour_cells_to_colliding_surface_node = bbtree_boundarymesh.compute_entity_collisions(colliding_meshVertexIDX_which_is_on_boundary)
            cell_to_verticesIDX = mesh.topology()(mesh.topology().dim(), 0)
            all_vertices_in_neighbour_boundaryfaces = cell_to_verticesIDX(neighbour_cells_to_colliding_surface_node[:])
            
            if closest_boundarynodes_IDX[i] not in all_vertices_in_neighbour_boundaryfaces: # fenics.cells(vertex) -> all cells that contain vertex https://fenicsproject.org/qa/6799/how-to-get-the-patch-of-the-cells-from-the-vertex-index/
                closest_boundarynode_on_opposite_surface_COORDS = boundarymesh.coordinates()[ closest_boundarynodes_IDX[i] ]
                r12vec = closest_boundarynode_on_opposite_surface_COORDS - colliding_meshVertexCOORDS_which_is_on_boundary 

                dist_to_boundary.append(distances2boundary[i]) 
                vector_to_boundary.append( r12vec ) 
                closest_nodeIDX_on_boundary.append(closest_boundarynodes_IDX[i]) # closest_boundarynodes_IDX[i] = closest boundary node on opposite surface
    """          
                
    """
    debug_plot_functions.plot_vertices_onto_mesh3D(mesh, mesh_colliding_nodes_coords, "mesh self-collision nodes")
    debug_plot_functions.plot_vertex_onto_mesh3D(mesh, coordinates_mesh[343], "node 343")
    """

    # 4. Compute contact forces to correct these collisions
    #######################################################

    # Reinitialize tmp forces (Function(V))
    fcontact_global = fenics.Function(V) 

    # linear penalty force parameters. 
    epsilon = fenics.DOLFIN_EPS 
    Kc = 6 * float(K) # Kc: contact stiffness; K: bulk modulus. BrainGrowth: Kc=10*K
            
    # Compute anti-collision forces to apply onto colliding mesh nodes and associated projection node on boundary:
    for colliding_MeshNode_thatIsOnSurface_IDX, boundarymeshVertices_of_Tet_indicesOnBoundary, r12vec in r12vec_collidingMeshNodeOnBoundarymesh_to_boundaryFaceInRealCollisionTet:  # colliding_boundary_node_L: node idx, node coords
        
        f12 = unilateral_contact_modeled_by_penalization_method_XYZ(r12vec, Kc, epsilon) # should be a vector (the contact penalty force vector)
        #f12 = unilateral_contact_modeled_by_penalization_method_XYZ_smallcontactcorrected(vector_to_boundary[i], Kc)
        #f12 = contact_force_XYZ_Tallinen(r12vec, Kc, mean_mesh_spacing, epsilon)
        
        fcontact_global.vector()[ vertex2dofs_V[ colliding_MeshNode_thatIsOnSurface_IDX ] ] += f12  
        
        """
        realCollisionTet_4vertices_indices = mesh.cells()[real_collision_associated_Tet_IDX]
        fcontact_global.vector()[ vertex2dofs_V[ realCollisionTet_4vertices_indices[0] ] ] -= f12  
        fcontact_global.vector()[ vertex2dofs_V[ realCollisionTet_4vertices_indices[1] ] ] -= f12 
        fcontact_global.vector()[ vertex2dofs_V[ realCollisionTet_4vertices_indices[2] ] ] -= f12 
        fcontact_global.vector()[ vertex2dofs_V[ realCollisionTet_4vertices_indices[3] ] ] -= f12 
        """
        """      
        fcontact_global.vector()[ vertex2dofs_V[ boundarymeshFace_of_Tet_indicesOnBoundary[0] ] ] -= f12 * ubt
        fcontact_global.vector()[ vertex2dofs_V[ boundarymeshFace_of_Tet_indicesOnBoundary[1] ] ] -= f12 * vbt
        fcontact_global.vector()[ vertex2dofs_V[ boundarymeshFace_of_Tet_indicesOnBoundary[2] ] ] -= f12 * wbt
        """ 
        for i in range(len(boundarymeshVertices_of_Tet_indicesOnBoundary)):
            fcontact_global.vector()[ vertex2dofs_V[ boundarymeshVertices_of_Tet_indicesOnBoundary[i] ] ] -= f12 
                    
    return fcontact_global


def correct_collisions_3(mesh, bmesh, V,
                         vertex2dofs_V, vertexB_2_dofsV_mapping, vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping_t0, 
                         K, mean_mesh_spacing):
    """detect global collisions in the whole mesh and compute global contact forces"""

    """source: https://fenicsproject.discourse.group/t/speed-up-collision-detection-bounding-box-tree-for-point-cloud/7163/3"""

    # mesh
    # ----
    bbtree_mesh = fenics.BoundingBoxTree()
    bbtree_mesh.build(mesh, mesh.topology().dim())
    coordinates_mesh = mesh.coordinates()
    
    c_to_v = mesh.topology()(mesh.topology().dim(), 0)
    
    cells = mesh.cells() # cell1: [node1, node2, node3]; cell1: [node1, node4, node10]
    cells_flat = cells.flatten().tolist()
    counter = Counter(cells_flat) # returns all node indices 

    # bmesh
    # -----
    boundarymesh = fenics.BoundaryMesh(mesh, "exterior")
    bbtree_boundarymesh = fenics.BoundingBoxTree()
    #treeBoundaryMesh = cKDTree(boundarymesh.coordinates()) 
    bbtree_boundarymesh.build(boundarymesh, boundarymesh.topology().dim())
    coordinates_boundarymesh = boundarymesh.coordinates()
    

    # 1. Detect collisions in the mesh
    ##################################
    
    # 1.1 Detect surface node IDX (indexation of the whole mesh) that experience "real" collision with "extra" tets of the mesh
    # -------------------------------------------------------------------------------------------------------------------------
    #detected_collisionTets_for_each_meshVertex = [] # debug: [i for i in range(coordinates_mesh.shape[0]) if len(number_of_collisions_experiences_by_each_vertex[:][i]) != 0]
    #colliding_meshVertexIDX_which_are_on_boundary = []
    collidingMeshNodes_detectedTetsIDXs = []

    # Compute the number of collisions with tets for each mesh node i
    for i in range(coordinates_mesh.shape[0]):
        if mesh.coordinates()[i] in boundarymesh.coordinates():
            x = fenics.Point(*coordinates_mesh[i])
            all_collision_tets_IDX = bbtree_mesh.compute_entity_collisions(x)  # VERTEX indexation 
            """
            https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.geometry.html
            https://fenicsproject.org/olddocs/dolfin/1.3.0/python/programmers-reference/cpp/mesh/GenericBoundingBoxTree.html
            --> compute_closest_entity(x) instead of compute_entity_collisions(x)? 
            """ 
            #c_to_v(all_collision_tets_IDX[i]), mesh.coordinates()[c_to_v(all_collision_tets_IDX[i])] # https://fenicsproject.discourse.group/t/how-to-determine-whether-a-point-belongs-to-a-mesh-cell/6857/2
            
            #cell = fenics.Cell(mesh, all_collision_tets_IDX[i]) # https://fenicsproject.discourse.group/t/evaluate-basis-gives-strange-results/2153
            #elements = V.elements()
            #coord = elements.tabulate_dof_coordinates(cell)
            
            #detected_collisionTets_for_each_meshVertex.append(collision_tets)
            num_collisions = len(all_collision_tets_IDX) # number of collisions experienced by that vertex with any cell. at least equals to the number of neighbour-cells to which the node belongs.

            # get the precomputed value
            num_cells_associated_to_that_vertex = counter[i] # counter[i] returns the number of times node "i" is a component of a cell (e.g. counter[315]: 7 meaning node 315 enters into the composition of 7 cells)

            if num_collisions > num_cells_associated_to_that_vertex: # => there exists a "extra" tetrahedron which is not a neighbour of the Point, and into which the point collides
            # keeping only colliding nodes that belong to the boundary (surface collisions)
                #colliding_meshVertexIDX_which_are_on_boundary.append(i)
                collidingMeshNodes_detectedTetsIDXs.append([i, all_collision_tets_IDX]) # including neighbour tets
                #colliding_meshVertexCOORDS_which_is_on_boundary.append(mesh.coordinates()[i]) # indexation in the whole mesh
    
    
    # 2. For colliding mesh nodes detected and which belong to boundarymesh, get the "real" collisions
    ##################################################################################################
    real_collision_tets_IDX = [] # for each colliding surface node, return all "real" colliding face IDX (that do not include neighbour faces)
    for colliding_MeshNode_thatIsOnSurface_IDX, detected_Tets_IDX in collidingMeshNodes_detectedTetsIDXs: 
        
        #colliding_MeshNode_thatIsOnSurface_COORDS = coordinates_mesh[colliding_MeshNode_thatIsOnSurface_IDX]        
        for tet_IDX in detected_Tets_IDX:
            detectedTet_4vertices_IDX = c_to_v(tet_IDX)
            #detectedTet_4vertices_IDX = faceIDX_to_verticesIDX_mapping(detected_boundaryFace_IDX) # 4 nodes indices of the tet colliding with the point          
            
            if colliding_MeshNode_thatIsOnSurface_IDX not in detectedTet_4vertices_IDX[:]:  # if  node does not belong to the tetraedron (= real collision) (= "if pt != faces[tri,0] and pt != faces[tri,1] and pt != faces[tri,2]:"")
                real_collision_tets_IDX.append([colliding_MeshNode_thatIsOnSurface_IDX, tet_IDX]) # real_collision_tets_IDX (tet_IDX) = NNLt in T.Tallinen
    
    """
    plot_vertices_and_collidingvertex_onto_mesh3D(mesh, 
                                                  coordinates_mesh[colliding_MeshNode_thatIsOnSurface_IDX], 
                                                  (coordinates_mesh[mesh.cells()[tet_IDX][:]]), 
                                                  "colliding node and associated vertices of the tet in which it collides")"""
    
    
    # 3. For colliding mesh nodes detected, compute distance vector to the tetraedron in which it "real" collides
    #############################################################################################################
    r12vec_collidingMeshNodeOnBoundarymesh_to_boundaryFaceInRealCollisionTet = [] # r12vec
    # get vector r12 (surface node to 1st colliding cell COG)
    for colliding_MeshNode_thatIsOnSurface_IDX, real_collision_associated_Tet_IDX in real_collision_tets_IDX: # may include various tets
        
        colliding_MeshNode_thatIsOnSurface_COORDS = coordinates_mesh[colliding_MeshNode_thatIsOnSurface_IDX]
        
        realCollisionTet_4vertices_indices = c_to_v(real_collision_associated_Tet_IDX)
        
        boundarymeshVertices_of_Tet = []
        for vertex in realCollisionTet_4vertices_indices: # vertex is expressed in the whole mesh indexation
            if coordinates_mesh[vertex] in coordinates_boundarymesh: # vertex belongs to boundary
                boundarymeshVertices_of_Tet.append(vertex)
        
        num_vert = len(boundarymeshVertices_of_Tet)
        if num_vert != 0: # there is at least one node of the tetraedron that belong to the boundary (collision close to the boundary)
            COORDS_of_closest_point_inBoundaryMesh_to_RealCollisionTetVertices = np.sum(coordinates_mesh[boundarymeshVertices_of_Tet[:]], axis=0) / num_vert # compute COG of the nodes that belong to the boundary (in the colliding tetraedron)
            
            """
            closest_point_inBoundaryMesh_belonging_RealCollisionTet_COORDS, ubt, vbt, wbt = closestPoint_in_Face( colliding_MeshNode_thatIsOnSurface_COORDS, 
                                                                                                                coordinates_mesh[boundarymeshFace_of_Tet[0]], 
                                                                                                                coordinates_mesh[boundarymeshFace_of_Tet[1]],
                                                                                                                coordinates_mesh[boundarymeshFace_of_Tet[2]])
            """
                    
            """
            realCollisionTet_4vertices_indices = mesh.cells()[real_collision_associated_Tet_IDX]
            realCollisionTet_COG_COORDS = 0.25 * (coordinates_mesh[ realCollisionTet_4vertices_indices[0] ] + 
                                                coordinates_mesh[ realCollisionTet_4vertices_indices[1] ] +
                                                coordinates_mesh[ realCollisionTet_4vertices_indices[2] ] +
                                                coordinates_mesh[ realCollisionTet_4vertices_indices[3] ]
                                                )
            """
        
            r12vec = COORDS_of_closest_point_inBoundaryMesh_to_RealCollisionTetVertices - colliding_MeshNode_thatIsOnSurface_COORDS # 1: colliding mesh vertex on boundary, 2: closest point of colliding tetraedron (cell)
            r12vec_collidingMeshNodeOnBoundarymesh_to_boundaryFaceInRealCollisionTet.append([colliding_MeshNode_thatIsOnSurface_IDX, boundarymeshVertices_of_Tet, r12vec])
        
        else: # vertices of the colliding tetraedron do not belong to boundarymesh (deeper collision)
            IDX_of_closest_4_Vertices_on_boundary_to_RealCollidingTet = []
            COORDS_of_closest_4_Vertices_on_boundary_to_RealCollidingTet = []
            for vertex in realCollisionTet_4vertices_indices: # vertex is expressed in the whole mesh indexation
                IDX_Bref_of_closest_vertex_on_initialboundary = vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping_t0[vertex] # for each vertex of the colliding tetrahedron, compute vertex IDX (B ref) that was closest vertex at t=0 (i.e. tetrahedron vertices projection onto surface in initial reference, before contact. Otherwise, the projection can provide close boundary vertices on the colliding boundary)
                
                IDX_of_closest_4_Vertices_on_boundary_to_RealCollidingTet.append(IDX_Bref_of_closest_vertex_on_initialboundary)
                
                boundary_vertex_COORDS = bmesh.coordinates()[IDX_Bref_of_closest_vertex_on_initialboundary]
                COORDS_of_closest_4_Vertices_on_boundary_to_RealCollidingTet.append(boundary_vertex_COORDS)
            
            COORDS_of_closest_4_Vertices_on_boundary_to_RealCollidingTet = np.array(COORDS_of_closest_4_Vertices_on_boundary_to_RealCollidingTet)
            COORDS_of_closest_point_inBoundaryMesh_to_RealCollisionTetVertices = np.sum(COORDS_of_closest_4_Vertices_on_boundary_to_RealCollidingTet[:], axis=0) / 4
            
            r12vec = COORDS_of_closest_point_inBoundaryMesh_to_RealCollisionTetVertices - colliding_MeshNode_thatIsOnSurface_COORDS # 1: colliding mesh vertex on boundary, 2: closest point of colliding tetraedron (cell)
            r12vec_collidingMeshNodeOnBoundarymesh_to_boundaryFaceInRealCollisionTet.append([colliding_MeshNode_thatIsOnSurface_IDX, IDX_of_closest_4_Vertices_on_boundary_to_RealCollidingTet, r12vec])
                
            """
            for vertex in realCollisionTet_4vertices_indices:
                closest_point_on_boundarmesh_B_IDX, distance = bbtree_boundarymesh.compute_closest_entity(fenics.Point(*coordinates_mesh[vertex]))
                r12vec = 
                r12vec_collidingMeshNodeOnBoundarymesh_to_boundaryFaceInRealCollisionTet.append([colliding_MeshNode_thatIsOnSurface_IDX, boundarymeshVertices_of_Tet, r12vec])
            """
    """
    dist_to_boundary = [] # distance of colliding node to self boundary (r12)
    vector_to_boundary = [] # r12vec
    closest_nodeIDX_on_boundary = []
    
    #for detected_colliding_surface_node_idx_V, detected_colliding_surface_node_COORDS in enumerate(colliding_meshVertexCOORDS_which_is_on_boundary): # indexation in the whole mesh
    for colliding_meshVertexIDX_which_is_on_boundary in colliding_meshVertexIDX_which_are_on_boundary:
        
        colliding_meshVertexCOORDS_which_is_on_boundary = coordinates_mesh(colliding_meshVertexIDX_which_are_on_boundary)
        
        distances2boundary, closest_boundarynodes_IDX = treeBoundaryMesh.query(colliding_meshVertexCOORDS_which_is_on_boundary, k=10) # compute closest nodes on boundary to colliding surface Point
        # TODO: problem with k=10?
        
        for i in range(len(distances2boundary)): # =k=10
            neighbour_cells_to_colliding_surface_node = bbtree_boundarymesh.compute_entity_collisions(colliding_meshVertexIDX_which_is_on_boundary)
            cell_to_verticesIDX = mesh.topology()(mesh.topology().dim(), 0)
            all_vertices_in_neighbour_boundaryfaces = cell_to_verticesIDX(neighbour_cells_to_colliding_surface_node[:])
            
            if closest_boundarynodes_IDX[i] not in all_vertices_in_neighbour_boundaryfaces: # fenics.cells(vertex) -> all cells that contain vertex https://fenicsproject.org/qa/6799/how-to-get-the-patch-of-the-cells-from-the-vertex-index/
                closest_boundarynode_on_opposite_surface_COORDS = boundarymesh.coordinates()[ closest_boundarynodes_IDX[i] ]
                r12vec = closest_boundarynode_on_opposite_surface_COORDS - colliding_meshVertexCOORDS_which_is_on_boundary 

                dist_to_boundary.append(distances2boundary[i]) 
                vector_to_boundary.append( r12vec ) 
                closest_nodeIDX_on_boundary.append(closest_boundarynodes_IDX[i]) # closest_boundarynodes_IDX[i] = closest boundary node on opposite surface
    """          
                
    """
    debug_plot_functions.plot_vertices_onto_mesh3D(mesh, mesh_colliding_nodes_coords, "mesh self-collision nodes")
    debug_plot_functions.plot_vertex_onto_mesh3D(mesh, coordinates_mesh[343], "node 343")
    """

    # 4. Compute contact forces to correct these collisions
    #######################################################

    # Reinitialize tmp forces (Function(V))
    fcontact_global = fenics.Function(V) 

    # linear penalty force parameters. 
    epsilon = fenics.DOLFIN_EPS 
    Kc = 6 * float(K) # Kc: contact stiffness; K: bulk modulus. BrainGrowth: Kc=10*K
            
    # Compute anti-collision forces to apply onto colliding mesh nodes and associated projection node on boundary:
    for colliding_MeshNode_thatIsOnSurface_IDX, boundarymeshVertices_associated_to_CollidingTet, r12vec in r12vec_collidingMeshNodeOnBoundarymesh_to_boundaryFaceInRealCollisionTet:  # colliding_boundary_node_L: node idx, node coords
        
        f12 = unilateral_contact_modeled_by_penalization_method_XYZ(r12vec, Kc, epsilon) # should be a vector (the contact penalty force vector)
        #f12 = unilateral_contact_modeled_by_penalization_method_XYZ_smallcontactcorrected(vector_to_boundary[i], Kc)
        #f12 = contact_force_XYZ_Tallinen(r12vec, Kc, mean_mesh_spacing, epsilon)
        
        fcontact_global.vector()[ vertex2dofs_V[ colliding_MeshNode_thatIsOnSurface_IDX ] ] += f12  
        
        """
        realCollisionTet_4vertices_indices = mesh.cells()[real_collision_associated_Tet_IDX]
        fcontact_global.vector()[ vertex2dofs_V[ realCollisionTet_4vertices_indices[0] ] ] -= f12  
        fcontact_global.vector()[ vertex2dofs_V[ realCollisionTet_4vertices_indices[1] ] ] -= f12 
        fcontact_global.vector()[ vertex2dofs_V[ realCollisionTet_4vertices_indices[2] ] ] -= f12 
        fcontact_global.vector()[ vertex2dofs_V[ realCollisionTet_4vertices_indices[3] ] ] -= f12 
        """
        """      
        fcontact_global.vector()[ vertex2dofs_V[ boundarymeshFace_of_Tet_indicesOnBoundary[0] ] ] -= f12 * ubt
        fcontact_global.vector()[ vertex2dofs_V[ boundarymeshFace_of_Tet_indicesOnBoundary[1] ] ] -= f12 * vbt
        fcontact_global.vector()[ vertex2dofs_V[ boundarymeshFace_of_Tet_indicesOnBoundary[2] ] ] -= f12 * wbt
        """ 
        for i in range(len(boundarymeshVertices_associated_to_CollidingTet)):
            fcontact_global.vector()[ vertex2dofs_V[ boundarymeshVertices_associated_to_CollidingTet[i] ] ] -= f12 
                    
    return fcontact_global

#############################
# Functions from T.Tallinen (2016)
#############################

# Find the closest point of triangle abc to point p, if not, p projection through the barycenter inside the triangle
def closestPoint_on_ProximityTriangle(p, a, b, c):
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        u = 1.0
        v = 0.0
        w = 0.0
        return a, u, v, w

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        u = 0.0
        v = 1.0
        w = 0.0
        return b, u, v, w

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        u = 1.0 - v
        w = 0.0
        return a + ab * v, u, v, w

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        u = 0.0
        v = 0.0
        w = 1.0
        return c, u, v, w

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        u = 1.0 - w
        v = 0.0
        return a + ac * w, u, v, w

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        u = 0.0
        v = 1.0 - w
        return b + (c - b) * w, u, v, w

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    u = 1.0 - v - w

    return a + ab * v + ac * w, u, v, w 
    # a + ab * v + ac * w: coordinates of the 1st-cooliding-face-projected point 
    # u: ponderation to apply to the repulsive force vector (face COG) to get the force to apply to node 0 of the colliding face
    # v: idem for node 1 of the colliding face
    # w: idem for node 2 of the colliding face

def norm_dim_3(a):
    b = np.zeros(len(a), dtype=np.float64)
    b[:] = np.sqrt(a[:, 0] * a[:, 0] + a[:, 1] * a[:, 1] + a[:, 2] * a[:, 2])

    return b

def cross_dim_2(a, b):
    c = np.zeros(3)
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c

def createNNLtriangle(NNLt, average_mesh_spacing,
                      coordinates, faces, n_surface_nodes, n_faces):
    """Generates point-triangle proximity lists (NNLt) using the linked cell algorithm"""
  
     # NNLt parameters
    bounding_box = 3.2
    cell_width = 8 * average_mesh_spacing
    prox_skin = 0.6 * average_mesh_spacing
  
     # generate cubic bounding box and mark each voxel of the bounding box that contain a boundary face with its boundary face index
     # -----------------------------------------------------------------------------------------------------------------------------
    mx = max(1, int(bounding_box/cell_width))  # = 40 cells, bounding_box=3.2, cell_width=0.08
    head = [-1]*mx*mx*mx # mx*mx*mx cells nomber, size mx*mx*mx list with all values are -1, 40*40*40 = 64000
    lists = [0]*n_faces
    ub = vb = wb = 0.0  # Barycentric coordinates of triangles
    for i in range(n_faces):  # Divide triangle faces into cells, i index of face
        cog = (coordinates[faces[i,0]] + coordinates[faces[i,1]] + coordinates[faces[i,2]])/3.0
        xa = int((cog[0]+0.5*bounding_box)/bounding_box*mx) # from brain mesh indexation (coordinates) to the bounding box indexation (voxel)
        ya = int((cog[1]+0.5*bounding_box)/bounding_box*mx)
        za = int((cog[2]+0.5*bounding_box)/bounding_box*mx)
        """tmp = mx*mx*za + mx*ya + xa""" # Parse all voxels till reaching the one containing the concerned face
        # print ('cog.x is ' + str(cog[0]) + ' cog y is ' + str(cog[1]) + ' cog.z is ' + str(cog[2]) + ' xa is ' + str(xa) + ' ya is ' + str(ya) + ' za is ' + str(za) + ' tmp is ' + str(tmp))
        lists[i] = head[mx*mx*za + mx*ya + xa]
        head[mx*mx*za + mx*ya + xa] = i # allocate the index of the face "detected" to the position number of the bounding box voxel

    for i in range(n_surface_nodes):   # Search cells around each surface point and build proximity list
        pt = i # pt = nodal_idx[i]
        NNLt[i][:] = []
        xa = int((coordinates[pt,0]+0.5*bounding_box)/bounding_box*mx)
        ya = int((coordinates[pt,1]+0.5*bounding_box)/bounding_box*mx)
        za = int((coordinates[pt,2]+0.5*bounding_box)/bounding_box*mx)

        for xi, yi, zi in zip(range(max(0,xa-1), min(mx-1, xa+1)+1), range(max(0,ya-1), min(mx-1, ya+1)+1), range(max(0,za-1), min(mx-1, za+1)+1)): # Browse head list
            tri = head[mx*mx*zi + mx*yi + xi]
            while tri != -1:
                if pt != faces[tri,0] and pt != faces[tri,1] and pt != faces[tri,2]:
                   pc, ubt, vbt, wbt = closestPoint_on_ProximityTriangle(coordinates[pt], coordinates[faces[tri,0]], coordinates[faces[tri,1]], coordinates[faces[tri,2]])
                   if np.linalg.norm(pc - coordinates[pt]) < prox_skin: # if closest point on the proximity boundarymesh triangle is too close to the concerned node, then consider the face likely to collide with the node
                       NNLt[i].append(tri)
                tri = lists[tri]

    return NNLt

def anticipate_collisions_Tallinen(NNLt, 
                                   V, vertexB_2_dofsV_mapping,
                                   S, vertexBoundaryMesh_2_dofScalarFunctionSpaceWholeMesh_mapping,
                                   K, average_mesh_spacing, grNoGrowthZones,
                                   mesh, bmesh, coordinates_old):
    """
    Calculate contact forces if distance between two nodes is below a certain threshold
    """
    # contact process parameters
    prox_skin = 0.6 * average_mesh_spacing
    repuls_skin = 0.2 * average_mesh_spacing
    contact_stiffness = 100 * K.values()[0]

    #coordinates = mesh.coordinates()
    coordinates = bmesh.coordinates()
    faces = bmesh.cells()
    n_faces = len(faces)
    n_surface_nodes = bmesh.num_vertices()

    # intialize FEniCS contact Function
    Ft = fenics.Function(V) 
    
    maxDist = 0.0
    ub = vb = wb = 0.0  # Barycentric coordinates of triangles
    maxDist = max(norm_dim_3(coordinates[:] - coordinates_old[:])) 
    if maxDist > 0.5*(prox_skin-repuls_skin): 
        NNLt = createNNLtriangle(NNLt, average_mesh_spacing, coordinates, faces, n_surface_nodes, n_faces) # Generates point-triangle proximity lists (NNLt[n_surface_nodes]) using the linked cell algorithm. The proximity triangles to surface node 'i' are all triangles around 'i' that do not contain the surface node (these are no neighbor triangles). 
        for i in range(n_surface_nodes):
            coordinates_old[i] = coordinates[i]
  
    for i in range(n_surface_nodes): # Loop through surface points
        for tp in range(len(NNLt[i])): # Parse all proximity triangles to surface node 'i'
            #pt = nodal_idx[i] # A surface point index
            pt = i
            tri = NNLt[i][tp] # proximity triangle (to surface node 'i') of index 'tp'
            pc, ubt, vbt, wbt = closestPoint_on_ProximityTriangle(coordinates[pt], coordinates[faces[tri,0]], coordinates[faces[tri,1]], coordinates[faces[tri,2]])  # Find the nearest point to Barycentric
            cc = pc - coordinates[pt] # moinus to all nodes
            # closestPointTriangle returns the closest point of triangle abc to point p (returns a or b or c, if not, pt projection through the barycenter inside the triangle)
            
            rc = np.linalg.norm(cc)   # Distance between the closest point in the triangle to the point, sqrt(x*x+y*y+z*z)
            if rc < repuls_skin and grNoGrowthZones.vector()[vertexBoundaryMesh_2_dofScalarFunctionSpaceWholeMesh_mapping[pt]] + grNoGrowthZones.vector()[vertexBoundaryMesh_2_dofScalarFunctionSpaceWholeMesh_mapping[faces[tri][0]]] > 0.0:  # Calculate contact force if within the contact range
                cc *= 1.0/rc
                fn = cc*(rc-repuls_skin)/repuls_skin*contact_stiffness*average_mesh_spacing*average_mesh_spacing # kc = 10.0*K Contact stiffness
                
                #Ntri = cross_dim_2(coordinates[faces[tri,1]] - coordinates[faces[tri,0]], coordinates[faces[tri,2]] - coordinates[faces[tri,0]]) # Triangle normal
                #Ntri *= 1.0/np.linalg.norm(Ntri)
                
                #if np.dot(fn, Ntri) < 0.0:
                #    fn -= Ntri*np.dot(fn, Ntri)*2.0

                """
                Ft[pt] += fn

                Ft[faces[tri,0]] -= fn*ubt
                Ft[faces[tri,1]] -= fn*vbt
                Ft[faces[tri,2]] -= fn*wbt
                """
                """
                Ft.vector()[ vertex2dofs_V[ pt ] ] += fn 

                Ft.vector()[ vertex2dofs_V[ faces[tri,0] ] ] -= fn * ubt
                Ft.vector()[ vertex2dofs_V[ faces[tri,1] ] ] -= fn * vbt
                Ft.vector()[ vertex2dofs_V[ faces[tri,2] ] ] -= fn * wbt
                """
                Ft.vector()[ vertexB_2_dofsV_mapping[ pt ] ] += fn 

                Ft.vector()[ vertexB_2_dofsV_mapping[ faces[tri,0] ] ] -= fn * ubt
                Ft.vector()[ vertexB_2_dofsV_mapping[ faces[tri,1] ] ] -= fn * vbt
                Ft.vector()[ vertexB_2_dofsV_mapping[ faces[tri,2] ] ] -= fn * wbt
        
    return Ft, NNLt