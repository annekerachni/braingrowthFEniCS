import fenics
import numpy as np
from collections import Counter
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.dirname(sys.path[0]))

from braingrowth_3D.phenomenological_dynamic_3D.FEM_biomechanical_model import projection, mappings, growth

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

#################################
# Penalty method force function # 
#################################

def penalty_force(r12vec, penalty_coefficient, normal_vector_at_Boundary1, epsilon):

    r12 = np.linalg.norm(r12vec) 

    if (r12 <= epsilon): # does not taking into account "small" penetration to avoid correcting mesh close contact, that is acceptable.
        return np.array([0.0, 0.0, 0.0])
    else:
        #r12hat = r12vec/np.linalg.norm(r12vec)
        return penalty_coefficient * np.dot(r12vec, normal_vector_at_Boundary1) * normal_vector_at_Boundary1 # penalty-force formula
        # since force is computed when penetration occurs, np.dot(r12vec, normal_vector_at_Boundary1) < 0, so the force is in the opposite direction of normal nM at "master" node
    

def penalty_force_smallContactCorrected(r12vec, penalty_coefficient, normal_vector_at_Boundary1):
    #r12 = np.linalg.norm(r12vec) 
    #r12hat = r12vec/np.linalg.norm(r12vec)
    return penalty_coefficient * np.dot(r12vec, normal_vector_at_Boundary1) * normal_vector_at_Boundary1 # penalty-force formula
    
###########################################
# Other functions from [T.Tallinen, 2016] #
###########################################

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
        return a, u, v, w # A is the closest point to P

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        u = 0.0
        v = 1.0
        w = 0.0
        return b, u, v, w # B is the closest point to P

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

    # TODO: equivalent to "else"?
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

#######################################
# CONTACT MECHANICS ALGO SELF-CONTACT #
#######################################

from numba import prange
def contact_mechanics_algo_SelfContact_ClosestNodeFaceTallinen_PENALTY(NNLt, u_avg, v_test,
                                                                        mesh, V, vertexBoundaryMesh_2_dofsVWholeMesh_mapping, # vertex index in Boundary referential --> vector dofs in Whole Mesh referential
                                                                        S, vertexBoundaryMesh_2_dofSWholeMesh_mapping, # vertex index in Boundary referential --> scalar dof in Whole Mesh referential
                                                                        K, muCortex,
                                                                        average_mesh_spacing, bmesh, coordinates_old):

    """
    Penalizing self-contact (unknown master / slave) within FEM problem resolution 
    --> "penalty method" to optimize the resolution
    --> point-to-surface detection algorithm
    """

    u_unknown_alphaGeneralizedMethod = fenics.Function(V)
    projection.local_project(u_avg, V, u_unknown_alphaGeneralizedMethod)

    # contact_stiffness = penalty coefficient
    """epsilon_n = 100 * K.values()[0]""" # T.Tallinen
    
    E = 9*muCortex.values()[0]*K.values()[0]/(3*K.values()[0] + muCortex.values()[0]) # Young modulus
    h = mesh.hmin() # element size
    epsilon_n = E/h 

    #coordinates = mesh.coordinates()
    coordinates = bmesh.coordinates()
    faces = bmesh.cells()
    n_faces = len(faces)
    n_surface_nodes = bmesh.num_vertices()

    # intialize FEniCS contact Function
    #gn = fenics.Function(S, name="NormalGap")
    #contact_res_form_integrand_S = fenics.Function(S) # variation of contact energy 
    #contact_res_form_integrand_V = fenics.Function(V)
    f_penalty_self_contact_V = fenics.Function(V)
    
    # Generates point-triangle proximity lists (NNLt[n_surface_nodes]) using the linked cell algorithm. The proximity triangles to surface node 'i' are all triangles around 'i' that do not contain the surface node (these are no neighbor triangles). 
    """NNLt = createNNLtriangle(NNLt, average_mesh_spacing, coordinates, faces, n_surface_nodes, n_faces)"""
    maxDist = 0.0
    maxDist = max(norm_dim_3(coordinates[:] - coordinates_old[:])) 

    prox_skin = 0.6 * average_mesh_spacing
    repuls_skin = 0.2 * average_mesh_spacing
    
    if maxDist > 0.5 * (prox_skin - repuls_skin): 
        NNLt = createNNLtriangle(NNLt, average_mesh_spacing, coordinates, faces, n_surface_nodes, n_faces) # Generates point-triangle proximity lists (NNLt[n_surface_nodes]) using the linked cell algorithm. The proximity triangles to surface node 'i' are all triangles around 'i' that do not contain the surface node (these are no neighbor triangles). 
        for i in prange(n_surface_nodes):
            coordinates_old[i] = coordinates[i] 

    # Penalize contact in case of penetration (normal gap < 0)
    for i_point_slave in prange(n_surface_nodes): # parse all "slave" surface nodes
        
        for i_NNLt in prange(len(NNLt[i_point_slave])): # For each slave surface node 'i', parse all proximity triangles ("master" surface)

            rS = coordinates[i_point_slave]

            # compute closest node "pc" on the proximity "master" triangle:
            i_proximity_face_master = NNLt[i_point_slave][i_NNLt] # proximity triangle (to surface node 'i_point_slave') of index 'i_face_master'
            
            # absolute gap between "slave" node and its closest node on "master" surface (vector) 
            """g_MS = rS - p_rS_on_master""" # = rS - rM (absolute gap_master_to_slave)

            xS_tplus1 = rS + u_unknown_alphaGeneralizedMethod.vector()[vertexBoundaryMesh_2_dofsVWholeMesh_mapping[i_point_slave]]

            # compute closest point on master face (i.e. proximity face within NNLt list)
            p_rS_on_master_tplus1, ubt, vbt, wbt = closestPoint_on_ProximityTriangle(xS_tplus1, 
                                                                                     coordinates[faces[i_proximity_face_master, 0]] + u_unknown_alphaGeneralizedMethod.vector()[vertexBoundaryMesh_2_dofsVWholeMesh_mapping[faces[i_proximity_face_master, 0]]], 
                                                                                     coordinates[faces[i_proximity_face_master, 1]] + u_unknown_alphaGeneralizedMethod.vector()[vertexBoundaryMesh_2_dofsVWholeMesh_mapping[faces[i_proximity_face_master, 1]]], 
                                                                                     coordinates[faces[i_proximity_face_master, 2]]) + u_unknown_alphaGeneralizedMethod.vector()[vertexBoundaryMesh_2_dofsVWholeMesh_mapping[faces[i_proximity_face_master, 2]]] # Find the nearest point to Barycentric
            
            g_MS = xS_tplus1 - p_rS_on_master_tplus1 # = rS_avg_{t+1} - rM_avg_{t+1} (absolute gap_master_to_slave)

            # compute normal of the proximity "master" face
            """
            Ntri = cross_dim_2(coordinates[faces[i_proximity_face_master, 1]] - coordinates[faces[i_proximity_face_master, 0]], 
                               coordinates[faces[i_proximity_face_master, 2]] - coordinates[faces[i_proximity_face_master, 0]]) # Triangle normal
            """
            Ntri = cross_dim_2(coordinates[faces[i_proximity_face_master, 1]] + u_unknown_alphaGeneralizedMethod.vector()[vertexBoundaryMesh_2_dofsVWholeMesh_mapping[faces[i_proximity_face_master, 1]]] - (coordinates[faces[i_proximity_face_master, 0]] + u_unknown_alphaGeneralizedMethod.vector()[vertexBoundaryMesh_2_dofsVWholeMesh_mapping[faces[i_proximity_face_master, 0]]]), 
                               coordinates[faces[i_proximity_face_master, 2]] + u_unknown_alphaGeneralizedMethod.vector()[vertexBoundaryMesh_2_dofsVWholeMesh_mapping[faces[i_proximity_face_master, 2]]] - (coordinates[faces[i_proximity_face_master, 0]] + u_unknown_alphaGeneralizedMethod.vector()[vertexBoundaryMesh_2_dofsVWholeMesh_mapping[faces[i_proximity_face_master, 0]]])) # Triangle normal
            
            Ntri *= 1.0/np.linalg.norm(Ntri)
            
            # compute normal gap (scalar)
            gn = np.dot(g_MS, Ntri) 

            if gn < 0. - fenics.DOLFIN_EPS: # penetration (Signorini conditions for non penetration)
                
                # ****************** Notes ************
                # compute δgn (master-slave) (scalar)
                # in case only slave considered (we have only dr):
                """dgn = np.dot(xS_tplus1, Ntri)""" # (δrS).nM = uS.nM ?  

                # in case both slave & projected master considered:
                """drS_drM =  u.vector()[vertexBoundaryMesh_2_dofsVWholeMesh_mapping[i_point_slave]] - u.vector()[p_rS_on_master]
                dgn = np.dot(drS_drM, Ntri)"""
                # *************************************

                # dgn = (δrS - δrM).nM = (vS_test - vM_test).Ntri # --> considered twice since all slave nodes are parsed 
                # -------------------------------------------------------------------------------------------------------
                # integration only on the "slave" contact boundary Gamma_c^2 (See equations)
                f_penalty_self_contact_V.vector()[ vertexBoundaryMesh_2_dofsVWholeMesh_mapping[i_point_slave] ] += epsilon_n * (-gn) * Ntri # Need to be mulitplied by v_test[xS_tplus1] * ds(101) 

                # integration only on the "master" contact boundary Gamma_c^1 (See equations)
                f_penalty_self_contact_V.vector()[ vertexBoundaryMesh_2_dofsVWholeMesh_mapping[ faces[i_proximity_face_master, 0] ] ] -= epsilon_n * (-gn) * Ntri * ubt  # Need to be mulitplied by + v_test[p_rS_on_master_tplus1] * ds(101)
                f_penalty_self_contact_V.vector()[ vertexBoundaryMesh_2_dofsVWholeMesh_mapping[ faces[i_proximity_face_master, 1] ] ] -= epsilon_n * (-gn) * Ntri * vbt  # Need to be mulitplied by + v_test[p_rS_on_master_tplus1] * ds(101)
                f_penalty_self_contact_V.vector()[ vertexBoundaryMesh_2_dofsVWholeMesh_mapping[ faces[i_proximity_face_master, 2] ] ] -= epsilon_n * (-gn) * Ntri * wbt  # Need to be mulitplied by + v_test[p_rS_on_master_tplus1] * ds(101)

    return f_penalty_self_contact_V, NNLt # contact_res_form_integrand_V


def contact_mechanics_algo_SelfContact_ClosestNodeFaceTallinen_PENALTY_2(penalty_coefficient, 
                                                                        mesh, bmesh, surface_coordinates_old,
                                                                        K, muCortex, average_mesh_spacing,
                                                                        NNLt, 
                                                                        S, V, vertexBoundaryMesh_2_dofsVWholeMesh_mapping,
                                                                        vertexB_2_dofS_mapping, vertexB_2_dofsV_mapping, # vertexBoundaryMesh_2_dofSWholeMesh_mapping, vertexBoundaryMesh_2_dofsVWholeMesh_mapping
                                                                        u_avg, v_test, ds,
                                                                        BoundaryMesh_Nt):
    """
    Contact mechanics:
    - close nodes detection: "slave" node to "master" triangle face element (through its closest node)
    - normal gap gn: gn = MS_vec.nM (normal at the "master" face)
    - contact penalization (penalty method): contact_integrand = epsilon_n * (-gn) *(vS - vM) * ds, where vS and vM local test functions
    """
    #coordinates = mesh.coordinates()
    surface_coordinates = bmesh.coordinates()
    faces = bmesh.cells()
    n_faces = len(faces)
    n_surface_nodes = bmesh.num_vertices()

    u_unknown_alphaGeneralizedMethod = fenics.Function(V)
    projection.local_project(u_avg, V, u_unknown_alphaGeneralizedMethod)

    # contact process parameters
    ############################
    prox_skin = 0.6 * average_mesh_spacing # ~ 0.7 * average_mesh_spacing
    repuls_skin = 0.2 * average_mesh_spacing # should be h = 1/3 * a ~ 0.3 * average_mesh_spacing. See [T.Tallinen et al. 2014, Gyrification from constrained cortical expansion]

    #young_modulus = fenics.Function(S)
    # young_modulus = 9 * K.values()[0] * mu.vector()[:] / (3*K.values()[0] + mu.vector()[:])
    young_modulus = 9 * K.values()[0] * muCortex.values()[0] / (3*K.values()[0] + muCortex.values()[0]) # https://en.wikipedia.org/wiki/Lam%C3%A9_parameters
    h = mesh.hmin() # element size # mesh.hmin(); CellDiameter(mesh) https://fenicsproject.discourse.group/t/penalty-formulation-in-fenics/6111

    #epsilon_n = fenics.Function(S)
    #epsilon_n.vector()[:] = young_modulus.vector()[:] / h # epsilon_n: penalty coefficient.  Take epsilon_n = E/h as prior estimation
    epsilon_n = young_modulus/h # young_modulus/h (2240 instead of 50 for BrainGrowth)

    # if surface node underwent sufficient move, consider collisions
    ################################################################
    #max_surface_displacement = 0.0
    max_surface_displacement = max(norm_dim_3(surface_coordinates[:] - surface_coordinates_old[:])) 

    if max_surface_displacement > 0.5 * (prox_skin - repuls_skin): # if surface node underwent sufficient move, consider collisions
        NNLt = createNNLtriangle(NNLt, average_mesh_spacing, surface_coordinates, faces, n_surface_nodes, n_faces) # Generates point-triangle proximity lists (NNLt[n_surface_nodes]) using the linked cell algorithm. The proximity triangles to surface node 'i' are all triangles around 'i' that do not contain the surface node (these are no neighbor triangles). 
        for i in range(n_surface_nodes):
            surface_coordinates_old[i] = surface_coordinates[i]
    
    # For each surface node, compute the contact mechanics FEM integrand to add to penalize self-penetration 
    ########################################################################################################
    
    ForceV_penalty_integrand = fenics.Function(V) # intialize FEniCS contact Function
    ForceS_penalty_integrand = fenics.Function(S) 

    for i in range(n_surface_nodes): # Loop through surface points
        for tri_prox_IDX in range(len(NNLt[i])): # Parse all proximity triangles to surface node 'i'
            #pt = nodal_idx[i] # A surface point index --> pt = i
            xS = surface_coordinates[i] 

            tri = NNLt[i][tri_prox_IDX] # proximity triangle (to surface node 'i') of index 'tri_prox_IDX'

            xM, ubt, vbt, wbt = closestPoint_on_ProximityTriangle(xS, 
                                                                  surface_coordinates[faces[tri,0]], 
                                                                  surface_coordinates[faces[tri,1]], 
                                                                  surface_coordinates[faces[tri,2]])  # Find the nearest point to Barycentric

            # gap (vector)
            # Find u such that: # See Hertzian penalty method FEniCS
            DOFsV_tri0, DOFsV_tri1, DOFsV_tri2 = vertexB_2_dofsV_mapping[faces[tri, 0]], vertexB_2_dofsV_mapping[faces[tri, 1]], vertexB_2_dofsV_mapping[faces[tri, 2]] # get local displacement (v_test) at xM
            u_approximated_at_xM = ubt * u_unknown_alphaGeneralizedMethod.vector()[DOFsV_tri0] + vbt * u_unknown_alphaGeneralizedMethod.vector()[DOFsV_tri1] + wbt * u_unknown_alphaGeneralizedMethod.vector()[DOFsV_tri2] # approximate the local displacement function u at xM using v_test at the 3 points of the triangle. 
            
            #MS_vec = fenics.Function(V)
            #MS_vec = u.vector()[vertexB_2_dofsV_mapping[i]] - u_approximated_at_xM # uS - uM
            xS_avg = xS + u_unknown_alphaGeneralizedMethod.vector()[vertexB_2_dofsV_mapping[i]] 
            xM_avg = xM + u_approximated_at_xM
            MS_vec = xS_avg - xM_avg # as contact mechanics residual form takes into account the unknown (future) state of the mesh, it is necessary to express the future gap between 2 nodes. In reality, it is expressed using alpha-genralized method (so at the average time and not t+1 --> u_avg and not u)

            # normal of the "master" face (nM) --> vector
            """
            n_tri = cross_dim_2(surface_coordinates[faces[tri, 1]] - surface_coordinates[faces[tri, 0]], 
                                surface_coordinates[faces[tri, 2]] - surface_coordinates[faces[tri, 0]]) # triangle normal
            """
    
            n_tri = cross_dim_2(surface_coordinates[faces[tri, 1]] + u_unknown_alphaGeneralizedMethod.vector()[vertexBoundaryMesh_2_dofsVWholeMesh_mapping[faces[tri, 1]]] - (surface_coordinates[faces[tri, 0]] + u_unknown_alphaGeneralizedMethod.vector()[vertexBoundaryMesh_2_dofsVWholeMesh_mapping[faces[tri, 0]]]), 
                               surface_coordinates[faces[tri, 2]] + u_unknown_alphaGeneralizedMethod.vector()[vertexBoundaryMesh_2_dofsVWholeMesh_mapping[faces[tri, 2]]] - (surface_coordinates[faces[tri, 0]] + u_unknown_alphaGeneralizedMethod.vector()[vertexBoundaryMesh_2_dofsVWholeMesh_mapping[faces[tri, 0]]])) # Triangle normal
            
            
            n_tri *= 1.0/np.linalg.norm(n_tri)

            # compute normal gap (integer)
            gn = np.dot(MS_vec, n_tri) # gn = MS_vec.nM

            ###
            ###
            """
            #MS_vec = fenics.Function(V)
            MS_vec_V = fenics.Function(V)
            MS_vec_V.vector()[DOFsV_tri0] = u.vector()[vertexB_2_dofsV_mapping[i]] - u_approximated_at_xM # uS - uM

            # normal of the "master" face (nM) --> fenics.Function(V)
            nM_V = fenics.Function(V)
            nM_V.vector()[DOFsV_tri0] = 1/3 * (BoundaryMesh_Nt.vector()[DOFsV_tri0] + BoundaryMesh_Nt.vector()[DOFsV_tri1] + BoundaryMesh_Nt.vector()[DOFsV_tri2] )

            # compute normal gap (integer) fenics.Function(S)
            gn = fenics.dot(MS_vec_V, nM_V) # gn = MS_vec.nM
            integrand_V = epsilon_n * fenics.dot( -gn, fenics.dot(nM_V, v_test))
            """

            if gn < 0.0 + fenics.DOLFIN_EPS: #- fenics.DOLFIN_EPS: # penetration (Signorini conditions for non penetration)

                #DOFsV_tri0, DOFsV_tri1, DOFsV_tri2 = vertexB_2_dofsV_mapping[faces[tri, 0]], vertexB_2_dofsV_mapping[faces[tri, 1]], vertexB_2_dofsV_mapping[faces[tri, 2]] # get local displacement (v_test) at xM
                #v_test_approximated_at_xM = 1/3 * (v_test.vector()[DOFsV_tri0] + v_test.vector()[DOFsV_tri1] + v_test.vector()[DOFsV_tri2] ) # approximate the local test displacement function v_test at xM using v_test at the 3 points of the triangle. 
                
                # compute dgn = nM.(δrS - δrM) = nM.(δuS - δuM) (= nM.(vS - vM)? )
                """dgn = fenics.dot(n_tri, v_test.vector()[vertexB_2_dofsV_mapping[i]] - v_test_approximated_at_xM)"""

                #contact_penalty_integrand = epsilon_n * (-gn) * ( v_test.vector()[coordinates_2_DOFsV[xS]] - v_test.vector()[coordinates_2_DOFsV[xM]] )
                """contact_penalty_FEniCS_FEM_integrand = epsilon_n * (-gn)""" # scalar. rather use np.dot?

                ForceV_penalty_integrand.vector()[ vertexB_2_dofsV_mapping[ i ] ] += epsilon_n * (-gn) * n_tri  # add penalization on the "slave" contact boundary Gamma_c^2 also?

                # integration only on the "master" contact boundary Gamma_c^1 (See equations)
                """
                ForceS_penalty_integrand.vector()[ vertexB_2_dofS_mapping[ faces[tri, 0] ] ] += epsilon_n * fenics.dot( -gn, np.dot(n_tri, v_test)) * ubt
                ForceS_penalty_integrand.vector()[ vertexB_2_dofS_mapping[ faces[tri, 1] ] ] += epsilon_n * fenics.dot( -gn, np.dot(n_tri, v_test)) * vbt
                ForceS_penalty_integrand.vector()[ vertexB_2_dofS_mapping[ faces[tri, 2] ] ] += epsilon_n * fenics.dot( -gn, np.dot(n_tri, v_test)) * wbt
                """

                """
                ForceV_penalty_integrand.vector()[ vertexB_2_dofsV_mapping[ faces[tri, 0] ] ] += epsilon_n * (-MS_vec) * ubt
                ForceV_penalty_integrand.vector()[ vertexB_2_dofsV_mapping[ faces[tri, 1] ] ] += epsilon_n * (-MS_vec) * vbt
                ForceV_penalty_integrand.vector()[ vertexB_2_dofsV_mapping[ faces[tri, 2] ] ] += epsilon_n * (-MS_vec) * wbt
                """

                # REF version is the following:
                ForceV_penalty_integrand.vector()[ vertexB_2_dofsV_mapping[ faces[tri, 0] ] ] -= epsilon_n * (-gn) * n_tri * ubt 
                ForceV_penalty_integrand.vector()[ vertexB_2_dofsV_mapping[ faces[tri, 1] ] ] -= epsilon_n * (-gn) * n_tri * vbt
                ForceV_penalty_integrand.vector()[ vertexB_2_dofsV_mapping[ faces[tri, 2] ] ] -= epsilon_n * (-gn) * n_tri * wbt

                #epsilon_n * fenics.dot( -gn, np.dot(n_tri, v_test))
        
    return ForceV_penalty_integrand

#######
#######

def contact_mechanics_algo_SelfContact_CollidingClosestTetNodeFEniCS_PENALTY(mesh, min_mesh_spacing, V, 
                                                                            boundarymesh,
                                                                            BoundaryMesh_Nt,
                                                                            K, muCortex, 
                                                                            vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping_T0, 
                                                                            vertexBoundaryMesh_2_dofSWholeMesh_mapping, vertexBoundaryMesh_2_dofsVWholeMesh_mapping, 
                                                                            u_avg):

    """
    Penalizing self-contact (unknown master / slave) within FEM problem resolution 
    --> "penalty method" to optimize the resolution
    --> point-to-surface detection algorithm
    """

    # contact_stiffness = penalty coefficient
    """epsilon_n = 100 * K.values()[0]""" # T.Tallinen
    
    E = 9*muCortex.values()[0]*K.values()[0]/(3*K.values()[0] + muCortex.values()[0]) # Young modulus
    h = min_mesh_spacing # element size
    epsilon_n = E/h

    # intialize FEniCS contact Function
    f_penalty_self_contact_V = fenics.Function(V)

    # 1. Detect collision in the mesh
    # -------------------------------
    """source: https://fenicsproject.discourse.group/t/speed-up-collision-detection-bounding-box-tree-for-point-cloud/7163/3"""

    bbtree_mesh = fenics.BoundingBoxTree()
    bbtree_mesh.build(mesh, mesh.topology().dim())
    coordinates_mesh = mesh.coordinates()
    c_to_v = mesh.topology()(mesh.topology().dim(), 0)

    faces = boundarymesh.cells()
    n_faces = len(faces)
    coordinates_boundarymesh = boundarymesh.coordinates()
    bbtree_bmesh = fenics.BoundingBoxTree()
    bbtree_bmesh.build(boundarymesh, boundarymesh.topology().dim())

    cells = mesh.cells() # cell1: [node1, node2, node3]; cell1: [node1, node4, node10]
    cells_flat = cells.flatten().tolist()
    counter = Counter(cells_flat) # returns all node indices 

    collision_tets_indices_for_each_surface_vertex = [] # debug: [i for i in range(coordinates_mesh.shape[0]) if len(number_of_collisions_experiences_by_each_vertex[:][i]) != 0]
    detected_boundarymesh_nodes_that_experience_collision_IDX = []
    detected_boundarymesh_colliding_nodes_COORDS = []

    for i in range(coordinates_boundarymesh.shape[0]): # i: surface node index --> parse only surface nodes to detect collisions
        # Compute the number of collisions with mesh tetraedrons for surface node i
        x = fenics.Point(coordinates_mesh[i]) #TODO: + u.vector()[ vertexBoundaryMesh_2_dofsVWholeMesh_mapping[i] ]

        collision_cells = bbtree_mesh.compute_entity_collisions(x)
        collision_tets_indices_for_each_surface_vertex.append(collision_cells) 
        num_collisions = len(collision_cells) # number of collisions experienced by that vertex with any cell. at least equals to the number of neighbour-cells to which the node belongs.

        # get the precomputed value
        iB_2_iWholeMesh = mappings.boundaryVertexIndex_2_meshVertexIndex(coordinates_boundarymesh, coordinates_mesh)
        num_cells_associated_to_that_vertex = counter[iB_2_iWholeMesh[i]] # counter[i] returns the number of times node "i" is a component of a cell (e.g. counter[315]: 7 meaning node 315 enters into the composition of 7 cells)

        if num_collisions > num_cells_associated_to_that_vertex:
            detected_boundarymesh_nodes_that_experience_collision_IDX.append(i) # add surface node index as colliding node
            detected_boundarymesh_colliding_nodes_COORDS.append(boundarymesh.coordinates()[i])
    
    detected_boundarymesh_nodes_that_experience_collision_IDX = np.array(detected_boundarymesh_nodes_that_experience_collision_IDX)
    
    """
    plot_vertices_onto_mesh3D(mesh, mesh_colliding_nodes_coords, "mesh self-collision nodes")
    plot_vertex_onto_mesh3D(mesh, coordinates_mesh[343], "node 343")
    """

    # 2. Get real collision couples (surface vertex IDX, mesh cell IDX)
    # -----------------------------------------------------------------
    surface_colliding_node_IDX_2_real_collision_tets_IDX = [] # [surface node that experiments collision IDX, real colliding cell 1], [surface node that experiments collision IDX, real colliding cell 2]?
    
    for i, surface_vertex_index in enumerate(detected_boundarymesh_nodes_that_experience_collision_IDX):

        collision_tets_IDX = collision_tets_indices_for_each_surface_vertex[i]

        # if surface vertex is not included in cell: real colliding cell
        for tet_IDX in collision_tets_IDX:
            tet_4vertices_IDX = c_to_v(tet_IDX)
            #detectedTet_4vertices_IDX = faceIDX_to_verticesIDX_mapping(detected_boundaryFace_IDX) # 4 nodes indices of the tet colliding with the point          
            
            if surface_vertex_index not in tet_4vertices_IDX[:]:  # if  node does not belong to the tetraedron (= real collision) (= "if pt != faces[tri,0] and pt != faces[tri,1] and pt != faces[tri,2]:"")
                surface_colliding_node_IDX_2_real_collision_tets_IDX.append([surface_vertex_index, tet_IDX])

    surface_colliding_node_IDX_2_real_collision_tets_IDX = np.array(surface_colliding_node_IDX_2_real_collision_tets_IDX)

    # Penalize contact in case of penetration (normal gap < 0)
    for i, couple in enumerate(surface_colliding_node_IDX_2_real_collision_tets_IDX): # parse all "slave" surface nodes
        
        # 3. Compute normal gap between colliding "slave" surface node and master associated mesh cell (tetraedron)
        # ---------------------------------------------------------------------------------------------------------
        slave_surface_node_IDX = couple[0]
        rS = coordinates_boundarymesh[slave_surface_node_IDX] # coordinate of colliding surface node

        tet_IDX = couple[1]
        tet_4vertices_IDX = c_to_v(tet_IDX)

        # compute closest node to rS in colliding tet
        min_dist = 100 # distance between S to tet vertices
        closest_tet_vertex_IDX = -1 # get closest vertex to S
        for vertex_IDX in tet_4vertices_IDX: 
            coords_tet_vertex_i = coordinates_mesh[vertex_IDX]
            S_tetVi = np.linalg.norm(rS - coords_tet_vertex_i)
            min_dist = min(S_tetVi, min_dist) 
            if min_dist == S_tetVi:
                closest_tet_vertex_IDX = vertex_IDX # replace closest vertex

        # project tet node onto original surface node to get an approximation of projection of S onto surface (otherwise, slave node will be considered as closest boundary node)
        projection_of_S_on_master_IDX = vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping_T0[closest_tet_vertex_IDX] # in boundarymesh reference

        # compute coordinates of approximated projection of S
        p_rS_on_master = coordinates_boundarymesh[projection_of_S_on_master_IDX] # projection_of_S_on_master_IDX: IDX du noeud de surface "projeté" du vertex du tétrèdre le plus proche

        # absolute gap between "slave" node and its closest node on "master" surface (vector) 
        g_MS = rS - p_rS_on_master # = rS - rM (absolute gap_master_to_slave)
        #TODO: rS + u.vector()[ vertexBoundaryMesh_2_dofsVWholeMesh_mapping[i] ] - (p_rS_on_master + u.vector()[ vertexBoundaryMesh_2_dofsVWholeMesh_mapping[projection_of_S_on_master_IDX] )

        # compute normal of the proximity "master" cell/face
        """Ntri = cross_dim_2(coordinates[faces[i_proximity_face_master, 1]] - coordinates[faces[i_proximity_face_master, 0]], 
                           coordinates[faces[i_proximity_face_master, 2]] - coordinates[faces[i_proximity_face_master, 0]]) # Triangle normal
        Ntri *= 1.0/np.linalg.norm(Ntri)"""
        #n = BoundaryMesh_Nt.vector()[vertexBoundaryMesh_2_dofsVWholeMesh_mapping[xxx]] # xxx=closest surface boundary node IDX to tet vertex
        nM = BoundaryMesh_Nt.vector()[ vertexBoundaryMesh_2_dofsVWholeMesh_mapping[projection_of_S_on_master_IDX] ]

        # compute normal gap (scalar)
        gn = np.dot(g_MS, nM)

        # 4. Compute additional residual term due to contact penalization (in the contact zone)
        # -------------------------------------------------------------------------------------
        if gn < 0. - fenics.DOLFIN_EPS: # penetration (Signorini conditions for non penetration)

            # dgn = (δrS - δrM).nM = (vS_test - vM_test).Ntri # --> considered twice since all slave nodes are parsed 
            # -------------------------------------------------------------------------------------------------------
            # integration only on the "slave" contact boundary Gamma_c^2 (See equations) --> epsilon_n * (- gn) * nM * v_test_S
            f_penalty_self_contact_V.vector()[ vertexBoundaryMesh_2_dofsVWholeMesh_mapping[slave_surface_node_IDX] ] += epsilon_n * (-gn) * nM # Need to be mulitplied by v_test[xS_tplus1] * ds(101) 

            # integration only on the "master" contact boundary Gamma_c^1 (See equations) --> epsilon_n * (- gn) * nM * (- v_test_M)
            f_penalty_self_contact_V.vector()[ vertexBoundaryMesh_2_dofsVWholeMesh_mapping[ projection_of_S_on_master_IDX] ] -= epsilon_n * (-gn) * nM  # Need to be mulitplied by + v_test[p_rS_on_master_tplus1] * ds(101)

    return f_penalty_self_contact_V 

def contact_mechanics_algo_SelfContact_CollidingClosestTetNodeFEniCS_PENALTY_future(mesh, min_mesh_spacing, V, 
                                                                            boundarymesh,
                                                                            BoundaryMesh_Nt,
                                                                            K, muCortex, 
                                                                            vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping_T0, 
                                                                            vertexBoundaryMesh_2_dofSWholeMesh_mapping, vertexBoundaryMesh_2_dofsVWholeMesh_mapping, 
                                                                            u_avg):

    """
    Penalizing self-contact (unknown master / slave) within FEM problem resolution 
    --> "penalty method" to optimize the resolution
    --> point-to-surface detection algorithm
    """

    u_unknown_alphaGeneralizedMethod = fenics.Function(V)
    projection.local_project(u_avg, V, u_unknown_alphaGeneralizedMethod)

    # contact_stiffness = penalty coefficient
    """epsilon_n = 100 * K.values()[0]""" # T.Tallinen
    
    E = 9*muCortex.values()[0]*K.values()[0]/(3*K.values()[0] + muCortex.values()[0]) # Young modulus
    h = min_mesh_spacing # element size
    epsilon_n = E/h 
    
    # intialize FEniCS contact Function
    f_penalty_self_contact_V = fenics.Function(V)

    # 1. Detect collision in the mesh
    # -------------------------------
    """source: https://fenicsproject.discourse.group/t/speed-up-collision-detection-bounding-box-tree-for-point-cloud/7163/3"""

    bbtree_mesh = fenics.BoundingBoxTree()
    bbtree_mesh.build(mesh, mesh.topology().dim())
    coordinates_mesh = mesh.coordinates()
    c_to_v = mesh.topology()(mesh.topology().dim(), 0)

    faces = boundarymesh.cells()
    n_faces = len(faces)
    coordinates_boundarymesh = boundarymesh.coordinates()
    bbtree_bmesh = fenics.BoundingBoxTree()
    bbtree_bmesh.build(boundarymesh, boundarymesh.topology().dim())

    cells = mesh.cells() # cell1: [node1, node2, node3]; cell1: [node1, node4, node10]
    cells_flat = cells.flatten().tolist()
    counter = Counter(cells_flat) # returns all node indices 

    collision_tets_indices_for_each_surface_vertex = [] # debug: [i for i in range(coordinates_mesh.shape[0]) if len(number_of_collisions_experiences_by_each_vertex[:][i]) != 0]
    detected_boundarymesh_nodes_that_experience_collision_IDX = []
    detected_boundarymesh_colliding_nodes_COORDS = []

    for i in range(coordinates_boundarymesh.shape[0]): # i: surface node index --> parse only surface nodes to detect collisions
        # Compute the number of collisions with mesh tetraedrons for surface node i
        x = fenics.Point(coordinates_mesh[i] + u_unknown_alphaGeneralizedMethod.vector()[ vertexBoundaryMesh_2_dofsVWholeMesh_mapping[i] ])

        collision_cells = bbtree_mesh.compute_entity_collisions(x)
        collision_tets_indices_for_each_surface_vertex.append(collision_cells) 
        num_collisions = len(collision_cells) # number of collisions experienced by that vertex with any cell. at least equals to the number of neighbour-cells to which the node belongs.

        # get the precomputed value
        iB_2_iWholeMesh = mappings.boundaryVertexIndex_2_meshVertexIndex(coordinates_boundarymesh, coordinates_mesh)
        num_cells_associated_to_that_vertex = counter[iB_2_iWholeMesh[i]] # counter[i] returns the number of times node "i" is a component of a cell (e.g. counter[315]: 7 meaning node 315 enters into the composition of 7 cells)

        if num_collisions > num_cells_associated_to_that_vertex:
            detected_boundarymesh_nodes_that_experience_collision_IDX.append(i) # add surface node index as colliding node
            detected_boundarymesh_colliding_nodes_COORDS.append(boundarymesh.coordinates()[i])
    
    detected_boundarymesh_nodes_that_experience_collision_IDX = np.array(detected_boundarymesh_nodes_that_experience_collision_IDX)
    
    """
    plot_vertices_onto_mesh3D(mesh, mesh_colliding_nodes_coords, "mesh self-collision nodes")
    plot_vertex_onto_mesh3D(mesh, coordinates_mesh[343], "node 343")
    """

    # 2. Get real collision couples (surface vertex IDX, mesh cell IDX)
    # -----------------------------------------------------------------
    surface_colliding_node_IDX_2_real_collision_tets_IDX = [] # [surface node that experiments collision IDX, real colliding cell 1], [surface node that experiments collision IDX, real colliding cell 2]?
    
    for i, surface_vertex_index in enumerate(detected_boundarymesh_nodes_that_experience_collision_IDX):

        collision_tets_IDX = collision_tets_indices_for_each_surface_vertex[i]

        # if surface vertex is not included in cell: real colliding cell
        for tet_IDX in collision_tets_IDX:
            tet_4vertices_IDX = c_to_v(tet_IDX)
            #detectedTet_4vertices_IDX = faceIDX_to_verticesIDX_mapping(detected_boundaryFace_IDX) # 4 nodes indices of the tet colliding with the point          
            
            if surface_vertex_index not in tet_4vertices_IDX[:]:  # if  node does not belong to the tetraedron (= real collision) (= "if pt != faces[tri,0] and pt != faces[tri,1] and pt != faces[tri,2]:"")
                surface_colliding_node_IDX_2_real_collision_tets_IDX.append([surface_vertex_index, tet_IDX])

    surface_colliding_node_IDX_2_real_collision_tets_IDX = np.array(surface_colliding_node_IDX_2_real_collision_tets_IDX)

    # Penalize contact in case of penetration (normal gap < 0)
    for i, couple in enumerate(surface_colliding_node_IDX_2_real_collision_tets_IDX): # parse all "slave" surface nodes
        
        # 3. Compute normal gap between colliding "slave" surface node and master associated mesh cell (tetraedron)
        # ---------------------------------------------------------------------------------------------------------
        slave_surface_node_IDX = couple[0]
        rS = coordinates_boundarymesh[slave_surface_node_IDX] # coordinate of colliding surface node

        tet_IDX = couple[1]
        tet_4vertices_IDX = c_to_v(tet_IDX)

        # compute closest node to rS in colliding tet
        min_dist = 100 # distance between S to tet vertices
        closest_tet_vertex_IDX = -1 # get closest vertex to S
        for vertex_IDX in tet_4vertices_IDX: 
            coords_tet_vertex_i = coordinates_mesh[vertex_IDX]
            S_tetVi = np.linalg.norm(rS - coords_tet_vertex_i)
            min_dist = min(S_tetVi, min_dist) 
            if min_dist == S_tetVi:
                closest_tet_vertex_IDX = vertex_IDX # replace closest vertex

        # project tet node onto original surface node to get an approximation of projection of S onto surface (otherwise, slave node will be considered as closest boundary node)
        projection_of_S_on_master_IDX = vertexWholeMeshIDX_to_projectedVertexBoundaryMeshIDX_mapping_T0[closest_tet_vertex_IDX] # in boundarymesh reference

        # compute coordinates of approximated projection of S
        p_rS_on_master = coordinates_boundarymesh[projection_of_S_on_master_IDX] # projection_of_S_on_master_IDX: IDX du noeud de surface "projeté" du vertex du tétrèdre le plus proche

        # absolute gap between "slave" node and its closest node on "master" surface (vector) 
        g_MS = rS + u_unknown_alphaGeneralizedMethod.vector()[ vertexBoundaryMesh_2_dofsVWholeMesh_mapping[i] ] \
            - (p_rS_on_master + u_unknown_alphaGeneralizedMethod.vector()[ vertexBoundaryMesh_2_dofsVWholeMesh_mapping[projection_of_S_on_master_IDX] ] ) # = rS - rM (absolute gap_master_to_slave)

        # compute normal of the proximity "master" cell/face
        """Ntri = cross_dim_2(coordinates[faces[i_proximity_face_master, 1]] - coordinates[faces[i_proximity_face_master, 0]], 
                           coordinates[faces[i_proximity_face_master, 2]] - coordinates[faces[i_proximity_face_master, 0]]) # Triangle normal
        Ntri *= 1.0/np.linalg.norm(Ntri)"""
        #n = BoundaryMesh_Nt.vector()[vertexBoundaryMesh_2_dofsVWholeMesh_mapping[xxx]] # xxx=closest surface boundary node IDX to tet vertex
        nM = BoundaryMesh_Nt.vector()[ vertexBoundaryMesh_2_dofsVWholeMesh_mapping[projection_of_S_on_master_IDX] ]

        # compute normal gap (scalar)
        gn = np.dot(g_MS, nM)

        # 4. Compute additional residual term due to contact penalization (in the contact zone)
        # -------------------------------------------------------------------------------------
        if gn < 0. - fenics.DOLFIN_EPS: # penetration (Signorini conditions for non penetration)

            # dgn = (δrS - δrM).nM = (vS_test - vM_test).Ntri # --> considered twice since all slave nodes are parsed 
            # -------------------------------------------------------------------------------------------------------
            # integration only on the "slave" contact boundary Gamma_c^2 (See equations) --> epsilon_n * (- gn) * nM * v_test_S
            f_penalty_self_contact_V.vector()[ vertexBoundaryMesh_2_dofsVWholeMesh_mapping[slave_surface_node_IDX] ] += epsilon_n * (-gn) * nM # Need to be mulitplied by v_test[xS_tplus1] * ds(101) 

            # integration only on the "master" contact boundary Gamma_c^1 (See equations) --> epsilon_n * (- gn) * nM * (- v_test_M)
            f_penalty_self_contact_V.vector()[ vertexBoundaryMesh_2_dofsVWholeMesh_mapping[ projection_of_S_on_master_IDX] ] -= epsilon_n * (-gn) * nM  # Need to be mulitplied by + v_test[p_rS_on_master_tplus1] * ds(101)

    return f_penalty_self_contact_V 

########################################
# CONTACT MECHANICS ALGO 2 HEMISPHERES #
########################################

from scipy import spatial

def contact_mechanics_algo_2HemispheresSubmeshes_COLLISIONS_PENALTY(mesh, BoundaryMesh_Nt, u_avg, V, grNoGrowthZones,
                                                                    subdomains, V_BrainHemisphere1, V_BrainHemisphere2, 
                                                                    BoundaryMesh_Nt_Hemisphere1, BoundaryMesh_Nt_Hemisphere2, 
                                                                    vertexidxV_to_DOFsV_mapping_Hemisphere1, vertexidxV_to_DOFsV_mapping_Hemisphere2,
                                                                    SH1_2_S_dofmap, VH1_2_V_dofmap, vertexidxH1_to_DOF_SH1_mapping, vertexidxV_to_DOFsV_mapping_H1, 
                                                                    SH2_2_S_dofmap, VH2_2_V_dofmap, vertexidxH2_to_DOF_SH2_mapping, vertexidxV_to_DOFsV_mapping_H2,
                                                                    penalty_coefficient):
    """
    Contact Mechanics algorithm to avoid self-penetration (especially between the 2 brain hemispheres)
    - 2 submeshes to distinguish the two brain hemispheres are built
    - tetraedrons (DOFs_H1, DOFS_H2) that penetrate into the other hemisphere are detected for each hemisphere 
    - For each "colliding tetraedron", in each Hemisphere, "colliding" nodes IDX at the boundary (of H1 or H2) are collected (fenics function "compute_entity_collisions"):
        - its nodes that belong to the boundary are deduced if surface tetraedron (small penetration)
        - if none of its nodes belongs to the boundary (deep penetration), a projection of its 4 nodes at the boundary is computed 
    - For each Hemipshere, for each "colliding" boundary node:
        - the closest node on the other Hemisphere boundary and the gap vector (r12vec; r21vec) are computed 
        - the normal gap is computed (< 0 since penetration) at the future time (to consider penetration at t+1 and penalize it at t in the FEM residual form)
        - penalization force is computed in the VH1 and VH2 respective function spaces 
    - Output: penalization force (for all colliding boundary nodes) Function in the VH1 and VH2 respective function spaces
    """

    u_unknown_alphaGeneralizedMethod = fenics.Function(V)
    projection.local_project(u_avg, V, u_unknown_alphaGeneralizedMethod)

    # Reinitialize tmp forces (Function(V))
    # -------------------------------------
    fcontact_Hemisphere1 = fenics.Function(V_BrainHemisphere1)
    fcontact_Hemisphere2 = fenics.Function(V_BrainHemisphere2)

    # linear penalty force parameters. 
    # --------------------------------
    epsilon = fenics.DOLFIN_EPS 
    #Kc = 10 * K # Kc: contact stiffness; K: bulk modulus

    # 2 hemispheres submeshes & associated bounding box trees 
    # -------------------------------------------------------
    meshHemisphere1 = fenics.SubMesh(mesh, subdomains, 1) 
    meshHemisphere2 = fenics.SubMesh(mesh, subdomains, 2) 

    bbtreeHemisphere1 = fenics.BoundingBoxTree()
    bbtreeHemisphere1.build(meshHemisphere1, meshHemisphere1.topology().dim())

    bbtreeHemisphere2 = fenics.BoundingBoxTree()
    bbtreeHemisphere2.build(meshHemisphere2, meshHemisphere2.topology().dim())

    # Get vertices indices of each colliding tetrahedron (vertex indexation)
    c_to_v_Hemisphere1 = meshHemisphere1.topology()(meshHemisphere1.topology().dim(), 0) 
    c_to_v_Hemisphere2 = meshHemisphere2.topology()(meshHemisphere2.topology().dim(), 0)

    # Boundary of the 2 hemispheres submeshes
    # ---------------------------------------
    boundarymesh_Hemisphere1 = fenics.BoundaryMesh(meshHemisphere1, "exterior")
    boundarymesh_Hemisphere2 = fenics.BoundaryMesh(meshHemisphere2, "exterior")
    
    # Compute hemispheres anti-collision forces
    # -----------------------------------------
    colliding_entities_DOFs_Hemisphere1, colliding_entities_DOFs_Hemisphere2 = bbtreeHemisphere1.compute_entity_collisions(bbtreeHemisphere2) # Get the list of pair-colliding tetrahedrons in Hemisphere1 lobe with Hemisphere2 lobe (dof indexation)

    if len(colliding_entities_DOFs_Hemisphere1) == 0:
        pass
    else: 
        # Compute coordinates of center of gravity of each colliding paires-tetrahedrons
        colliding_nodes_onBoundary_Hemisphere1_indices = []
        colliding_nodes_onBoundary_Hemisphere1_coords = []

        colliding_nodes_onBoundary_Hemisphere2_indices = []
        colliding_nodes_onBoundary_Hemisphere2_coords = []
        
        for i, (cell_H1_dof, cell_H2_dof) in enumerate(zip(colliding_entities_DOFs_Hemisphere1, colliding_entities_DOFs_Hemisphere2)): 
            vertex_indices_Hemisphere1 = c_to_v_Hemisphere1(cell_H1_dof) # --> vertex index in VH1. 4 vertices indices for tetrahedron cellHemisphere1 ( vertices_Hemisphere1 (vertex indexation) & cellHemisphere1 (dof indexation) )
            vertex_indices_Hemisphere2 = c_to_v_Hemisphere2(cell_H2_dof) # --> vertex index in VH2. 4 vertices indices for tetrahedron cellHemisphere2 / 

            # Keep colliding nodes that are on the boundaries only:
            for node_idx_H1 in vertex_indices_Hemisphere1: # node from colliding entity
                if meshHemisphere1.coordinates()[node_idx_H1] in boundarymesh_Hemisphere1.coordinates() and grNoGrowthZones.vector()[ SH1_2_S_dofmap[vertexidxH1_to_DOF_SH1_mapping[node_idx_H1]] ] != 0: # boundarymesh_Hemisphere1
                    colliding_nodes_onBoundary_Hemisphere1_indices.append(node_idx_H1)
                    colliding_nodes_onBoundary_Hemisphere1_coords.append(meshHemisphere1.coordinates()[node_idx_H1])
                else: # deep penetration. tetraedron is a pure volume cell. Need to get closest nodes at the boundary and add them to the list of "colliding surface nodes"
                    # get closest node at the contact boundary 1
                    closestBoundaryH1PointIDX_inBH1ref = np.argmin(np.linalg.norm(boundarymesh_Hemisphere1.coordinates() - meshHemisphere1.coordinates()[node_idx_H1], axis=1)) # B index (in the B ref) for 2D closest Point/Vertex
                    #closestBoundaryH1PointIDX_inVH1ref = vertexB_2_dofsV_mapping[closestBoundaryH1PointIDX_inBH1ref]
                    closestBoundaryH1PointIDX_inVH1ref = [ i_V_H1_ref for i_V_H1_ref in range(meshHemisphere1.num_vertices()) if meshHemisphere1.coordinates()[i_V_H1_ref].all() == boundarymesh_Hemisphere1.coordinates()[closestBoundaryH1PointIDX_inBH1ref].all() ][0]

                    colliding_nodes_onBoundary_Hemisphere1_indices.append(closestBoundaryH1PointIDX_inVH1ref) # add closest node on H1 boundary to colliding nodes list
                    colliding_nodes_onBoundary_Hemisphere1_coords.append(meshHemisphere1.coordinates()[closestBoundaryH1PointIDX_inVH1ref])

            for node_idx_H2 in vertex_indices_Hemisphere2: # node from colliding entity
                if meshHemisphere2.coordinates()[node_idx_H2] in boundarymesh_Hemisphere2.coordinates() and grNoGrowthZones.vector()[ SH2_2_S_dofmap[vertexidxH2_to_DOF_SH2_mapping[node_idx_H2]] ] != 0: # boundarymesh_Hemisphere2
                    colliding_nodes_onBoundary_Hemisphere2_indices.append(node_idx_H2)
                    colliding_nodes_onBoundary_Hemisphere2_coords.append(meshHemisphere2.coordinates()[node_idx_H2])
                else: # deep penetration. tetraedron is a pure volume cell. Need to get closest nodes at the boundary and add them to the list of "colliding surface nodes"
                    # get closest node at the contact boundary 2
                    closestBoundaryH2PointIDX_inBH2ref = np.argmin(np.linalg.norm(boundarymesh_Hemisphere2.coordinates() - meshHemisphere2.coordinates()[node_idx_H2], axis=1)) # B index (in the B ref) for 2D closest Point/Vertex
                    #closestBoundaryH2PointIDX_inVH2ref = vertexB_2_dofsV_mapping[closestBoundaryH2PointIDX_inBH2ref]
                    closestBoundaryH2PointIDX_inVH2ref = [ i_V_H2_ref for i_V_H2_ref in range(meshHemisphere2.num_vertices()) if meshHemisphere2.coordinates()[i_V_H2_ref].all() == boundarymesh_Hemisphere2.coordinates()[closestBoundaryH2PointIDX_inBH2ref].all() ][0]

                    colliding_nodes_onBoundary_Hemisphere2_indices.append(closestBoundaryH2PointIDX_inVH2ref) # add closest node on H2 boundary to colliding nodes list
                    colliding_nodes_onBoundary_Hemisphere2_coords.append(meshHemisphere2.coordinates()[closestBoundaryH2PointIDX_inVH2ref])

        # remove redundant nodes/coords:
        colliding_nodes_onBoundary_Hemisphere1_indices = np.unique(colliding_nodes_onBoundary_Hemisphere1_indices)
        colliding_nodes_onBoundary_Hemisphere1_coords = np.unique(colliding_nodes_onBoundary_Hemisphere1_coords, axis=0)

        colliding_nodes_onBoundary_Hemisphere2_indices = np.unique(colliding_nodes_onBoundary_Hemisphere2_indices)
        colliding_nodes_onBoundary_Hemisphere2_coords = np.unique(colliding_nodes_onBoundary_Hemisphere2_coords, axis=0)

        # Check simultaneously Hemisphere1 & Hemisphere2 paired-colliding tetrahedrons COGs: 
        """ if len(colliding_nodes_onBoundary_Hemisphere1_indices) != 0:
            plot_colliding_nodes_onto_mesh3D(mesh_unknown, colliding_nodes_onBoundary_Hemisphere1_coords, 'red', colliding_nodes_onBoundary_Hemisphere2_coords, 'blue', "Colliding nodes in Hemisphere1 hemisphere (RED) & Hemisphere2 hemisphere (BLUE) that belong to boundaries (volumic colliding nodes not considered)") """

        # Trees of meshes boundaries

        treeBoundaryMeshHemisphere1 = spatial.cKDTree(boundarymesh_Hemisphere1.coordinates()) 
        treeBoundaryMeshHemisphere2 = spatial.cKDTree(boundarymesh_Hemisphere2.coordinates())
        
        # for each boundary node (Hemisphere1 or Hemisphere2 hemisphere), compute closest distance (contact gap) to other hemisphere boundary:
        # Hemisphere1 hemisphere (colliding boundary nodes)
        dis_to_Hemisphere2 = [] # distance to Hemisphere2 hemisphere (r12)
        vector_to_Hemisphere2 = [] # r12vec
        closestHemisphere2Nodes_indices = []

        #for node_coords_H1 in colliding_nodes_onBoundary_Hemisphere1_coords:
        for node_idx_H1 in colliding_nodes_onBoundary_Hemisphere1_indices:

            node_coords_H1 = meshHemisphere1.coordinates()[node_idx_H1]

            distance_to_closest_boundaryNodeHemisphere2, closest_boundaryNodeHemisphere2_idx = treeBoundaryMeshHemisphere2.query(node_coords_H1) # compute closest node on Hemisphere2 boundary to Point
            closest_boundaryNodeHemisphere2_coords = boundarymesh_Hemisphere2.coordinates()[closest_boundaryNodeHemisphere2_idx]

            #r12vec = closest_boundaryNodeHemisphere2_coords - node_coords_H1 # x2 - x1 
            x1_avg = node_coords_H1 + u_unknown_alphaGeneralizedMethod.vector()[ VH1_2_V_dofmap[vertexidxV_to_DOFsV_mapping_H1[node_idx_H1]] ] 
            x2_avg = closest_boundaryNodeHemisphere2_coords + u_unknown_alphaGeneralizedMethod.vector()[ VH2_2_V_dofmap[vertexidxV_to_DOFsV_mapping_H2[closest_boundaryNodeHemisphere2_idx]] ]
            r12vec = x2_avg - x1_avg # x2_t+1_{avg} - x1_t+1_{avg} 

            dis_to_Hemisphere2.append(distance_to_closest_boundaryNodeHemisphere2) # gaps of Hemisphere1 nodes
            closestHemisphere2Nodes_indices.append(closest_boundaryNodeHemisphere2_idx)
            vector_to_Hemisphere2.append( r12vec ) 

        """
        if visualization == True:
            #plot_colliding_nodes_onto_mesh3D(mesh, colliding_nodes_onBoundary_Hemisphere1_coords, 'black', boundarymesh_Hemisphere2.coordinates()[closestHemisphere2Nodes_indices[:]], 'green', "Closest nodes on the Hemisphere2 hemisphere boundary detected (GREEN) to Hemisphere1 nodes BELONGING TO COLLIDING TETS that belong themselves to boundary (BLACK)")
            
            contact_idx_Hemisphere1 = np.where( np.array(vector2Hemisphere2Hemisphere)[:,1] < - epsilon)[0] # rLR[1] < - epsilon
            plot_colliding_nodes_onto_mesh3D(mesh, colliding_nodes_onBoundary_Hemisphere1_coords[contact_idx_Hemisphere1], 'black', boundarymesh_Hemisphere2.coordinates()[np.array(closestHemisphere2Nodes_indices)[contact_idx_Hemisphere1]], 'green', "Closest nodes on the Hemisphere2 hemisphere boundary detected (GREEN) to COLLIDING Hemisphere1 nodes that belong to boundary (BLACK)")
        """
        
        # Hemisphere2 hemisphere (colliding boundary nodes)
        dis_to_Hemisphere1 = [] # distance to Hemisphere2 hemisphere
        vector_to_Hemisphere1 = []
        closestHemisphere1Nodes_indices = []

        for node_idx_H2 in colliding_nodes_onBoundary_Hemisphere2_indices:

            node_coords_H2 = meshHemisphere2.coordinates()[node_idx_H2]

            distance_to_closest_boundaryNodeHemisphere1, closest_boundaryNodeHemisphere1_idx = treeBoundaryMeshHemisphere1.query(node_coords_H2) 
            closest_boundaryNodeHemisphere1_coords = boundarymesh_Hemisphere1.coordinates()[closest_boundaryNodeHemisphere1_idx]
            
            # r21vec = closest_boundaryNodeHemisphere1_coords - node_coords_H2 # x1 - x2 
            x2_avg = node_coords_H2 + u_unknown_alphaGeneralizedMethod.vector()[ VH2_2_V_dofmap[vertexidxV_to_DOFsV_mapping_H2[node_idx_H2]] ] 
            x1_avg = closest_boundaryNodeHemisphere1_coords + u_unknown_alphaGeneralizedMethod.vector()[ VH1_2_V_dofmap[vertexidxV_to_DOFsV_mapping_H1[closest_boundaryNodeHemisphere1_idx]] ]
            r21vec = x1_avg - x2_avg # x1_t+1_{avg} - x2_t+1_{avg} 

            dis_to_Hemisphere1.append(distance_to_closest_boundaryNodeHemisphere1) # gaps of Hemisphere1 nodes
            closestHemisphere1Nodes_indices.append(closest_boundaryNodeHemisphere1_idx)
            vector_to_Hemisphere1.append( r21vec )
        
        """if visualization == True:
            #plot_colliding_nodes_onto_mesh3D(mesh, colliding_nodes_onBoundary_Hemisphere2_coords, 'black', boundarymesh_Hemisphere1.coordinates()[closestHemisphere1Nodes_indices[:]], 'green', "Closest nodes on the Hemisphere1 hemisphere boundary detected (GREEN) to Hemisphere2 nodes BELONGING TO COLLIDING TETS that belong themselves to boundary (BLACK)")
            
            contact_idx_Hemisphere2 = np.where( - np.array(vector2Hemisphere1Hemisphere)[:,1] < - epsilon)[0] # - rRL[1] < - epsilon
            plot_colliding_nodes_onto_mesh3D(mesh, colliding_nodes_onBoundary_Hemisphere2_coords[contact_idx_Hemisphere2], 'black', boundarymesh_Hemisphere1.coordinates()[np.array(closestHemisphere1Nodes_indices)[contact_idx_Hemisphere2]], 'green', "Closest nodes on the Hemisphere1 hemisphere boundary detected (GREEN) to COLLIDING Hemisphere2 nodes that belong to boundary (BLACK)")
        """

        # Compute anti-collision forces to apply onto Hemisphere1 boundary colliding nodes:
        for i, colliding_boundary_node_H1_idx in enumerate(colliding_nodes_onBoundary_Hemisphere1_indices): # colliding_boundary_node_H1: node idx, node coords
            
            #f_H1 = penalty_force(vector_to_Hemisphere2[i], penalty_coefficient, BoundaryMesh_Nt_Hemisphere1.vector()[ vertexidxV_to_DOFsV_mapping_Hemisphere1[ colliding_boundary_node_H1_idx ]], epsilon) # See annexe * // debug: np.where(np.array(vector2Hemisphere2Hemisphere)[:,1] < - epsilon)
            normal_vector_at_Boundary1 = BoundaryMesh_Nt_Hemisphere1.vector()[ vertexidxV_to_DOFsV_mapping_Hemisphere1[ colliding_boundary_node_H1_idx ]]
            fcontact_Hemisphere1.vector()[ vertexidxV_to_DOFsV_mapping_Hemisphere1[ colliding_boundary_node_H1_idx ] ] += penalty_coefficient * np.dot(r12vec, normal_vector_at_Boundary1) * normal_vector_at_Boundary1  

        # Compute anti-collision forces to apply onto Hemisphere2 boundary colliding nodes:
        for i, colliding_boundary_node_H2_idx in enumerate(colliding_nodes_onBoundary_Hemisphere2_indices): # colliding_tets_COG_coords_Hemisphere1 (current configuration) --> corresponds to x1
            
            #f_H2 = penalty_force( vector_to_Hemisphere1[i], penalty_coefficient, BoundaryMesh_Nt_Hemisphere2.vector()[ vertexidxV_to_DOFsV_mapping_Hemisphere2[ colliding_boundary_node_H2_idx ]], epsilon) # See annexe *
            #fcontact_Hemisphere2.vector()[ vertexidxV_to_DOFsV_mapping_Hemisphere2[ colliding_boundary_node_H2_idx ] ] += f_H2 
            normal_vector_at_Boundary2 = BoundaryMesh_Nt_Hemisphere2.vector()[ vertexidxV_to_DOFsV_mapping_Hemisphere2[ colliding_boundary_node_H2_idx ]]
            fcontact_Hemisphere2.vector()[ vertexidxV_to_DOFsV_mapping_Hemisphere2[ colliding_boundary_node_H2_idx ] ] += penalty_coefficient * np.dot(vector_to_Hemisphere1[i], normal_vector_at_Boundary2) * normal_vector_at_Boundary2 # vector_to_Hemisphere1[i] = r21_vec
    
    """
    * 
    Annexe:
    --------
    1: Hemisphere2; 2: Hemisphere2

    contact conditions for Hemisphere1 nodes:
    r12.y > 0 --> no contact
    r12.y < 0 and r12.y > -ε --> contact (ε is the error margin. small penetration is allowed by numerical imprecision, that should be the order of displacement precision)
    

    contact conditions for Hemisphere2 hemisphere nodes:
    r21.y < 0 --> no contact
    r21.y > 0 and r12.y < ε --> contact (ε is the error margin. small penetration is allowed by numerical imprecision, that should be the order of displacement precision)
    <=>
    -r21.y > 0 
    -r21.y < 0 and -r21.y > -ε 
    """

    return fcontact_Hemisphere1, fcontact_Hemisphere2 # Function(V)


def contact_mechanics_algo_2HemispheresSubmeshes_DISTANCE_PENALTY(mesh, V, u_avg, average_mesh_spacing, grNoGrowthZones,
                                                                  subdomains, V_BrainHemisphere1, V_BrainHemisphere2, 
                                                                  BoundaryMesh_Nt_Hemisphere1, BoundaryMesh_Nt_Hemisphere2,
                                                                  VH1_2_V_dofmap, VH2_2_V_dofmap, 
                                                                  vertexBH1_2_dofsVH1_mapping, vertexBH2_2_dofsVH2_mapping,
                                                                  vertexboundaryHemisphere2_2_dofScalarFunctionSpaceWholeMesh_mapping,
                                                                  penalty_coefficient):
                                                           
    """
    Contact Mechanics algorithm to avoid self-penetration (especially between the 2 brain hemispheres)
    - 2 submeshes to distinguish the two brain hemispheres are built
    - Here, we consider "master" contact boundary of Hemisphere 1 only on which to compute penalyzation residual form equations (see Penalty Method theory). In fine, it is equivalent to applying penalty forces on each hemisphere boundary:
        - For each surface node of Hemisphere 1, closest node IDX at the boundary of Hemisphere 2 is collected (treequery)
        - The gap vector (r12vec) and associated normal gap (g12_n) are computed, at the future time (to consider penetration at t+1 and penalize it at t in the FEM residual form
        - if g12_n < 0, there is penetration --> penalization forces are computed at node 1 ("master") in the VH1 Function Space and closest node 2 ("slave") in the VH1 Function Space to apply an equivalence of : 
          integral[ sigma(u) : ∇v * dx ] + integral[ ϵn · (- g12_n ) · n_H1 · ( v_test_H2 _ v_test_H1) * ds(H1) ] = 0  --> ( v_test_H2 - v_test_H1) * ds(H1) ] <=> ϵn * (-g12_n) * n[at node 1] * v_test[at closest node 2] * ds(103) - ϵn * (-g12_n) * n[at node 1] * v_test[at node 1] * v_test * ds(102) 
    - Output: penalization force (for all boundary nodes at H1 & H2) Function in the VH1 and VH2 respective function spaces
    """

    u_unknown_alphaGeneralizedMethod = fenics.Function(V)
    projection.local_project(u_avg, V, u_unknown_alphaGeneralizedMethod)

    # Reinitialize tmp forces (Function(V))
    # -------------------------------------
    fcontact_V_Hemisphere1 = fenics.Function(V_BrainHemisphere1)
    fcontact_V_Hemisphere2 = fenics.Function(V_BrainHemisphere2)

    # Build bounding box trees for 2 lobes 
    # ------------------------------------
    meshHemisphere1 = fenics.SubMesh(mesh, subdomains, 1) 
    meshHemisphere2 = fenics.SubMesh(mesh, subdomains, 2) 

    bmesh_Hemisphere1 = fenics.BoundaryMesh(meshHemisphere1, "exterior")
    bmesh_Hemisphere2 = fenics.BoundaryMesh(meshHemisphere2, "exterior")
    
    bbtreeHemisphere1 = fenics.BoundingBoxTree()
    bbtreeHemisphere1.build(meshHemisphere1, meshHemisphere1.topology().dim())

    bbtreeHemisphere2 = fenics.BoundingBoxTree()
    bbtreeHemisphere2.build(meshHemisphere2, meshHemisphere2.topology().dim())

    # Trees of meshes boundaries
    treeBoundaryMeshHemisphere1 = spatial.cKDTree(bmesh_Hemisphere1.coordinates()) 
    treeBoundaryMeshHemisphere2 = spatial.cKDTree(bmesh_Hemisphere2.coordinates())

    # Detect closest nodes and compute normal gaps for all "master" (1) surface node of the contact boundary (1)
    # ----------------------------------------------------------------------------------------------------------
    for surface_node_IDX_Hemisphere_2, surface_node_coords_Hemisphere_2 in enumerate(bmesh_Hemisphere2.coordinates()): # 2: "slave" / 1: "master"

        if grNoGrowthZones.vector()[ vertexboundaryHemisphere2_2_dofScalarFunctionSpaceWholeMesh_mapping[surface_node_IDX_Hemisphere_2] ] != 0: # remove surface node in H1 & H2 that are in the interior of the volume Whole mesh but where generated by taking the surface of the 2 submeshes (otherwise, they will necessarily enter into contact)
            
            distance_to_closest_boundary_node_in_Hemisphere1, closest_boundaryNode_IDX_in_Hemisphere1 = treeBoundaryMeshHemisphere1.query(surface_node_coords_Hemisphere_2) # compute closest node on Hemisphere2 boundary to Point
            
            #r12_vec = bmesh_Hemisphere2.coordinates()[closest_boundaryNode_IDX_in_Hemisphere2] - surface_node_coords_Hemisphere_1 # MS vector (master to slave gap)
            x2_avg = surface_node_coords_Hemisphere_2 + u_unknown_alphaGeneralizedMethod.vector()[ VH2_2_V_dofmap[vertexBH2_2_dofsVH2_mapping[surface_node_IDX_Hemisphere_2]] ] 
            x1_avg = bmesh_Hemisphere1.coordinates()[closest_boundaryNode_IDX_in_Hemisphere1] + u_unknown_alphaGeneralizedMethod.vector()[ VH1_2_V_dofmap[vertexBH1_2_dofsVH1_mapping[closest_boundaryNode_IDX_in_Hemisphere1]] ]
            r12_vec = x2_avg - x1_avg # x2_t+1_{avg} - x1_t+1_{avg} / r12_vec = MS_vec (master to slave)

            #normal_vec_at_Hemisphere1 = BoundaryMesh_Nt.vector()[ VH1_2_V_dofmap[ vertexBH1_2_dofsVH1_mapping[surface_node_IDX_Hemisphere_1] ] ]
    
            #n1 = growth.compute_topboundary_normals(meshHemisphere1, ds(102), V_BrainHemisphere1) # n1 --> return fenics.Function(V_BrainHemisphere1)
            #normal_vec_at_Hemisphere1 = n1.vector()[ vertexBH1_2_dofsVH1_mapping[surface_node_IDX_Hemisphere_1] ] 

            normal_vec_at_Hemisphere1 = BoundaryMesh_Nt_Hemisphere1.vector()[ vertexBH1_2_dofsVH1_mapping[closest_boundaryNode_IDX_in_Hemisphere1] ] 

            g12_n = np.dot(r12_vec, normal_vec_at_Hemisphere1) # Compute normal gap gn = MS_vec.nM

            # if normal gn < 0, compute residual form to add (via contact forces in V Function space for "master" contact boundary)
            # ---------------------------------------------------------------------------------------------------------------------

            if g12_n < 0.0 + 0.2*average_mesh_spacing: #- fenics.DOLFIN_EPS: # penetration (Signorini conditions for non penetration)

                #DOFsV_tri0, DOFsV_tri1, DOFsV_tri2 = vertexB_2_dofsV_mapping[faces[tri, 0]], vertexB_2_dofsV_mapping[faces[tri, 1]], vertexB_2_dofsV_mapping[faces[tri, 2]] # get local displacement (v_test) at xM
                #v_test_approximated_at_xM = 1/3 * (v_test.vector()[DOFsV_tri0] + v_test.vector()[DOFsV_tri1] + v_test.vector()[DOFsV_tri2] ) # approximate the local test displacement function v_test at xM using v_test at the 3 points of the triangle. 
                
                # compute dgn = nM.(δrS - δrM) = nM.(δuS - δuM) (= nM.(vS - vM)? )
                """dgn = fenics.dot(n_tri, v_test.vector()[vertexB_2_dofsV_mapping[i]] - v_test_approximated_at_xM)"""

                #contact_penalty_integrand = epsilon_n * (-gn) * ( v_test.vector()[coordinates_2_DOFsV[xS]] - v_test.vector()[coordinates_2_DOFsV[xM]] )
                """contact_penalty_FEniCS_FEM_integrand = epsilon_n * (-gn)""" # scalar. rather use np.dot?

                # Hemisphere 2 ("slave contact boundary")
                fcontact_V_Hemisphere2.vector()[ vertexBH2_2_dofsVH2_mapping[ surface_node_IDX_Hemisphere_2 ] ] += penalty_coefficient * (-g12_n) * normal_vec_at_Hemisphere1  

                # Hemisphere 1 ("master contact boundary") --> N.B. the residual form will be an integration only on the "master" contact boundary Gamma_c^1 (See equations)
                fcontact_V_Hemisphere1.vector()[ vertexBH1_2_dofsVH1_mapping[ closest_boundaryNode_IDX_in_Hemisphere1 ] ] -= penalty_coefficient * (-g12_n) * normal_vec_at_Hemisphere1  

                #epsilon_n * fenics.dot( -gn, np.dot(n_tri, v_test))
        

    return fcontact_V_Hemisphere1, fcontact_V_Hemisphere2 # Function(V)