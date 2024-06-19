import fenics
import numpy as np
from collections import Counter
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os, sys

sys.path.append(os.path.dirname(sys.path[0]))

from braingrowth_3D.phenomenological_dynamic_3D.FEM_biomechanical_model import projection, mappings, growth


####################################################
# Anti-collision force function [T.Tallinen, 2016] #
####################################################

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

def anticipate_collisions(NNLt, 
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
    
    # contact_stiffness = penalty coefficient
    contact_stiffness = 100 * K.values()[0] # T.Tallinen
    
    """
    E = 9*muCortex*K/(3*K + muCortex) # Young modulus
    h = min_mesh_spacing # element size
    epsilon_n = E/h 
    """

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
                
                Ntri = cross_dim_2(coordinates[faces[tri,1]] - coordinates[faces[tri,0]], coordinates[faces[tri,2]] - coordinates[faces[tri,0]]) # Triangle normal
                Ntri *= 1.0/np.linalg.norm(Ntri)
                
                if np.dot(fn, Ntri) < 0.0:
                    fn -= Ntri*np.dot(fn, Ntri)*2.0

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