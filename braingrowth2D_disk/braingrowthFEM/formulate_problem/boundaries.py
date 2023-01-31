import fenics

# Boundaries classes 
# ------------------   
class BrainSurface(fenics.SubDomain): 
    

    def __init__(self, brainsurface_bmesh_bbtree, boundaries): 
        fenics.SubDomain.__init__(self)
        self.brainsurface_bmesh_bbtree = brainsurface_bmesh_bbtree
        self.boundaries = boundaries
        self.brainsurface_mark = 101


    def inside(self, x, on_boundary):
        _, distance = self.brainsurface_bmesh_bbtree.compute_closest_entity(fenics.Point(*x))
        return fenics.near(distance, fenics.DOLFIN_EPS)   
    

    def mark_brainsurface_boundary(self): 
        self.mark(self.boundaries, self.brainsurface_mark, check_midpoint=False) # Mark brainsurface boundary https://fenicsproject.discourse.group/t/how-to-compute-boundary-mesh-and-submesh-from-an-halfdisk-mesh/9812/2
        return self.boundaries


# Define Dirichlet boundary conditions
# ------------------------------------           
class DirichletBoundaryConditions:


    def __init__(self, 
                 mesh_dimension, 
                 VectorSpace_CG1_mesh, 
                 boundaries, 
                 consider_brainsurface_bc_TrueorFalse, brainsurface_mark, brainsurface_bc): 
        
        self.dimension = mesh_dimension
        self.bcs = [] # by default boundary condition --> brainsurface is not fixed --> to be adapted using set_dirichlet_brainsurface_boundary_conditions

        if consider_brainsurface_bc_TrueorFalse == True:
            brainsurface_bc = self.set_dirichlet_brainsurface_boundary_conditions(VectorSpace_CG1_mesh, boundaries, brainsurface_mark, brainsurface_bc)
            self.bcs.append(brainsurface_bc)


    def set_dirichlet_brainsurface_boundary_conditions(self, VectorSpace_CG1_mesh, boundaries, brainsurface_mark, brainsurface_bc):  
        bc_brainsurface = fenics.DirichletBC(VectorSpace_CG1_mesh, brainsurface_bc , boundaries, brainsurface_mark)
        return bc_brainsurface

