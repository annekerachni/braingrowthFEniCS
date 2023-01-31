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
        self.mark(self.boundaries, self.brainsurface_mark, check_midpoint=False) # https://fenicsproject.discourse.group/t/how-to-compute-boundary-mesh-and-submesh-from-an-halfdisk-mesh/9812/2
        return self.boundaries

""" 
class Bottom(fenics.SubDomain): 
    pass


class Left(fenics.SubDomain):
    pass


class Right(fenics.SubDomain):
    pass


class Front(fenics.SubDomain): 
    pass


class Back(fenics.SubDomain): 
    pass
 """


# Define Dirichlet boundary conditions
# ------------------------------------           
class DirichletBoundaryConditions:


    def __init__(self, 
                 mesh_dimension, 
                 VectorSpace_CG1_mesh, 
                 boundaries, 
                 consider_brainsurface_bc_TrueorFalse, brainsurface_mark, brainsurface_bc): 
                # consider_left_bc_TrueorFalse, left_mark, left_bc_type, 
                # consider_right_bc_TrueorFalse, right_mark, right_bc_type,
                # consider_bottom_bc_TrueorFalse, bottom_mark, bottom_bc_type,
                # front, front_bc_type, back, back_bc_type 
        
        self.dimension = mesh_dimension
        self.bcs = [] # by default boundary condition --> brainsurface is not fixed --> to be adapted using set_dirichlet_brainsurface_boundary_conditions

        if consider_brainsurface_bc_TrueorFalse == True:
            brainsurface_bc = self.set_dirichlet_brainsurface_boundary_conditions(VectorSpace_CG1_mesh, boundaries, brainsurface_mark, brainsurface_bc)
            self.bcs.append(brainsurface_bc)

        # if consider_left_bc_TrueorFalse == True:
        #     left_bc = self.set_dirichlet_left_boundary_conditions(VectorSpace_CG1_mesh, boundaries, left_mark, left_bc_type)
        #     self.bcs.append(left_bc)

        # if consider_right_bc_TrueorFalse == True:
        #     right_bc = self.set_dirichlet_right_boundary_conditions(VectorSpace_CG1_mesh, boundaries, right_mark, right_bc_type)
        #     self.bcs.append(right_bc)

        # if consider_bottom_bc_TrueorFalse == True:
        #     bottom_bc = self.set_dirichlet_bottom_boundary_conditions(VectorSpace_CG1_mesh, boundaries, bottom_mark, bottom_bc_type)
        #     self.bcs.append(bottom_bc)
        
        """
        if consider_front_bc_TrueorFalse == True:
            front_bc = self.set_dirichlet_front_boundary_conditions(VectorSpace_CG1_mesh, boundaries, front_mark, front_bc_type)
            self.bcs.append(front_bc)

        if consider_back_bc_TrueorFalse == True:
            back_bc = self.set_dirichlet_back_boundary_conditions(VectorSpace_CG1_mesh, boundaries, back_mark, back_bc_type)
            self.bcs.append(back_bc)
        """


    def set_dirichlet_brainsurface_boundary_conditions(self, VectorSpace_CG1_mesh, boundaries, brainsurface_mark, brainsurface_bc):  
        
        bc_brainsurface = fenics.DirichletBC(VectorSpace_CG1_mesh, brainsurface_bc , boundaries, brainsurface_mark)
        
        return bc_brainsurface


    def set_dirichlet_left_boundary_conditions(self, VectorSpace_CG1_mesh, boundaries, left_mark, left_bc_type): 

        if left_bc_type == "fixed":
            bc_left = fenics.DirichletBC(VectorSpace_CG1_mesh, fenics.Constant((0., 0., 0.)) , boundaries, left_mark)

        elif left_bc_type == "plan_rolling":
            bc_left = fenics.DirichletBC(VectorSpace_CG1_mesh.sub(0), fenics.Constant(0.) , boundaries, left_mark) # V.sub(0) =  x. Points of of Left plan cannot move in the X direction, but can roll in the (y,z) plan. 
            
        return bc_left


    def set_dirichlet_right_boundary_conditions(self, VectorSpace_CG1_mesh, boundaries, right_mark, right_bc_type): 

        if right_bc_type == "fixed":
            bc_right = fenics.DirichletBC(VectorSpace_CG1_mesh, fenics.Constant((0., 0., 0.)) , boundaries, right_mark)

        elif right_bc_type == "plan_rolling":
            bc_right = fenics.DirichletBC(VectorSpace_CG1_mesh.sub(0), fenics.Constant(0.) , boundaries, right_mark) # V.sub(0) =  x. Points of of Right plan cannot move in the X direction, but can roll in the (y,z) plan. 
            
        return bc_right


    def set_dirichlet_bottom_boundary_conditions(self, VectorSpace_CG1_mesh, boundaries, bottom_mark, bottom_bc_type): 

        if bottom_bc_type == "fixed":
            bc_bottom = fenics.DirichletBC(VectorSpace_CG1_mesh, fenics.Constant((0., 0., 0.)) , boundaries, bottom_mark)

        elif bottom_bc_type == "plan_rolling":
            bc_bottom = fenics.DirichletBC(VectorSpace_CG1_mesh.sub(2), fenics.Constant(0.), boundaries, bottom_mark) # able to move in (X, Y) plan but not in Z axis
            
        elif bottom_bc_type == "curving": # (cf. Carlos Lugo's work)
            pass
            
        return bc_bottom

