import fenics

# Boundaries classes 
# ------------------   
class BrainSurface(fenics.SubDomain): 
    

    def __init__(self, meshObj, boundaries): 
        fenics.SubDomain.__init__(self)
        self.meshObj = meshObj
        self.boundaries = boundaries
        self.brainsurface_mark = 101


    def inside(self, x, on_boundary):
        return fenics.near(x[1], self.meshObj.mesh_parameters['ymax'])  # https://fenicsproject.discourse.group/t/how-to-build-vectorfunctionspace-on-part-of-a-2d-mesh-boundary/9648/4  
    

    def mark_brainsurface_boundary(self): 
        self.mark(self.boundaries, self.brainsurface_mark, check_midpoint=False) # Mark brainsurface boundary https://fenicsproject.discourse.group/t/how-to-compute-boundary-mesh-and-submesh-from-an-halfdisk-mesh/9812/2
        return self.boundaries
    

class Left(fenics.SubDomain): 


    def __init__(self, meshObj, boundaries): 
        fenics.SubDomain.__init__(self)
        self.meshObj = meshObj
        self.boundaries = boundaries
        self.left_mark = 102
    

    def inside(self, x, on_boundary):
        return fenics.near(x[0], self.meshObj.mesh_parameters['xmin']) and on_boundary
    

    def mark_left_boundary(self):
        self.mark(self.boundaries, self.left_mark)
        return self.boundaries        


class Right(fenics.SubDomain): 


    def __init__(self, meshObj, boundaries): 
        fenics.SubDomain.__init__(self)
        self.meshObj = meshObj
        self.boundaries = boundaries
        self.right_mark = 103


    def inside(self, x, on_boundary):
        return fenics.near(x[0], self.meshObj.mesh_parameters['xmax']) and on_boundary
    

    def mark_right_boundary(self):
        self.mark(self.boundaries, self.right_mark)
        return self.boundaries 
    

class Bottom(fenics.SubDomain): 


    def __init__(self, meshObj, boundaries): 
        fenics.SubDomain.__init__(self)
        self.meshObj = meshObj
        self.boundaries = boundaries
        self.bottom_mark = 104


    def inside(self, x, on_boundary):
        return fenics.near(x[1], self.meshObj.mesh_parameters['ymin']) and on_boundary
    

    def mark_bottom_boundary(self):
        self.mark(self.boundaries, self.bottom_mark)
        return self.boundaries 


# Define Dirichlet boundary conditions
# ------------------------------------           
class DirichletBoundaryConditions:


    def __init__(self, 
                 mesh_dimension, 
                 VectorSpace_CG1_mesh, 
                 boundaries, 
                 consider_brainsurface_bc_TrueorFalse, brainsurface_mark, brainsurface_bc, 
                 consider_left_bc_TrueorFalse, left_mark, left_bc_type, 
                 consider_right_bc_TrueorFalse, right_mark, right_bc_type,
                 consider_bottom_bc_TrueorFalse, bottom_mark, bottom_bc_type): # front, front_bc_type, back, back_bc_type 
        
        self.dimension = mesh_dimension
        self.bcs = [] # by default boundary condition --> brainsurface is not fixed --> to be adapted using set_dirichlet_brainsurface_boundary_conditions

        if consider_brainsurface_bc_TrueorFalse == True:
            brainsurface_bc = self.set_dirichlet_brainsurface_boundary_conditions(VectorSpace_CG1_mesh, boundaries, brainsurface_mark, brainsurface_bc)
            self.bcs.append(brainsurface_bc)

        if consider_left_bc_TrueorFalse == True:
            left_bc = self.set_dirichlet_left_boundary_conditions(VectorSpace_CG1_mesh, boundaries, left_mark, left_bc_type)
            self.bcs.append(left_bc)

        if consider_right_bc_TrueorFalse == True:
            right_bc = self.set_dirichlet_right_boundary_conditions(VectorSpace_CG1_mesh, boundaries, right_mark, right_bc_type)
            self.bcs.append(right_bc)

        if consider_bottom_bc_TrueorFalse == True:
            bottom_bc = self.set_dirichlet_bottom_boundary_conditions(VectorSpace_CG1_mesh, boundaries, bottom_mark, bottom_bc_type)
            self.bcs.append(bottom_bc)


    def set_dirichlet_brainsurface_boundary_conditions(self, VectorSpace_CG1_mesh, boundaries, brainsurface_mark, brainsurface_bc):  
        bc_brainsurface = fenics.DirichletBC(VectorSpace_CG1_mesh, brainsurface_bc , boundaries, brainsurface_mark)
        return bc_brainsurface

    
    def set_dirichlet_left_boundary_conditions(self, VectorSpace_CG1_mesh, boundaries, left_mark, left_bc_type): 

        if left_bc_type == "fixed":
            bc_left = fenics.DirichletBC(VectorSpace_CG1_mesh, fenics.Constant((0., 0.)) , boundaries, left_mark)

        elif left_bc_type == "plan_rolling":
            bc_left = fenics.DirichletBC(VectorSpace_CG1_mesh.sub(0), fenics.Constant(0.), boundaries, left_mark) # V.sub(0) = x. Points of Left edge cannot move in the X direction (normal direction)
            
        return bc_left
    

    def set_dirichlet_right_boundary_conditions(self, VectorSpace_CG1_mesh, boundaries, right_mark, right_bc_type): 

        if right_bc_type == "fixed":
            bc_right = fenics.DirichletBC(VectorSpace_CG1_mesh, fenics.Constant((0., 0.)) , boundaries, right_mark)
            
        elif right_bc_type == "plan_rolling":
            bc_right = fenics.DirichletBC(VectorSpace_CG1_mesh.sub(0), fenics.Constant(0.), boundaries, right_mark) # V.sub(0) = x. Points of Right edge cannot move in the X direction (normal direction)
            
        return bc_right
    

    def set_dirichlet_bottom_boundary_conditions(self, VectorSpace_CG1_mesh, boundaries, bottom_mark, bottom_bc_type): 

        if bottom_bc_type == "fixed":
            bc_bottom = fenics.DirichletBC(VectorSpace_CG1_mesh, fenics.Constant((0., 0.)) , boundaries, bottom_mark)

        elif bottom_bc_type == "plan_rolling":
            bc_bottom = fenics.DirichletBC(VectorSpace_CG1_mesh.sub(1), fenics.Constant(0.), boundaries, bottom_mark) # able to move in X axis but not in Y axis

        elif bottom_bc_type == "curving": # (cf. Carlos Lugo's work)
            """
            if self.dimension == 3:
            
            elif self.dimension == 2:
                hpi=np.pi
                buck_perturbation_coef = 0.0 # to adjust R value 
                curving_bottom_fct = fenics.Expression(("0.0","p*sin(k*x[0])"),kx=0.0, ky=0.0, k=hpi/DISK_RADIUS, p=buck_perturbation_coef, degree=1) # bottom buckling small perturbations
                bc_bottom = fenics.DirichletBC(VectorSpace_CG1, curving_bottom_fct, self.boundaries, self.bottom_mark)
            """
            pass
            
        return bc_bottom

