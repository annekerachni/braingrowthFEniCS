"""Define spatial and temporal parameters to discretize the PDE variational form"""

import fenics 

# spatial discretization of the variational form
# ----------------------------------------------
# Code structure inspired from: https://gitlab.inria.fr/mgenet/dolfin_mech/-/tree/master/dolfin_mech

class MassResForm:
    """Write part of the variational form corresponding to: ρ * <u'', v>"""


    def __init__(self, 
                 a_solution, # --> a_n_plus_1_minus_alphaM (expected acceleration between t_n and t_n+1, integrated using alpha-generalized method)
                 v_test,  
                 brain_material, 
                 integration_measure_on_mesh):
    
        self.material = brain_material
        self.dx = integration_measure_on_mesh
    
        self.residual_form = self.material.rho * fenics.dot(a_solution, v_test) * self.dx 


class DampingResForm:
    """Write part of the variational form corresponding to: γ * <u', v>"""


    def __init__(self, 
                 v_solution, # --> v_n_plus_1_minus_alphaF (expected velocity between t_n and t_n+1, integrated using alpha-generalized method)
                 v_test, 
                 brain_material, 
                 integration_measure_on_mesh):

        self.material = brain_material
        self.dx = integration_measure_on_mesh

        self.residual_form = self.material.damping * fenics.dot(v_solution, v_test) * self.dx


class StiffnessResForm:
    """Write part of the variational form corresponding to: <Te(u), ∇v> in Eulerian or <Pe(u), ∇v> in Lagrangian"""


    def __init__(self, 
                 cortex_material, 
                 core_material,
                 cortex_mark,
                 core_mark,
                 v_test, 
                 integration_measure_on_mesh, 
                 formulation="PKII"): # Cauchy; PKI; PKII 
        
        self.dx = integration_measure_on_mesh
        self.formulation = formulation

        if self.formulation == "PKII":
            self.residual_form = fenics.inner(cortex_material.compute_PKII_stress(), fenics.grad(v_test)) * self.dx(cortex_mark) + \
                                 fenics.inner(core_material.compute_PKII_stress(), fenics.grad(v_test)) * self.dx(core_mark) 
            
        """ elif self.formulation == "Cauchy": 
            self.residual_form = fenics.inner(cortex_material.compute_Cauchy_stress(), fenics.grad(v_test)) * self.dx(cortex_mark) + \
                                 fenics.inner(core_material.compute_Cauchy_stress(), fenics.grad(v_test)) * self.dx(core_mark)  """


class VolumeForcesResForm:
    """Write part of the variational form corresponding to: <f_ext, v>"""
    

    def __init__(self,
                 body_forces,
                 v_test,
                 integration_measure_on_mesh):

        self.dx = integration_measure_on_mesh

        self.residual_form = fenics.dot(body_forces, v_test) * self.dx
        

class TractionResForm:
    """Write part of the variational form corresponding to: <T, v> = < Te.n, v> = < Pe.N, v>"""
    
    
    def __init__(self,
                 v_test,
                 integration_measure_on_boundarymesh):

        self.ds = integration_measure_on_boundarymesh
        traction = fenics.Constant((0., 0., 0.))
        self.residual_form = fenics.dot(traction, v_test) * self.ds  

    """
    # Real expression?
    def __init__(self,
                 cortex_material,
                 BoundaryMesh_Nt,
                 v_test,
                 integration_measure_on_boundarymesh):

        self.ds = integration_measure_on_boundarymesh

        self.residual_form = fenics.dot( fenics.dot(cortex_material.compute_PKII_stress(), BoundaryMesh_Nt), v_test) * self.ds
    """


# SurfaceForceLoading 
# SurfacePressureLoading; SurfaceTensionLoading
# PenaltyForceResForm


# temporal discretization of the variational form
# -----------------------------------------------
class TimeIntegrator:
    # Code source: https://fenicsproject.org/olddocs/dolfin/latest/python/demos/elastodynamics/demo_elastodynamics.py.html

    def __init__(self, alphaM, alphaF, u_solution, u_old, v_old, a_old, dt):
        """Generalized-α method parameters for temporal integration of the variational form (PDE)"""

        self.alphaM = alphaM 
        self.alphaF = alphaF 
        self.gamma = 0.5 + self.alphaF - self.alphaM 
        self.beta = 0.25 * (self.gamma + 0.5)**2 

        self.u_solution = u_solution
        self.u_old = u_old
        self.v_old = v_old
        self.a_old = a_old

        self.dt = dt # timestep defined in the solver


    def update_acceleration(self, u_solution, u_old, v_old, a_old, ufl=True):
        """Compute Newmark-β method temporal unknowns (unknows u_solution, v_solution, a_solution) / acceleration"""
        # Update formula for acceleration
        # a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)

        if ufl:
            dt_ = self.dt
            beta_ = self.beta 
        else:
            dt_ = float(self.dt)
            beta_ = float(self.beta)
        
        a_solution = (u_solution - u_old - dt_*v_old) / beta_ / dt_**2 - (1 - 2*beta_) / 2 / beta_*a_old

        return a_solution


    def update_velocity(self, a_solution, v_old, a_old, ufl=True):
        """Compute Newmark-β method temporal unknowns (unknows u_solution, v_solution, a_solution) / velocity"""
        # Update formula for velocity
        # v = dt * ((1-gamma)*a0 + gamma*a) + v0
        
        if ufl:
            dt_ = self.dt
            gamma_ = self.gamma
        else:
            dt_ = float(self.dt)
            gamma_ = float(self.gamma)
        
        v_solution = v_old + dt_*( (1 - gamma_) * a_old + gamma_*a_solution ) 
            
        return v_solution


    def x_n_plus_1_minus_alpha(self, x_n, x_n_plus_1, alpha): # Xn+1−α = (1−α)*Xn+1 + α*Xn
        """
        Compute generalized-alpha method temporal unknowns.

        The generalized-α method is the temporal integration method used here. 
        --> is: an extension of the widely used Newmark-beta method in structural dynamics (if ALPHA_F, ALPHA_M = 0, 0 => get back the Newmark-beta temporal integration method)
        --> consists of: solving the dynamic evolution equation (non linear elastodynamics PDE) at intermediate time between tn and tn+1.
        --> characteristics:
        - implicit (unconditionally stable with proper coefficients)
        - high frequency dissipation -> alpha coefficients enable to control the degree of damping of high frequencies https://www.comsol.com/support/knowledgebase/1062
        - second-order accuracy (o(Dt)²)
        """
        return alpha * x_n + (1 - alpha) * x_n_plus_1  # x can be u, v or a


    def define_generalized_alpha_method_temporal_variables(self):
        """Compute the Newmark-β 'n+1' approximations for acceleration and velocity (a_n+1 and v_n+1)"""

        a_new = self.update_acceleration(self.u_solution, self.u_old, self.v_old, self.a_old, ufl=True) 
        v_new = self.update_velocity(a_new, self.v_old, self.a_old, ufl=True)

        # Compute the generalized-α 'n+1-αM' intermediate approximations for acceleration, velocity and displacement (a_n+1-αM, v_n+1-αM and u_solution_n+1-αM) 
        a_n_plus_1_minus_alphaM = self.x_n_plus_1_minus_alpha(self.a_old, a_new, self.alphaM) # aceleration_(n+1-αM) / = aceleration_(t+dt-αM) / a_t_plus_dt_minus_alphaM
        v_n_plus_1_minus_alphaF = self.x_n_plus_1_minus_alpha(self.v_old, v_new, self.alphaF) # velocity_(n+1-αF) / = velocity_(t+dt-αF)
        u_solution_n_plus_1_minus_alphaF = self.x_n_plus_1_minus_alpha(self.u_old, self.u_solution, self.alphaF) # u_solution_(n+1-αF) / = u_solution_(t+dt-αF)

        return u_solution_n_plus_1_minus_alphaF, v_n_plus_1_minus_alphaF, a_n_plus_1_minus_alphaM
    
 
    def update_fields(self):
        """Update values of displacement, velocity and acceleration at the end of each time step."""

        # Get vectors (references)
        u_vec, u0_vec  = self.u_solution.vector(), self.u_old.vector() # u_t+dt (u_solution unknown)
        v0_vec, a0_vec = self.v_old.vector(), self.a_old.vector()

        # use update functions using vector arguments
        a_vec = self.update_acceleration(u_vec, u0_vec, v0_vec, a0_vec, ufl=False) # a_t+dt (a_solution unknown)
        v_vec = self.update_velocity(a_vec, v0_vec, a0_vec, ufl=False) # v_t+dt (v_solution unknown)

        # Update (u_old, v_old, a_old <- u_solution, v_solution, a_solution)
        self.v_old.vector()[:], self.a_old.vector()[:] = v_vec, a_vec
        self.u_old.vector()[:] = self.u_solution.vector()  
