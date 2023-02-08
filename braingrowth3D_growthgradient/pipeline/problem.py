"""Build the FEM mechanics problem to solve with FEniCS"""

import fenics 
import matplotlib.pyplot as plt

from braingrowth3D_growthgradient.pipeline import boundaries, subdomains, growth, kinematics, material, numericalscheme
from utils import export_outputs


class NonLinearDynamicMechanicsProblem:
    # code structure inspired from: https://github.com/ElsevierSoftwareX/SOFTX_2018_73/blob/632609a5eea1def5d079d62e818aabbdfd3727dd/fenicsmechanics/solidmechanics.py

    def __init__(self,
                 preprocessedFEniCSmesh, 
                 subdomains_definition_parameters, # defined arbitrary here, does not depend on the mesh element size.
                 temporal_discretization_parameters, dt,
                 brain_material_parameters, 
                 cortex_material_parameters, 
                 core_material_parameters,
                 dirichlet_bcs_parameters,
                 body_forces,
                 results_folderpath,
                 visualization):

        self.results_folderpath = results_folderpath

        self.preprocessedFEniCSmesh = preprocessedFEniCSmesh
        self.brainsurface_bmesh = preprocessedFEniCSmesh.brainsurface_bmesh

        # Input mesh
        self.set_FEniCS_mesh()
        self.get_brainsurface_bmesh_bbtree()

        # Boundaries
        self.initialize_boundaries()
        self.define_and_mark_brainsurface_boundary()
        """ self.define_and_mark_other_boundaries() """
        self.export_marked_boundaries()

        # Cortex & Core Subdomains 
        """ self.get_brainsurface_bmesh() """
        # self.initalize_subdomains()
        # self.define_and_mark_subdomains(subdomains_definition_parameters)
        # self.analyse_subdomains_submeshes(visualization)
        # self.export_marked_subdomains()

        # ds, dx for variational form integration
        self.set_integration_measures()

        # Compiler parameters
        self.ffc_parameters()

        # FEM approximation function basis (u) and testing functions (v)
        self.define_function_spaces()
        self.define_functions()

        #Define gradient function for gradient-Growth "Fg" and gradient stiffness "mu"  
        self.define_cortical_thickness(subdomains_definition_parameters)
        self.compute_distance_to_brainsurface()
        self.gradient_function()

        # Discretized temporal domain
        self.get_temporal_variables(temporal_discretization_parameters, dt)

        # Growth-induceed deformations law
        self.set_growth_tensor()
        self.set_kinematics()
        self.define_material(brain_material_parameters, cortex_material_parameters, core_material_parameters)

        # Boundary conditions
        self.define_dirichlet_bcs(dirichlet_bcs_parameters)

        # F(u,v) = a(u,v) - L(v) -> 0
        self.define_residual_form(body_forces)
        self.build_nonlinear_variational_problem()
    

    def set_FEniCS_mesh(self): # mesh: geometry.Mesh(FEniCS-format mesh, time), bmesh already computed from mesh object
        self.mesh = self.preprocessedFEniCSmesh.mesh 
    
    """
    def get_brainsurface_bmesh(self): # subpart of the boundary mesh representing the brain surface ("partial" for rectangle, halfdisk or quarterdisk meshes; "whole" for disk, brain meshes)
        self.brainsurface_bmesh = fenics.BoundaryMesh(self.preprocessedFEniCSmesh.mesh, "exterior") # if mesh representing "whole" brain (2D or 3D) 
    """

    def get_brainsurface_bmesh_bbtree(self): # bounding box tree build from the brain surface mesh
        self.brainsurface_bmesh_bbtree = fenics.BoundingBoxTree()
        self.brainsurface_bmesh_bbtree.build(self.brainsurface_bmesh)


    def initialize_boundaries(self):
        self.boundaries = fenics.MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)  
        self.boundaries.set_all(100)


    def define_and_mark_brainsurface_boundary(self): # e.g. BrainSurface() for disk mesh; Top() for rectangle mesh
        self.brainsurface = boundaries.BrainSurface(self.brainsurface_bmesh_bbtree, self.boundaries)
        self.brainsurface_mark = self.brainsurface.brainsurface_mark
        self.boundaries = self.brainsurface.mark_brainsurface_boundary()


    """ 
    def define_and_mark_other_boundaries(self):
        self.left = boundaries.Left(self.preprocessedFEniCSmesh, self.boundaries)
        self.left_mark = self.left.left_mark
        self.boundaries = self.left.mark_left_boundary()

        self.right = boundaries.Right(self.preprocessedFEniCSmesh, self.boundaries)
        self.right_mark = self.right.right_mark
        self.boundaries = self.right.mark_right_boundary()

        self.bottom = boundaries.Bottom(self.preprocessedFEniCSmesh, self.boundaries)
        self.bottom_mark = self.bottom.bottom_mark
        self.boundaries = self.bottom.mark_bottom_boundary() 

        self.front

        self.back
    """

    
    def export_marked_boundaries(self):
        export_outputs.export_PVDfile(self.results_folderpath, 'boundaries', self.boundaries)


    """ def initalize_subdomains(self):
        self.subdomains = fenics.MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
        self.subdomains.set_all(1) 


    def define_and_mark_subdomains(self, subdomains_definition_parameters):
        self.cortex = subdomains.Cortex(self.brainsurface_bmesh_bbtree, subdomains_definition_parameters['cortical_thickness'], self.subdomains)
        self.cortex_mark = self.cortex.cortex_mark
        self.subdomains = self.cortex.mark_cortex()

        self.core = subdomains.Core(self.brainsurface_bmesh_bbtree, subdomains_definition_parameters['cortical_thickness'], self.subdomains)
        self.core_mark = self.core.core_mark
        self.subdomains = self.core.mark_core()

    
    def analyse_subdomains_submeshes(self, visualization):
        # Sanity check: Be sure that cortex delimitation is correctly set. Otherwise, following simulation will not make sense! 
        submesh_cortex = fenics.SubMesh(self.mesh, self.subdomains, self.cortex_mark)
        submesh_core = fenics.SubMesh(self.mesh, self.subdomains, self.core_mark)
        if visualization == True:
            fenics.plot(submesh_cortex, color='blue') 
            #fenics.plot(submesh_core, color='green') 
            plt.title("Cortex submesh analysis: \nif submesh presents holes -> increase 'cortical_thickness' value")
            plt.show()  


    def export_marked_subdomains(self):
        export_outputs.export_PVDfile(self.results_folderpath, 'subdomains', self.subdomains) """


    def set_integration_measures(self):
        #self.dx = fenics.Measure("dx", domain=self.mesh, subdomain_data=self.subdomains) # integration measure on mesh
        self.ds = fenics.Measure("ds", domain=self.mesh, subdomain_data=self.boundaries) # integration measure on boundary mesh


    def ffc_parameters(self):
        """
        The FEniCS Form Compiler FFC is a compiler for finite element variational forms, 
        translating high-level mathematical descriptions of variational forms into efficient low-level C++ code for finite element assembly.
        
        Parameters used by FEniCS to assemble the residual form. See https://fenicsproject.org/qa/377/optimize-parameter-in-ffc/
        
        N.B. FFC parameters should be defined before Functions

        More information: https://fenics.readthedocs.io/projects/ffc/en/latest/_modules/ffc/parameters.html; https://link.springer.com/content/pdf/10.1007/978-3-642-23099-8_11.pdf
        """

        fenics.parameters["form_compiler"]["cpp_optimize"] = True
        fenics.parameters["form_compiler"]["representation"]='uflacs' # parameter that optimize time and memory allocation for hyperelasticity problem https://fenicsproject.org/qa/12857/form-compilation-error-not-enough-memory/; https://fenicsproject.org/pub/workshops/fenics13/slides/Alnaes.pdf
        fenics.parameters["form_compiler"]["quadrature_degree"] = 7 # None: automatic (not working). Working well: 7 --> 'quadrature_degree' : used to compute integrals of variational form
        fenics.parameters["allow_extrapolation"] = True # required to compute Boundary mesh / https://fenicsproject.org/qa/7607/working-with-boundarymesh/
        fenics.parameters['std_out_all_processes'] = False # ?
        #parameters['form_compiler'].add('eliminate_zeros', False)
        #parameters['form_compiler']['representation'] = 'quadrature'
        #parameters['form_compiler']['optimize'] = True


    def define_function_spaces(self, family="CG", degree=1): # "CG"="Lagrange" 
        """Define Lagrangian Function Spaces (DOF indexation) for numerical FEM resolution"""

        # Scalar Function Spaces
        self.ScalarSpace_CG1_mesh = fenics.FunctionSpace(self.mesh, family, degree) 

        # Vector Function Spaces
        print("creating Lagrange FEM function spaces for volume mesh and boundary mesh...")
        self.VectorSpace_CG1_mesh = fenics.VectorFunctionSpace(self.mesh, family, degree) 
        self.VectorSpace_CG1_bmesh = fenics.VectorFunctionSpace(self.brainsurface_bmesh, family, degree) # used only in compute_new_Nt


    def define_functions(self, non_zero_solution_guess=False):
        print("creating test and solution FEM functions...")
        
        # Scalar functions
        # ----------------
        self.cortical_thickness = fenics.Function(self.ScalarSpace_CG1_mesh, name="H")
        self.distance_to_brainsurface = fenics.Function(self.ScalarSpace_CG1_mesh, name="d2s")
        self.gm = fenics.Function(self.ScalarSpace_CG1_mesh, name="gm") 
        #self.mu = fenics.Function(self.ScalarSpace_CG1_mesh, name="mu")         

        # Vector functions
        # ----------------
        # unknown function
        if non_zero_solution_guess == False:
            self.u_solution = fenics.Function(self.VectorSpace_CG1_mesh, name="Displacement") # allocates an instance for the displacement. Enables to store values then. --> initial guess is set at (0., 0., 0.) for all x of the Space
        else:
            pass
            #u_solution = interpolate(Expression(("0.", "0.", "0.2"), degree=0), VectorSpace_CG1)
            #u_solution = interpolate(Expression(("0.", "0.", "k * ( sin(x[0]/XMAX) + sin(x[1]/YMAX) )"), degree=1, k=0.2, XMAX=XMAX, YMAX=YMAX), VectorSpace_CG1) #specifying a non-zero initial guess https://fenicsproject.org/qa/9536/how-to-set-initial-guess-for-nonlinear-variational-problem/
        
        # test function
        self.v_test = fenics.TestFunction(self.VectorSpace_CG1_mesh)
        
        # initialize all the variables, derived from the unknown function, required for temporal integration of the residual form (PDE) and therefore for dynamic resolution of the elastodynamics problem.
        self.u_old = fenics.Function(self.VectorSpace_CG1_mesh) # Fields from previous time step (displacement, velocity, acceleration)
        self.v_old = fenics.Function(self.VectorSpace_CG1_mesh)
        self.a_old = fenics.Function(self.VectorSpace_CG1_mesh) 

        # Initialize adaptative normals Nt used to compute adaptative Growth Tensor
        print("initializing Normals to brainsurface boundary nodes and at all mesh nodes...")
        self.BoundaryMesh_Nt = fenics.Function(self.VectorSpace_CG1_mesh)
        self.Mesh_Nt = fenics.Function(self.VectorSpace_CG1_mesh)

    
    def define_cortical_thickness(self, subdomains_definition_parameters):
        #self.cortical_thickness = fenics.Expression('H + 0.01*t', H=subdomains_definition_parameters['cortical_thickness_0'], t=0., degree=1)
        #self.cortical_thickness = fenics.Expression('H', H=subdomains_definition_parameters['cortical_thickness_0']) # suppose pure tangential growth
        self.cortical_thickness.vector()[:] = subdomains_definition_parameters['cortical_thickness_0']

    def compute_distance_to_brainsurface(self):
        # https://fenicsproject.discourse.group/t/compute-distance-to-surface-for-all-mesh-elements/9315/2
        # https://fenicsproject.org/qa/9763/method-compute_closest_point/

        # for point, coords in self.mesh.coordinates(): # vertex
        #     distance_closest_node_brainsurfacebdry, idex_closest_node_brainsurfacebdry = self.brainsurface_bmesh_bbtree.query(point, k=1) # 1st closest neighbour on the boundary
       
        # for point in self.mesh.geometry.x:
        #     _, distance_closest_node_brainsurfacebdry = self.brainsurface_bmesh_bbtree.compute_closest_point(fenics.Point(*x))

        vertex2dofs_S = fenics.vertex_to_dof_map(self.ScalarSpace_CG1_mesh)
        
        for idx, x in enumerate(self.mesh.coordinates()): # point: idx (x[0]), coords (x[1])
            _, distance_to_brainsurface_x = self.brainsurface_bmesh_bbtree.compute_closest_entity(fenics.Point(*x)) # TODO: need to recompute self.distance_Point_2_closestNodeBrainsurfacebBdry?
            self.distance_to_brainsurface.vector()[vertex2dofs_S[idx]] = distance_to_brainsurface_x
        
        return self.distance_to_brainsurface
        
        #self.distance_to_brainsurface = fenics.project(self.distance_to_brainsurface, self.ScalarSpace_CG1_mesh)

        """ _, self.distance_to_brainsurface = self.brainsurface_bmesh_bbtree.compute_closest_entity(fenics.Point(*self.mesh.coordinates()))
        self.distance_to_brainsurface = fenics.project(self.distance_to_brainsurface, self.ScalarSpace_CG1_mesh) """


    def gradient_function(self): # compared to BrainGrowth, gm is a nodal function and not a tetrahedron function.

        # self.gm = fenics.Expression('1 / (1 + exp( 10 * (y/H - 1) ) )', 
        #                             H=self.cortical_thickness, 
        #                             y=self.distance_to_brainsurface,
        #                             degree=1) # CPP type
        # y=self.distance_to_brainsurface
        # H=self.cortical_thickness

        #self.gm = 1 / ( 1 + fenics.exp(10 * (self.distance_to_brainsurface/self.cortical_thickness - 1)) ) # FEniCS/Python instance
        #self.gm = fenics.project(self.gm, self.ScalarSpace_CG1_mesh)
        for dof in self.ScalarSpace_CG1_mesh.dofmap().dofs():
            self.gm.vector()[dof] = 1 / ( 1 + fenics.exp(10 * (self.distance_to_brainsurface.vector()[dof]/self.cortical_thickness.vector()[dof] - 1)) )
        
        
    def get_temporal_variables(self, temporal_discretization_parameters, dt):
        print("defining temporal numerical variables (displacement, velocity and acceleration) according to generalized-α method...")
        
        self.timeintegrator = numericalscheme.TimeIntegrator(temporal_discretization_parameters['alphaM'], 
                                                             temporal_discretization_parameters['alphaF'], 
                                                             self.u_solution, 
                                                             self.u_old, 
                                                             self.v_old,  
                                                             self.a_old, 
                                                             dt)
        
        # Get the expressions of the acceleration, velocity and displacement (a_n+1-αM, v_n+1-αM and u_solution_n+1-αM) with generalized-α method
        self.u_solution_n_plus_1_minus_alphaF, self.v_n_plus_1_minus_alphaF, self.a_n_plus_1_minus_alphaM = self.timeintegrator.define_generalized_alpha_method_temporal_variables()   


    def set_growth_tensor(self):
        
        print("initializing growth tensor components...")
        self.growthtensor = growth.GrowthTensor(self.ScalarSpace_CG1_mesh) # continuous gradient growth tensor

        print("defining differential (gradient), tangential and adaptative growth tensor...") # Lagrangian: the reference is the new mesh obtained after deformation at time t-dt
        self.Fg = self.growthtensor.define_adaptative_growth_tensor(self.u_solution_n_plus_1_minus_alphaF, self.Mesh_Nt) # Fg_cortex: Function(TensorSpaceCG1) by definition


    def set_kinematics(self):

        self.kinematics = kinematics.Kinematics(self.u_solution, self.Fg)


    def define_material(self, brain_material_parameters, cortex_material_parameters, core_material_parameters):

        if brain_material_parameters['constitutive_model'] == 'neo_hookean':

            self.brain_material = material.Material(brain_material_parameters)

            self.material = material.NeoHookeanElasticMaterial(brain_material_parameters, # self.material refers indifferently to Cortex / Core without specifying marks. Function gradient-growth defines by itself the Cortex/Core material limit.
                                                               self.kinematics,
                                                               cortex_material_parameters, 
                                                               core_material_parameters, 
                                                               self.gm)
            
            self.mu = self.material.compute_gradientbased_stiffnesses()
            self.mu = fenics.project(self.material.mu, self.ScalarSpace_CG1_mesh) # self.MATERIAL.mu (mu modified by class) --> make full FEniCS _cpp_object (otherwise 'Sum')
            
        

    def define_dirichlet_bcs(self, dirichlet_bcs_parameters):
        """Define Dirichlet boundary conditions"""
        self.bcs = boundaries.DirichletBoundaryConditions(self.mesh.geometry().dim(),
                                                          self.VectorSpace_CG1_mesh,
                                                          self.boundaries,
                                                          dirichlet_bcs_parameters['consider_brainsurface_bc_TrueorFalse'], self.brainsurface_mark, dirichlet_bcs_parameters['brainsurface_bc']).bcs # brainsurface bcs
                                                        

    def define_residual_form(self, body_forces):
        """ Write the Residual Form (F) of the variational/weak form of the braingrowth PDE. The aim of the nonlinear (e.g. Newton's method) solver is to minimize F after each step iterations (linear approximation)."""
        # -> Variational form of braingrowth PDE:
        #       - dynamic case: ρ * <u'', v> + γ * <u', v> + <σ(u), ∇v> = <f, v> + <T, v>  https://comet-fenics.readthedocs.io/en/latest/demo/elastodynamics/demo_elastodynamics.py.html#erl2002
        #       - static case: <σ(u), ∇v> = <f, v> + <T, v>
        # -> Numerical scheme obtained after FEM spatial integration and alpha-generalized method temporal integration of the previous variational form: [M]{a_n+1−αm} + [C]{v_n+1−αf} + [K]{u_n+1−αf} = {F(t_n+1−αf)} 
        #       - alpha-generalized method: FEniCS solver will provide solution to the elastic problem at the intermediate time between t_n and t_n+1 (according to αM and αF)
        # -> At each simulation step, solver converging <=> F-->0 
        
        print("defining the residual form (to minimize at each step iterations) from the PDE variational form...")
    
        self.F = ( numericalscheme.MassResForm(self.a_n_plus_1_minus_alphaM, self.v_test, self.brain_material).residual_form  
                 + numericalscheme.DampingResForm(self.v_n_plus_1_minus_alphaF, self.v_test, self.brain_material).residual_form
                 + numericalscheme.StiffnessResForm(self.material, self.v_test).residual_form 
                 - numericalscheme.VolumeForcesResForm(body_forces, self.v_test).residual_form 
                 - numericalscheme.TractionResForm(self.v_test, self.ds).residual_form )
                #- numericalscheme.TractionResForm(self.cortex_material, self.BoundaryMesh_Nt, self.v_test, self.ds).residual_form

        """ 
        self.F = ( mass_resform 
                 + damping_resform  
                 + stiffness_resform 
                 - volumeforces_resform 
                 - traction_resform ) """


    def build_nonlinear_variational_problem(self):
        # See https://www.emse.fr/~avril/SOFT_TISS_MECH/C7/FEA%20nonlinear%20elasticity.pdf
        self.jacobian = fenics.derivative(self.F, self.u_solution) # we want to find u that minimize F(u) = 0 (F(u): total potential energy of the system), where F is the residual form of the PDE => dF(u)/du 
        self.nonlinearvariationalproblem = fenics.NonlinearVariationalProblem(self.F, self.u_solution, self.bcs, self.jacobian)        