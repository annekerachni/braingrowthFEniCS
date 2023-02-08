import fenics
import numpy as np
import vedo.dolfin
import os
import time

from braingrowth3D_growthgradient.pipeline import mappings
from utils import export_outputs


class NonLinearDynamicMechanicsSolver: 
    
    
    def __init__(self, 
                 simulation_time_parameters, 
                 dt, 
                 braingrowth_problem,
                 growth_parameters,
                 results_folderpath,
                 results_filename,
                 results_format):

        # Simulation time-stepping
        self.tmax = simulation_time_parameters['tmax'] 
        self.number_steps = simulation_time_parameters['number_steps']
        self.dt = dt
        self.times = np.linspace(0, self.tmax, self.number_steps+1) 

        self.braingrowth_problem = braingrowth_problem

        self.mappings = mappings.Mapping(self.braingrowth_problem.preprocessedFEniCSmesh, 
                                         self.braingrowth_problem.VectorSpace_CG1_mesh, 
                                         self.braingrowth_problem.VectorSpace_CG1_bmesh)

        self.growth_parameters = growth_parameters

        self.solver = fenics.NonlinearVariationalSolver(self.braingrowth_problem.nonlinearvariationalproblem) 

        self.results_folderpath = results_folderpath
        self.results_filename = results_filename
        self.results_format = results_format


    def set_solver_parameters(self, 
                              nonlinearsolver,
                              linearsolver, 
                              linearsolver_preconditioner):
        
        """
        How to choose solver parameters:
        https://www.simscale.com/blog/how-to-choose-solvers-for-fem/ 
        https://fenicsproject.org/pub/documents/book/fenics-book-2010-05-07.pdf
        https://fenics-cerfacs.readthedocs.io/en/latest/how_to/solver_choice.html
        https://fenicsproject.discourse.group/t/set-krylov-linear-solver-paramters-in-newton-solver/1070/3 
        https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1006.html#working-with-linear-solvers

        Solver for non linear systems:
        https://www.emse.fr/~avril/SOFT_TISS_MECH/C7/FEA%20nonlinear%20elasticity.pdf
        https://home.simula.no/~hpl/homepage/fenics-tutorial/release-1.0/webm/nonlinear.html
        https://fenicsproject.org/pub/documents/book/fenics-book-2010-05-07.pdf
        https://fenicsproject.discourse.group/t/setup-preconditioners-for-linear-solver-in-nonlinearvariationalsolver/2565
        https://fenicsproject.org/qa/12749/choices-preconditioners-options-newton-solver-instabilities/
        """

        nonlinearpb_prm = self.solver.parameters # https://home.simula.no/~hpl/homepage/fenics-tutorial/release-1.0/webm/nonlinear.html

        # SOLVER PARAMETERS FOR NON-LINEAR PROBLEM
        # ----------------------------------------
        nonlinearpb_prm["nonlinear_solver"] = nonlinearsolver # "newton"
        nonlinearpb_prm['newton_solver']['absolute_tolerance'] = 1E-8
        nonlinearpb_prm['newton_solver']['relative_tolerance'] = 1E-7 # 1E-7; 1E-14
        nonlinearpb_prm['newton_solver']['maximum_iterations'] = 50
        nonlinearpb_prm['newton_solver']['relaxation_parameter'] = 1.0 

        # DIRECT OR ITERATIVE SOLVER PARAMETERS FOR LINEARIZED PROBLEM 
        # ------------------------------------------------------------
        nonlinearpb_prm['newton_solver']['linear_solver'] = linearsolver # linearized problem: AU=B --> Choose between direct method U=A⁻¹B O(N³) (e.g. 'mumps') or iterative/Krylov subspaces method U=A⁻¹B~(b + Ab + A²b + ...) O(num_iter * N²) (e.g. 'gmres' for non-symmetric problem , 'cg') to compute A⁻¹. See http://hplgit.github.io/fenics-tutorial/pub/html/._ftut1012.html; https://fenicsproject.org/pub/tutorial/html/._ftut1017.html#ftut:app:solver:prec
        nonlinearpb_prm['newton_solver']['preconditioner'] = linearsolver_preconditioner # preconditioner enables to reduce computational cost. https://computationalmechanics.in/preconditioners/
        nonlinearpb_prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-9 
        nonlinearpb_prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-7 
        nonlinearpb_prm['newton_solver']['krylov_solver']['maximum_iterations'] = 1000 # number of iterations with Krylov subspace method


    def initalize_deformation_export(self):
        try:
            if not os.path.exists(self.results_folderpath):
                os.makedirs(self.results_folderpath)
        except OSError:
            print ('Error: Creating directory. ' + self.results_folderpath)

        results_file_path = str(self.results_folderpath) + str(self.results_filename) + str(self.results_format)
        self.displacements_xdmf = fenics.XDMFFile(results_file_path) # e.g. './results/displacements.xdmf' 
        self.displacements_xdmf.parameters["flush_output"] = True
        self.displacements_xdmf.parameters["functions_share_mesh"] = True
        self.displacements_xdmf.parameters["rewrite_function_mesh"] = True # https://fenicsproject.discourse.group/t/how-to-write-a-xdmf-file-which-has-mesh-static-over-time/708

    
    def initialize_parameters_export(self, parameterFEniCSFunction):
        try:
            if not os.path.exists(self.results_folderpath):
                os.makedirs(self.results_folderpath)
        except OSError:
            print ('Error: Creating directory. ' + self.results_folderpath)

        results_file_path = str(self.results_folderpath) + str(parameterFEniCSFunction) + str(self.results_format)
        parameter_xdmf = fenics.XDMFFile(results_file_path) # e.g. './results/H.xdmf' 
        parameter_xdmf.parameters["flush_output"] = True
        parameter_xdmf.parameters["functions_share_mesh"] = True
        parameter_xdmf.parameters["rewrite_function_mesh"] = True

        return parameter_xdmf


    def launch_simulation(self, visualization):

        self.initalize_deformation_export()
        self.H_xdmf = self.initialize_parameters_export("cortical_thickness")
        self.d2s_xdmf = self.initialize_parameters_export("d2s")
        self.gm_xdmf = self.initialize_parameters_export("gm")
        self.mu_xdmf = self.initialize_parameters_export("mu")
        
        print("solving nonlinear elastodynamics variational problem...")
        for (i, self.dt ) in enumerate(np.diff(self.times)):

            t_current = self.times[i]
            print('Time: {0:.3f}'.format(t_current)) # from 0 to 0+(NSTEPS-1)*dt 
            step_to_be_applied = i+1 # step 1 to step NSTEPS (step 1 applied onto time 0 to get time 0+dt; ... step NSTEPS applied onto time 0+(NSTEPS-1)*dt to get final time 0+(NSTEPS)*dt=TMAX)
            t_reached = self.times[i+1] # from 0+dt to 0+(NSTEPS)*dt=TMAX 

            # Update cortical thickness H (FEniCS map on ScalarFunctionSpace)
            # ----------------------------
            #self.braingrowth_problem.cortical_thickness.t = self.dt # update 'cortical_thickness' fenics expression
            #H = self.braingrowth_problem.cortical_thickness # update 'gm' fenics expression
            self.braingrowth_problem.cortical_thickness.vector()[:] += 0.01 * self.dt 
            """ export_outputs.export_PVDfile(self.results_folderpath, 'H', self.braingrowth_problem.cortical_thickness) """

            # Update d2s (FEniCS map on ScalarFunctionSpace)
            # ----------
            self.braingrowth_problem.distance_to_brainsurface.assign( self.braingrowth_problem.compute_distance_to_brainsurface() ) # update 'gm' fenics expression
            """ export_outputs.export_PVDfile(self.results_folderpath, 'd2s', self.braingrowth_problem.distance_to_brainsurface) """
            
            # gm and mu automatically updated (FEniCS map on ScalarFunctionSpace)
            # -------------------------------            
            """ export_outputs.export_PVDfile(self.results_folderpath, 'gm', self.braingrowth_problem.gm) """ # automatically updated with H and d2s
            """ export_outputs.export_PVDfile(self.results_folderpath, 'mu', self.braingrowth_problem.mu) """ # Check update of 'mu' (mu=f(gm))

            # Update growth coefficients of Cortex growth tensor 
            # --------------------------------------------------
            # Define growth tensor at a second stage of brain development (e.g. pure tangential cortex development) / TODO: could be updated directly through the growth expression in the 'problem.py' (cf. material = f(self.gm))
            for dof in self.braingrowth_problem.ScalarSpace_CG1_mesh.dofmap().dofs():
                self.braingrowth_problem.growthtensor.dg_TAN.vector()[dof] = self.growth_parameters['gr_TAN'] * self.growth_parameters['alpha_TAN'] * self.braingrowth_problem.gm.vector()[dof] * self.dt  # Further experiments: try gr_TAN(X,t) and alpha_TAN(x,t)
                self.braingrowth_problem.growthtensor.dg_RAD.vector()[dof] = self.growth_parameters['gr_RAD'] * self.growth_parameters['alpha_RAD'] * self.dt  


            # update normal vectors at brainsurface boundary
            # ----------------------------------------------
            self.braingrowth_problem.BoundaryMesh_Nt.assign( self.braingrowth_problem.growthtensor.compute_topboundary_normals(self.braingrowth_problem.mesh, self.braingrowth_problem.ds, self.braingrowth_problem.VectorSpace_CG1_mesh, self.braingrowth_problem.brainsurface_mark) ) # BoundaryMesh_Nt
            """ if visualization == True:
                vedo.dolfin.plot(self.braingrowth_problem.BoundaryMesh_Nt, mode="mesh arrows", text="Step {} / {}: \nMesh at time {} / tmax={}\nNormals to BrainSurface".format(step_to_be_applied, self.number_steps, t_current, self.tmax), style="matplotlib", axes=4, azimuth=0, interactive=False, viewup=["0.","0.","1."], scale=0.05).clear()
                time.sleep(2.) """


            # Compute new mesh coordinates
            # ----------------------------
            mesh_vertex_coords = self.braingrowth_problem.mesh.coordinates()  # VERTEX indexation 
            bmeshB_vertex_coords = self.braingrowth_problem.brainsurface_bmesh.coordinates() 


            # update the projected normal vectors at all mesh points  
            # ------------------------------------------------------                                                                            
            self.braingrowth_problem.Mesh_Nt.assign( self.braingrowth_problem.growthtensor.compute_mesh_projected_normals(mesh_vertex_coords, bmeshB_vertex_coords, self.mappings.vertexB_2_dofinVref_mapping, self.mappings.vertex2dofs_V, self.braingrowth_problem.Mesh_Nt, self.braingrowth_problem.BoundaryMesh_Nt) )
            """ if visualization == True:
                vedo.dolfin.plot(self.braingrowth_problem.Mesh_Nt, mode="arrows", wireframe=True, text="Step {} / {}: \nMesh at time {} / tmax={}\nProjected Normals onto Mesh".format(step_to_be_applied, self.number_steps, t_current, self.tmax), style="matplotlib", axes=4, camera=dict(pos=(-6., -12., -6.)), scale=0.03, interactive=False).clear()
                time.sleep(2.) """


            # Solve for new (Lagrangian) displacement   
            # ---------------------------------------
            self.solver.solve()  


            # Save solution to XDMF format
            # ----------------------------
            # Save deformation
            self.displacements_xdmf.write(self.braingrowth_problem.u_solution, t_current) # display current mesh at t_current and displacement 'u_solution' to be applied
            #displacements_xdmf.write_mesh(fenics_mesh, t_current)
            #displacements_xdmf.write_meshtags(facet_tags)
            if visualization == True:
                vedo.dolfin.plot(self.braingrowth_problem.u_solution, mode='displace', text="Step {} / {}:\nMesh at time {} / tmax={}\nDisplacement to be applied".format(step_to_be_applied, self.number_steps, t_current, self.tmax), style='paraview', axes=4, camera=dict(pos=(-6., -12., -6.)), interactive=False).clear() # plot options https://vedo.embl.es/autodocs/content/vedo/dolfin.html
                #time.sleep(2.) 

            # Save parameters evolution
            self.H_xdmf.write(self.braingrowth_problem.cortical_thickness, t_current) 
            self.d2s_xdmf.write(self.braingrowth_problem.distance_to_brainsurface, t_current) 
            self.gm_xdmf.write(self.braingrowth_problem.gm, t_current) 
            self.mu_xdmf.write(self.braingrowth_problem.mu, t_current)   


            # move the mesh (to be able to compute new Normals) https://fenicsproject.discourse.group/t/how-to-compute-display-normals-to-one-deforming-boundary/9656/6
            # -------------
            fenics.ALE.move(self.braingrowth_problem.mesh, self.braingrowth_problem.u_solution)
            # fenics_bmesh.assign( BoundaryMesh(fenics_mesh, "exterior") ) #?
            # need to update subdomains / boundaries #?


            # Update old fields with new quantities
            # -------------------------------------
            self.braingrowth_problem.timeintegrator.update_fields()
            # if visualization == True:
                # vedo.dolfin.plot(self.braingrowth_problem.u_solution, mode='displace', text="After Step {} / {}:\nMesh at time {} / tmax={}\nDisplacement that was applied during the step".format(step_to_be_applied, self.number_steps, t_reached, self.tmax), style='paraview', axes=4, camera=dict(pos=(-6., -12., -6.)), interactive=False).clear() # plot options https://vedo.embl.es/autodocs/content/vedo/dolfin.html


        # Plot final deformed mesh
        if visualization == True:
            vedo.dolfin.plot(self.braingrowth_problem.mesh, mode='mesh', text="Mesh at final time {}".format(t_reached), style='paraview', axes=4, camera=dict(pos=(-6., -12., -6.)), interactive=True).clear()  

        