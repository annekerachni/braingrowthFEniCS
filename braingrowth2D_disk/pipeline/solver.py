import fenics
import numpy as np
import vedo.dolfin
import os
import time

from braingrowth2D_disk.pipeline import mappings


class NonLinearDynamicMechanicsSolver: 
    
    
    def __init__(self, 
                 simulation_time_parameters, 
                 dt, 
                 braingrowth_problem,
                 cortex_growth_parameters, core_growth_parameters,
                 results_folderpath,
                 results_filename,
                 results_format):

        # Simulation time-stepping
        self.tmax = simulation_time_parameters['tmax'] 
        self.number_steps = simulation_time_parameters['number_steps']
        self.dt = dt
        self.times = np.linspace(0, self.tmax, self.number_steps+1) 

        self.braingrowth_problem = braingrowth_problem

        self.mappings = mappings.Mapping(self.braingrowth_problem.mesh, 
                                         self.braingrowth_problem.brainsurface_bmesh,
                                         self.braingrowth_problem.VectorSpace_CG1_mesh, 
                                         self.braingrowth_problem.VectorSpace_CG1_bmesh)

        self.cortex_growth_parameters = cortex_growth_parameters
        self.core_growth_parameters = core_growth_parameters

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
        https://home.simula.no/~hpl/homepage/fenics-tutorial/release-1.0/webm/nonlinear.html
        https://fenicsproject.org/pub/documents/book/fenics-book-2010-05-07.pdf
        https://fenicsproject.discourse.group/t/setup-preconditioners-for-linear-solver-in-nonlinearvariationalsolver/2565
        https://fenicsproject.org/qa/12749/choices-preconditioners-options-newton-solver-instabilities/
        """

        prm = self.solver.parameters 

        # SOLVER PARAMETERS FOR NON-LINEAR PROBLEM
        # ----------------------------------------
        prm["nonlinear_solver"] = nonlinearsolver # "newton"
        prm['newton_solver']['absolute_tolerance'] = 1E-8
        prm['newton_solver']['relative_tolerance'] = 1E-14 # 1E-7
        prm['newton_solver']['maximum_iterations'] = 25
        prm['newton_solver']['relaxation_parameter'] = 1.0 

        # DIRECT OR ITERATIVE SOLVER PARAMETERS FOR LINEARIZED PROBLEM 
        # ------------------------------------------------------------
        prm['newton_solver']['linear_solver'] = linearsolver # linear problem: AU=B --> Choose between direct method U=A⁻¹B O(N³) (e.g. 'mumps') or iterative/Krylov subspaces method U=A⁻¹B~(b + Ab + A²b + ...) O(num_iter * N²) (e.g. 'gmres' for non-symmetric problem , 'cg') to compute A⁻¹. See http://hplgit.github.io/fenics-tutorial/pub/html/._ftut1012.html; https://fenicsproject.org/pub/tutorial/html/._ftut1017.html#ftut:app:solver:prec
        prm['newton_solver']['preconditioner'] = linearsolver_preconditioner # "Incomplete LU factorization" --> "It conditions a given problem into a form that is more suitable for numerical solving methods". Preconditionner enables to reduce computationnal cost. https://computationalmechanics.in/preconditioners/
        
        if linearsolver_preconditioner != None: # i.e. if linear solver is iterative (Krylov subspace method ("PETSc Krylov solver") --> it is required to specify a preconditionner). Else, linear solver is direct (e.g. mumps), well suitable for low range meshes (low n_nodes).
            prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-9
            prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-7 
            prm['newton_solver']['krylov_solver']['maximum_iterations'] = 100 # number of iterations with Krylov subspace method


    def export_results(self):
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


    def launch_simulation(self, visualization):

        self.export_results()

        # Solve iteratively variational problem (elastodynamics)
        print("solving nonlinear elastodynamics variational problem...")
        for (i, self.dt ) in enumerate(np.diff(self.times)):

            t_current = self.times[i]
            max_decimals = str(self.dt)[::-1].find('.') # for time display convenience
            t_current_2_display = round(t_current, max_decimals) # for time display convenience
            print('Time: {0:.3f}'.format(t_current)) # from 0 to 0+(NSTEPS-1)*dt 
            step_to_be_applied = i+1 # step 1 to step NSTEPS (step 1 applied onto time 0 to get time 0+dt; ... step NSTEPS applied onto time 0+(NSTEPS-1)*dt to get final time 0+(NSTEPS)*dt=TMAX)
            t_reached = self.times[i+1] # from 0+dt to 0+(NSTEPS)*dt=TMAX 


            # Update growth coefficients of Cortex growth tensor 
            # --------------------------------------------------
            # Define growth tensor at a first stage of brain development (cf. Carlos Lugo)
            """
            #if t < 0.1*TMAX:
            # Core
            # Increasing source-shape bottom plan (sinusoid) + tangential (0.1) & radial (1.0) growth 
            #buck_perturbation_coef += 0.01 * dt
            #cb.p = buck_perturbation_coef # introduce small buckling perturbations at the first beginning of Core tangential growth

            # Update Growth Tensor with differential 'dg' 
            # -------------------------------------------
            # Update growth coefficients of Cortex growth tensor 
            dg_cortex_TAN.dgTANCortex = gr_cortex_TAN * alpha_cortex_TAN * dt 
            dg_cortex_RAD.dgRADCortex = gr_cortex_RAD * alpha_cortex_RAD * dt 

            # Update growth coefficients of Core growth tensor 
            dg_core_TAN.dgTANCore = gr_core_TAN * alpha_core_TAN * dt  
            dg_core_RAD.dgRADCore = gr_core_RAD * alpha_core_RAD * dt  
            """
            
            # Define growth tensor at a second stage of brain development (e.g. pure tangential cortex development)
            self.braingrowth_problem.growthtensor.dg_cortex_TAN.dgTANCortex = self.cortex_growth_parameters['gr_cortex_TAN'] * self.cortex_growth_parameters['alpha_cortex_TAN'] * self.dt  # Further experiments: try gr_TAN(X,t) and alpha_TAN(x,t)
            self.braingrowth_problem.growthtensor.dg_cortex_RAD.dgRADCortex = self.cortex_growth_parameters['gr_cortex_RAD'] * self.cortex_growth_parameters['alpha_cortex_RAD']  * self.dt  

            """ 
            dg_core_TAN.dgTANCore = self.core_growth_parameters['gr_core_TAN'] * self.core_growth_parameters['alpha_core_TAN'] * self.dt  
            dg_core_RAD.dgRADCore = self.core_growth_parameters['gr_core_RAD'] * self.core_growth_parameters['alpha_core_RAD'] * self.dt  
            """


            # update normal vectors at brainsurface boundary
            # ----------------------------------------------
            self.braingrowth_problem.BoundaryMesh_Nt.assign( self.braingrowth_problem.growthtensor.compute_topboundary_normals(self.braingrowth_problem.mesh, self.braingrowth_problem.ds, self.braingrowth_problem.VectorSpace_CG1_mesh, self.braingrowth_problem.brainsurface_mark) ) # BoundaryMesh_Nt
            if visualization == True:
                vedo.dolfin.plot(self.braingrowth_problem.BoundaryMesh_Nt, mode="mesh arrows", text="Step {} / {}: \nMesh at time {} / tmax={}\nNormals to BrainSurface".format(step_to_be_applied, self.number_steps, t_current_2_display, self.tmax), style="matplotlib", axes=4, azimuth=0, interactive=False, viewup=["0.","0.","1."], scale=0.05).clear()
                time.sleep(2.)


            # Compute new mesh coordinates
            # ----------------------------
            mesh_vertex_coords = self.braingrowth_problem.mesh.coordinates()  # VERTEX indexation 
            brainsurface_bmesh_vertex_coords = self.braingrowth_problem.brainsurface_bmesh.coordinates() 


            # update the projected normal vectors at all mesh points 
            # ------------------------------------------------------                                                                            
            self.braingrowth_problem.Mesh_Nt.assign( self.braingrowth_problem.growthtensor.compute_mesh_projected_normals(mesh_vertex_coords, brainsurface_bmesh_vertex_coords, self.mappings.vertexB_2_dofinVref_mapping, self.mappings.vertex2dofs_V, self.braingrowth_problem.Mesh_Nt, self.braingrowth_problem.BoundaryMesh_Nt) )
            if visualization == True:
                vedo.dolfin.plot(self.braingrowth_problem.Mesh_Nt, mode="arrows", wireframe=True, text="Step {} / {}: \nMesh at time {} / tmax={}\nProjected Normals onto Mesh".format(step_to_be_applied, self.number_steps, t_current_2_display, self.tmax), style="matplotlib", axes=4, azimuth=0, viewup=['0.','0.','1.'], scale=0.03, interactive=False).clear()
                time.sleep(2.)


            # Solve for new (Lagrangian) displacement   
            # ---------------------------------------
            self.solver.solve()  


            # Save solution to XDMF format
            self.displacements_xdmf.write(self.braingrowth_problem.u_solution, t_current) # display current mesh at t_current and displacement 'u_solution' to be applied
            #displacements_xdmf.write_mesh(fenics_mesh, t_current)
            #displacements_xdmf.write_meshtags(facet_tags)
            if visualization == True:
                vedo.dolfin.plot(self.braingrowth_problem.u_solution, mode='displace', text="Step {} / {}:\nMesh at time {} / tmax={}\nDisplacement to be applied".format(step_to_be_applied, self.number_steps, t_current_2_display, self.tmax), style='paraview', axes=4,viewup=['0.','0.','1.'], interactive=False).clear() # plot options https://vedo.embl.es/autodocs/content/vedo/dolfin.html
                time.sleep(2.)   


            # move the mesh (to be able to compute new Normals) https://fenicsproject.discourse.group/t/how-to-compute-display-normals-to-one-deforming-boundary/9656/6
            # -------------
            fenics.ALE.move(self.braingrowth_problem.mesh, self.braingrowth_problem.u_solution)
            # fenics_bmesh.assign( BoundaryMesh(fenics_mesh, "exterior") ) #?
            # need to update subdomains / boundaries #?


            # Update old fields with new quantities
            # -------------------------------------
            self.braingrowth_problem.timeintegrator.update_fields()
            #vedo.dolfin.plot(self.braingrowth_problem.u_solution, mode='displace', text="After Step {} / {}:\nMesh at time {} / tmax={}\nDisplacement that was applied during the step".format(step_to_be_applied, self.number_steps, t_reached, self.tmax), style='paraview', axes=4, viewup=['0.','0.','1.'], interactive=False).clear() # plot options https://vedo.embl.es/autodocs/content/vedo/dolfin.html


        # Plot final deformed mesh
        if visualization == True:
            vedo.dolfin.plot(self.braingrowth_problem.mesh, mode='mesh', text="Mesh at final time {}".format(t_reached), style='paraview', axes=4, viewup=['0.','0.','1.'], interactive=True).clear()  

        # if 3D: camera=dict(pos=(-4., -8., -4.))
        # if 2D: viewup=['0.','0.','1.']