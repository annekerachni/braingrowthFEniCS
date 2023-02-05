# -*- coding: utf-8 -*-

import argparse
import fenics 
import vedo.dolfin
import matplotlib.pyplot as plt
import json

from braingrowth3D import preprocessing
from braingrowth3D.pipeline import problem
from braingrowth3D.pipeline import solver
from utils import normalize_mesh
from utils.converters import mesh_format_converters_3D

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='braingrowthFEniCS: brain growth elastodynamics 3D model')
    parser.add_argument('-i', '--input', help='Input mesh path (xml)', type=str, required=True) 
    parser.add_argument('-o', '--output', help='Output folder path', type=str, required=True)
    parser.add_argument('-n', '--normalization', help='Is normalization of the input mesh required? (required by braingrowthFEniCS)', type=bool, required=True, default=True)
    parser.add_argument('-prm', '--parameters', help='Specific model parameters file path (json)', type=str, required=True) 
    parser.add_argument('-nls', '--nonlinearsolver', help='Solver for non-linear residual form', type=str, required=False, default='newton') # Method used to solve non linear variational form -> linearization of the residual form: https://www.emse.fr/~avril/SOFT_TISS_MECH/C7/FEA%20nonlinear%20elasticity.pdf
    parser.add_argument('-ls', '--linearsolver', help='Iterative solver for linearized residual form', type=str, required=False, default='gmres') # Residual form F(u,v) = a(u,v) -l(v) to minimize => AU=L, with A nonlinear => U=A⁻¹L. Linear solver must be chosen between 'direct' or 'iterative' one. In case of 3D, 'iterative' theoritically reduces algorithm complexity to O(N².Niter) instead of O(N³).
    parser.add_argument('-p', '--preconditioner', help='Preconditioner to reduce linear solver complexity', type=str, required=False, default='sor') # 'ilu'; 'sor' 
    parser.add_argument('-v', '--visualization', help='Visualization during simulation', type=bool, required=True, default=True)
    parser.add_argument('-bf', '--bodyforces', help='Body forces expression in PDE/residual form', required=False, default=fenics.Constant((0., 0., 0.)))
    parser.add_argument('-t', '--tmax', help='Simulation end time', type=float, required=True, default=1.)
    parser.add_argument('-nst', '--nsteps', help='Number of iterations to solve problem', type=int, required=True, default=100) # For each timestep, several Newton-Raphson solving iterations are performed onto the nonlinear residual form to minimize. These split enables to approximate the residual form/system as "linear"
    parser.add_argument('-am', '--alpham', help='generalized-α method alphaM parameter', type=float, required=False, default=0.2) # 0. for Newmark-β method (energy conservative)
    parser.add_argument('-af', '--alphaf', help='generalized-α method alphaF parameter', type=float, required=False, default=0.4) # 0. for Newmark-β method (energy conservative)

    args = parser.parse_args() 
    

    # Mesh preprocessing 
    ####################
    # Get FEniCS mesh and pre-process it
    preprocessedFEniCSmesh = preprocessing.Mesh(args.input) # Get personalized mesh object from 'mesh' FEniCS object
    mesh = preprocessedFEniCSmesh.mesh
    if args.visualization == True:
        fenics.plot(mesh) 
        plt.title("input mesh")
        plt.show()  
        #vedo.dolfin.plot(mesh, wireframe=False, text='input mesh', style='paraview', axes=4).close()

    characteristics0 = preprocessedFEniCSmesh.compute_geometrical_characteristics() # n_nodes, coordinates, n_tets, n_faces_Surface, n_faces_Volume, minx, maxx, miny, maxy, minz, maxz 
    center_of_gravity0 = preprocessedFEniCSmesh.compute_center_of_gravity(characteristics0) # center of gravity
    min_mesh_spacing0, average_mesh_spacing0, max_mesh_spacing0 = preprocessedFEniCSmesh.compute_mesh_spacing()
    print('input mesh characteristics: {}'.format(characteristics0))
    print('input mesh COG = [xG0:{}, yG0:{}, zG0:{}]'.format(center_of_gravity0[0], center_of_gravity0[1], center_of_gravity0[2]))
    print("input mesh min_mesh_spacing: {:.3f} mm".format(min_mesh_spacing0))

    if args.normalization == True:
        preprocessedFEniCSmesh.normalize_mesh(characteristics0, center_of_gravity0)
        characteristics = preprocessedFEniCSmesh.compute_geometrical_characteristics() # n_nodes, coordinates, n_tets, n_faces_Surface, n_faces_Volume, minx, maxx, miny, maxy, minz, maxz 
        center_of_gravity = preprocessedFEniCSmesh.compute_center_of_gravity(characteristics) # center of gravity
        min_mesh_spacing, average_mesh_spacing, max_mesh_spacing = preprocessedFEniCSmesh.compute_mesh_spacing()
        print('normalized mesh characteristics: {}'.format(characteristics))
        print('normalized mesh COG = [xG:{}, yG:{}, zG:{}]'.format(center_of_gravity[0], center_of_gravity[1], center_of_gravity[2]))
        print("normalized min_mesh_spacing: {:.3f} mm\n".format(min_mesh_spacing))
        if args.visualization == True:
            fenics.plot(mesh) 
            plt.title("normalized mesh")
            plt.show()  
            #vedo.dolfin.plot(mesh, wireframe=False, text='normalized mesh', style='paraview', axes=4).close()


    # Simulation pipeline
    #####################

    # Import parameters
    # -----------------
    with open(args.parameters, mode="r") as json_file_object:
        parameters = json.load(json_file_object)

    dt = fenics.Constant(args.tmax/args.nsteps) # as ALPHA_M and ALPHA_F parameters for temporal discretization were choosen to provide IMPLICIT NUMERICAL SCHEME, 'dt' value is a priori arbitrary (no CFL condition).
    print('time step: ~{:5} s\n.'.format( float(dt) ))


    # Define elastodynamics problem
    # -----------------------------)
    braingrowth_problem = problem.NonLinearDynamicMechanicsProblem( preprocessedFEniCSmesh, 
                                                                    parameters['subdomains_definition_parameters'], 
                                                                    {'alphaM': args.alpham, 'alphaF': args.alphaf}, dt,
                                                                    parameters['brain_material_parameters'], 
                                                                    parameters['cortex_material_parameters'], 
                                                                    parameters['core_material_parameters'],
                                                                    parameters['dirichlet_bcs_parameters'],
                                                                    args.bodyforces,
                                                                    results_folderpath=args.output,
                                                                    visualization=args.visualization)

    # Define elastodynamics solver 
    # ----------------------------
    braingrowth_solver = solver.NonLinearDynamicMechanicsSolver( {'tmax': args.tmax, 'number_steps': args.nsteps}, 
                                                                 dt, 
                                                                 braingrowth_problem, 
                                                                 parameters['cortex_growth_parameters'], 
                                                                 parameters['core_growth_parameters'],
                                                                 results_folderpath=args.output,
                                                                 results_filename='displacements',
                                                                 results_format='.xmdf' )

    braingrowth_solver.set_solver_parameters( nonlinearsolver=args.nonlinearsolver, 
                                              linearsolver=args.linearsolver, 
                                              linearsolver_preconditioner=args.preconditioner ) 


    # Launch simulation (resolution of elastodynamics variational problem by Finite Element Method)
    # ---------------------------------------------------------------------------------------------
    braingrowth_solver.launch_simulation(visualization=args.visualization)

