# -*- coding: utf-8 -*-

import argparse
import fenics 
import vedo.dolfin
import matplotlib.pyplot as plt
import json

from braingrowth3D_growthgradient import preprocessing
from braingrowth3D_growthgradient.pipeline import problem
from braingrowth3D_growthgradient.pipeline import solver

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='braingrowthFEniCS: brain growth elastodynamics 3D model')
    parser.add_argument('-i', '--input', help='Input mesh path (xml)', type=str, required=False, default='/home/latim/FEniCS/Github3/data_test/ellipsoid.xml') 
    parser.add_argument('-n', '--normalization', help='Is normalization of the input mesh required? (required by braingrowthFEniCS)', type=bool, required=False, default=True)
    parser.add_argument('-o', '--output', help='Output folder path', type=str, required=False, default='/home/latim/FEniCS/Github3/results/')
    parser.add_argument('-prm', '--simulationparameters', help='Specific model parameters file path (json)', type=str, required=False, default='/home/latim/FEniCS/Github3/braingrowthFEniCS/braingrowth3D_growthgradient/parameters/parameters_braingrowth.json') # same parameters for all meshes
    parser.add_argument('-v', '--visualization', help='Visualization during simulation', type=bool, required=False, default=False)

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
    with open(args.simulationparameters, mode="r") as json_file_object:
        parameters = json.load(json_file_object)

    dt = fenics.Constant(parameters['simulation_time_parameters']['tmax']/parameters['simulation_time_parameters']['number_steps']) # as ALPHA_M and ALPHA_F parameters for temporal discretization were choosen to provide IMPLICIT NUMERICAL SCHEME, 'dt' value is a priori arbitrary (no CFL condition).
    print('time step: ~{:5} s\n.'.format( float(dt) ))


    # Define elastodynamics problem
    # -----------------------------)
    braingrowth_problem = problem.NonLinearDynamicMechanicsProblem( preprocessedFEniCSmesh,
                                                                    parameters['subdomains_definition_parameters'], 
                                                                    parameters['temporal_discretization_parameters'], dt,
                                                                    parameters['brain_material_parameters'], 
                                                                    parameters['cortex_material_parameters'], 
                                                                    parameters['core_material_parameters'],
                                                                    parameters['dirichlet_bcs_parameters'],
                                                                    fenics.Constant(parameters['body_forces']),
                                                                    results_folderpath=args.output,
                                                                    visualization=args.visualization)

    # Define elastodynamics solver 
    # ----------------------------
    braingrowth_solver = solver.NonLinearDynamicMechanicsSolver( parameters['simulation_time_parameters'], 
                                                                 dt, 
                                                                 braingrowth_problem, 
                                                                 parameters['growth_parameters'], 
                                                                 results_folderpath=args.output,
                                                                 results_filename='displacements',
                                                                 results_format='.xmdf' )

    braingrowth_solver.set_solver_parameters( nonlinearsolver=parameters['solver_parameters']['nonlinearsolver'], 
                                              linearsolver=parameters['solver_parameters']['linearsolver'], 
                                              linearsolver_preconditioner=parameters['solver_parameters']['preconditioner'] ) 


    # Launch simulation (resolution of elastodynamics variational problem by Finite Element Method)
    # ---------------------------------------------------------------------------------------------
    braingrowth_solver.launch_simulation(visualization=args.visualization)

