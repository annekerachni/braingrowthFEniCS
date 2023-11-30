"""This module enables to find the optimal parameters value for braingrowthFEniCS"""
# https://fenicsproject.discourse.group/t/running-a-parallel-sub-routine-within-the-main-routine-which-is-run-in-series/5581/4
# https://docs.python.org/3/library/concurrent.futures.html#module-concurrent.futures

import concurrent.futures
import json
import os
import argparse


def run_concurrent_processes(all_simulations_to_launch, max_workers):

    parallel_simulation_launcher = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) 
    parallel_simulation_launcher.map(os.system, all_simulations_to_launch) # launch simulation (// process) / results = parallel_simulation_launcher.map(my_task, my_items)

    return

def run_simulation_with_distinct_parameter_values(inputmesh_path,
                                                  input_parameters,
                                                  parameter_name,
                                                  parameter_value,
                                                  reference_output_path, # './results_nsteps100/'
                                                  parameter_value_in_outputpath): # e.g. "100"

    # update parameter
    input_parameters[parameter_name] = parameter_value # e.g. input_parameters["Nsteps"] 

    # create new output path with customed parameter name
    output_path_for_specific_value = reference_output_path.replace(parameter_value_in_outputpath, "{}".format(parameter_value))
    
    # script command line
    simulation_command = "python3 -i simulation_solverFg0.py" + \
                            " -i '{}'".format(inputmesh_path) + \
                            " -n True" + \
                            " -p '{}'".format(json.dumps(input_parameters)) + \
                            " -o '{}'".format(output_path_for_specific_value)
        
    return simulation_command


if __name__ == '__main__':
    
    """ python3 -i simulation_multiprocesses_meshsizeXtimestep.py -im './data/sphere_algoDelaunay1_tets005_refinedcoef5.xml' -p '{"H0": 0.03, "K": 100.0, "muCortex": 20.0, "muCore": 1.0, "rho": 0.01, "damping_coef": 0.5, "alphaTAN": 3, "alphaRAD": 0.0, "grTAN": 1.0, "grRAD": 1.0, "alphaM": 0.2, "alphaF": 0.4, "T0": 0.0, "Tmax": 1.0, "Nsteps": 100, "linearization_method":"newton", "linear_solver":"gmres", "preconditioner":"sor"}' -n 'Nsteps' -v 100 200 500 1000 2000 -ro './simulations/series_of_nsteps/nsteps100/' -rv 100 -c 8 """

    parser = argparse.ArgumentParser(description='multiprocessing braingrowth simulations to fine tune model parameters')
    
    parser.add_argument('-im', '--inputmeshpathlist', help='Input mesh path', type=str, required=False, 
                        default='./data/sphere_algoDelaunay1_tets005_refinedcoef5.xml')
    
    parser.add_argument('-p', '--inputparameters', help='Simulation input parameters', type=json.loads, required=False, 
                        default={"H0": 0.03, 
                                 "K": 100.0, 
                                 "muCortex": 20.0, "muCore": 1.0, 
                                 "rho": 0.01, 
                                 "damping_coef": 0.5,
                                 "alphaTAN": 3.0, "alphaRAD": 0.0, "grTAN": 1.0, "grRAD": 1.0, 
                                 "alphaM": 0.2, "alphaF": 0.4, 
                                 "T0": 0.0, "Tmax": 1.0, "Nsteps": 100,
                                 "linearization_method":"newton", "linear_solver":"gmres", "preconditioner":"sor"})
    
    parser.add_argument('-n', '--parametername', help='Name of the parameter in the input parameter previous list', type=str, required=False, 
                        default='Nsteps') 
    
    parser.add_argument('-v', '--parametervalues', help='List of nsteps values to test (in multiprocessing)', type=int, nargs='+', required=False, 
                        default=[100, 200, 500, 1000, 2000]) # 100 200 500 1000 2000 5000?
    
    parser.add_argument('-ro', '--referenceoutputfolderpath', help='Reference output path to modify', type=str, required=False, 
                        default='./simulations/series_of_nsteps/nsteps100/') 

    parser.add_argument('-rv', '--referencevaluetoreplace', help='Value of the parameter to be modified within the output path', type=str, required=False, 
                        default='100') 
    
    parser.add_argument('-c', '--numberofcpus', help='Number of CPUs', type=int, required=False, default=8)
    
    args = parser.parse_args() 

    # Multiprocessing simulation parameters
    # -------------------------------------
    input_meshes_list = args.inputmeshpathlist

    input_parameters = args.inputparameters 
    parameter_name = args.parametername
    parameter_values_to_test = args.parametervalues

    reference_output_path = args.referenceoutputfolderpath
    parameter_value_in_outputpath = args.referencevaluetoreplace

    # Cllect all simulations commands (with a given input parameter to modify)
    # -------------------------------
    all_simulations_to_launch = [] # initialize pool of simulations

    """for inputmesh_path in input_meshes_list:
        simulation_command = run_simulation_with_distinct_parameter_values(inputmesh_path, """
    
    for parameter_value in parameter_values_to_test:
        simulation_command = run_simulation_with_distinct_parameter_values( input_meshes_list,
                                                                            input_parameters,
                                                                            parameter_name,
                                                                            parameter_value,
                                                                            reference_output_path,
                                                                            parameter_value_in_outputpath)

        # save all scripts to run in parallel
        all_simulations_to_launch.append(simulation_command)


    # Run simulations in parallel on all CPUs
    # ---------------------------------------
    run_concurrent_processes(all_simulations_to_launch, args.numberofcpus)