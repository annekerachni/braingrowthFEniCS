
import concurrent.futures
import sys, os
import argparse

sys.path.append(sys.path[0]) # BrainGrowth3D
sys.path.append(os.path.dirname(sys.path[0])) # braingrowthFEniCS

# Function to launch series of simulations
##########################################
def run_concurrent_processes(all_simulations_to_launch, max_workers):

    parallel_simulation_launcher = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) 
    parallel_simulation_launcher.map(os.system, all_simulations_to_launch) # launch simulation (// process) / results = parallel_simulation_launcher.map(my_task, my_items)

    return

if __name__ == '__main__':
    
    
    # fixed parameters
    ##################
    muCore = 300 # [Pa]

    # Parameters space (2⁴ combinaisons)
    ##################
    H0_values = [0.8e-3, 1.5e-3, 2.25e-3, 6e-3] # [m] --> 2.6%, 5%, 7.5%, 21% of brain radius (30 mm)
    muCortex_values = [6000] # [Pa]
    nu_min_max = [0.45] # [-]
    alphaTAN_min_max = [2.0e-7] # [s⁻¹]
    dt_min_max = [43200] # [s] --> [1800, 3600, 43200, 86400]

    # input mesh
    ############
    mesh_inputpath = "./data/dHCPsurface21_pial_129960faces_456026tets_RAS.xdmf" # brain mesh in millimeters, in RAS+ orientation

    # Define simulations
    ####################
    simulations = []

    for H0 in H0_values:
        for muCortex in muCortex_values:
            for nu in nu_min_max:
                for alphaTAN in alphaTAN_min_max : 
                    for dt in dt_min_max:
                    
                        output_path = "./results/brain_growth_sensititivy_analysis_H0/dHCPsurface_fetalweek21pialsurf129960faces456026tets_H{}_muCortex{}_muCore{}_nu{}_alphaTAN{}_dt{}/".format(str(H0).replace('.', "_"), str(muCortex), str(muCore), str(nu).replace('.', "_"), str(alphaTAN).replace('.', "_").replace('-', "_"), str(dt))
                    
                        simulations.append("""python BrainGrowth3D/main_wholebrain_growth.py -i """ + mesh_inputpath + """ -c True -p '{"H0": """ + str(H0) + """, "muCortex": """+ str(muCortex) + """, "muCore": """ + str(muCore) + """, "nu": """ + str(nu) + """, "alphaTAN": """ + str(alphaTAN) + """, "alphaRAD": 0.0, "grTAN": 1.0, "grRAD": 1.0, "epsilon_n": 5e5, "T0_in_GW": 21.0, "Tmax_in_GW": 36.0, "dt_in_seconds": """ + str(dt) + """, "linearization_method":"newton", "newton_absolute_tolerance":1E-9, "newton_relative_tolerance":1E-6, "max_iter": 15, "linear_solver":"mumps"}' -o """ + output_path) 

    # Launch simulations
    ####################
    max_workers=1 # --> 'max_workers' equals 1 meaning only one per one simulation is launched, but the linear solver mumps is still parallelized on the number of CPUs set in 'main_wholebrain_growth.py' (e.g. 4)
    run_concurrent_processes(simulations, max_workers)