
import concurrent.futures
import sys, os
import argparse

sys.path.append(sys.path[0]) # BrainGrowth3D
sys.path.append(os.path.dirname(sys.path[0])) # braingrowthFEniCS

# Parameters min, max to test:
##############################
# ("H0": 1.8e-3) [m]

###---parameter---###----unit----###---min value---###---max value---###
###------------------------------------------------------------------###
###    muCortex   ###    [Pa]    ###      300      ###      1000     ### ("muCore" : 100)
###       nu      ###     [-]    ###     0.45      ###      0.49     ###
###    alphaTAN   ### [(m).s⁻¹]  ###     1.8e-8    ###     1.0e-6    ###
###       dt      ###    [s]     ###      1800     ###     43200     ###


# Function to launch series of simulations
##########################################
def run_concurrent_processes(all_simulations_to_launch, max_workers):

    parallel_simulation_launcher = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) 
    parallel_simulation_launcher.map(os.system, all_simulations_to_launch) # launch simulation (// process) / results = parallel_simulation_launcher.map(my_task, my_items)

    return

if __name__ == '__main__':
    
    
    # fixed parameters
    ##################
    H0 = 1.8e-3 # [m]
    muCore = 100 # [Pa]

    # Parameters space (2⁴ combinaisons)
    ##################
    muCortex_min_max = [300, 600] # [Pa]
    nu_min_max = [0.45, 0.475, 0.49] # [-]
    alphaTAN_min_max = [5.0e-8, 7.5e-8, 1.0e-7, 2.5e-7, 5.0e-7]  #alphaTAN_min_max = [1.0e-8, 2.5e-8, 5.0e-8, 7.5e-8, 1.0e-7, 2.5e-7, 5.0e-7, 7.5e-7, 1.0e-6] # [(m).s⁻¹] # computed values from dHCP atlas (volume) are: 1.8e-8 (m)s.⁻¹ (28GW / 21GW) and 3.6e-8 (m)s.⁻¹ (36GW / 21GW) 
    dt_min_max = [3600] # dt_min_max = [1800, 3600, 43200, 86400] # [s]

    # input mesh
    ############
    mesh_inputpath = "./data/dHCP_surface/fetal_week21_left_right_merged_V2.xdmf"
    # from dHCP volume niftis: "./data/21_28_36GW/transformed_niftis_meshes/transformed_dhcp21GW_isotropic_smoothed_TaubinSmooth50_refinedWidthCoef10.xdmf" (719399 tets)
    # from dHCP surface gifti mesh: "./data/dHCP_surface/fetal_week21_left_right_merged_V2.xdmf" (455351 tets)
    
    # Define simulations
    ####################
    simulations = []

    for muCortex in muCortex_min_max:
        for nu in nu_min_max:
            for alphaTAN in alphaTAN_min_max : 
                for dt in dt_min_max:
                    
                    output_path = "./results/sensibility_analysis/dhcp21GW_halfbrain_fromWholeMesh_Fgt_quasistatic_biophysical_H{}_muCortex{}_muCore{}_nu{}_alphaTAN{}_dt{}/".format(str(H0).replace('.', "_"), str(muCortex), str(muCore), str(nu).replace('.', "_"), str(alphaTAN).replace('.', "_").replace('-', "_"), str(dt))
                    
                    simulations.append("""python BrainGrowth3D/main_wholebrain_growth.py -i """ + mesh_inputpath + """ -p '{"H0": """ + str(H0) + """, "muCortex": """+ str(muCortex) + """, "muCore": """ + str(muCore) + """, "nu": """ + str(nu) + """, "alphaTAN": """ + str(alphaTAN) + """, "alphaRAD": 0.0, "grTAN": 1.0, "grRAD": 1.0, "T0_in_GW": 21.0, "Tmax_in_GW": 42.0, "dt_in_seconds": """ + str(dt) + """, "linearization_method":"newton", "linear_solver":"gmres", "preconditioner":"sor", "newton_absolute_tolerance":1E-9, "newton_relative_tolerance":1E-6, "max_iter": 10}' -o """ + output_path) 


    # Launch simulations
    ####################
    number_CPUs=2
    run_concurrent_processes(simulations, number_CPUs)