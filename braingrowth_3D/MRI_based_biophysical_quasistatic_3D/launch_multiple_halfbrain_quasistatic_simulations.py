
import concurrent.futures
import sys, os
import argparse

sys.path.append(sys.path[0])
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

# Input parameters of reference for simulations halfbrain dHCP Right quasistatic Fgt
####################################################################################
# "H0": 2.5e-3
# "KCortex": 3000.0, "KCore": 1000.0,
# "muCortex": 300.0, "muCore": 100.0
# "alphaTAN": 3.0e-6, "alphaRAD": 0.0, "grTAN": 1.0, "grRAD": 1.0
#"alphaM": 0.2, "alphaF": 0.4
# "T0_in_GW": 21.0, "Tmax_in_GW": 29.0, "dt_in_seconds": 3600,

# 1 GW = 604800s 
# 0.1 GW = 60480 s
# 1500 s ~ 0.0025 GW 
# 3600 s (1h) ~ 0.006 GW
# 7200 s (2h) ~ 0.012 GW --> alphaTAN = 7.0e-6
# 43200 s (1/2 day) ~ 0.07 GW --> alphaTAN = 1.16e-6
# 86400 s (1 day) ~ 0.14 GW

# Function to launch series of simulations
##########################################
def run_concurrent_processes(all_simulations_to_launch, max_workers):

    parallel_simulation_launcher = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) 
    parallel_simulation_launcher.map(os.system, all_simulations_to_launch) # launch simulation (// process) / results = parallel_simulation_launcher.map(my_task, my_items)

    return

if __name__ == '__main__':
    
    

    # Define simulations
    ####################
    
    simulations_halfbrainmesh = ["""python ./biophysical_quasistatic_3D/main_halfbrain_notReoriented_Fgt_quasistatic_biophysical_DirichletZoneLargeBand.py""", 
                                 """python ./MRI_based_biophysical_quasistatic_3D/main_halfbrain_notReoriented_Fgt_quasistatic_biophysical_DirichletZoneLargeBand_CortexDelineation.py""",
                                 """python ./MRI_based_biophysical_quasistatic_3D/main_halfbrain_notReoriented_Fgt_quasistatic_biophysical_DirichletZoneLargeBand_FA.py""", 
                                 """python ./MRI_based_biophysical_quasistatic_3D/main_halfbrain_notReoriented_Fgt_quasistatic_biophysical_DirichletZoneLargeBand_FA_CortexDelineation.py""", 
                                ]
    
    #simulations_wholebrainmesh = ["""python ./biophysical_quasistatic_3D/main_halfbrain_fromWholeMesh_notReoriented_Fgt_quasistatic_biophysical_DirichletZoneLargeBand.py""", 
    #                              """python ./MRI_based_biophysical_quasistatic_3D/main_halfbrain_fromWholeMesh_notReoriented_Fgt_quasistatic_biophysical_DirichletZoneLargeBand_CortexDelineation.py""",
    #                              """python ./MRI_based_biophysical_quasistatic_3D/main_halfbrain_fromWholeMesh_notReoriented_Fgt_quasistatic_biophysical_DirichletZoneLargeBand_FA.py""", 
    #                              """python ./MRI_based_biophysical_quasistatic_3D/main_halfbrain_fromWholeMesh_notReoriented_Fgt_quasistatic_biophysical_DirichletZoneLargeBand_FA_CortexDelineation.py""", 
    #                            ]
                    
    
    # Launch simulations
    ####################

    simulations = simulations_halfbrainmesh #+ simulations_wholebrainmesh
    number_CPUs=2
    run_concurrent_processes(simulations, number_CPUs)