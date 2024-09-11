
import concurrent.futures
import sys, os
import argparse

sys.path.append(sys.path[0])
#sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

# Function to launch series of simulations
##########################################
def run_concurrent_processes(all_simulations_to_launch, max_workers):

    parallel_simulation_launcher = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) 
    parallel_simulation_launcher.map(os.system, all_simulations_to_launch) # launch simulation (// process) / results = parallel_simulation_launcher.map(my_task, my_items)

    return

if __name__ == '__main__':
    
    

    # Define simulations
    ####################
    
    simulations_wholebrainmesh_H0cst = [
                                        """python main_wholebrain_growth_alphaTANcst_H0cst.py""",
                                        """python main_wholebrain_growth_alphaTAN_1_over_1plusFA_H0cst.py""", 
                                        """python main_wholebrain_growth_alphaTAN_linearFA_H0cst.py""", 
                                        """python main_wholebrain_growth_alphaTAN_1_over_1plusFA_expminust_H0cst.py""", 
                                        """python main_wholebrain_growth_alphaTAN_linearFA_expminust_H0cst.py"""
                                        ]
    
    simulations_wholebrainmesh_H0segmentation = [
                                                """python main_wholebrain_growth_alphaTANcst_H0segmentation.py""",
                                                """python main_wholebrain_growth_alphaTAN_1_over_1plusFA_H0segmentation.py""", 
                                                """python main_wholebrain_growth_alphaTAN_linearFA_H0segmentation.py""", 
                                                """python main_wholebrain_growth_alphaTAN_FA_1_over_1plusFA_expminust_H0segmentation.py""", 
                                                """python main_wholebrain_growth_alphaTAN_linearFA_expminust_H0segmentation.py"""
                                                ]
                    
    
    # Launch simulations
    ####################

    number_CPUs=1 # @Francois
    run_concurrent_processes(simulations_wholebrainmesh_H0cst, number_CPUs)