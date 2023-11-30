
import os

def export_maximum_time_iterations(output_folder_path,
                                   exportTXTfile_name,
                                   T0, Tmax, nsteps,
                                   max_simulation_time,
                                   max_iteration,
                                   total_computational_time):

    # output folder path
    try:
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
    except OSError:
        print ('Error: Creating directory. ' + output_folder_path) 

    # output file 
    completeName = os.path.join(output_folder_path, exportTXTfile_name)

    # write .txt file
    filetxt = open(completeName, "w")

    filetxt.write('>> initial T0, Tmax, nsteps :\n{} {} {}\n'.format(T0, Tmax, nsteps))
    filetxt.write('\n')

    filetxt.write('>> maximum simulation time reached : {} \n'.format(max_simulation_time))
    filetxt.write('\n')

    filetxt.write('>> maximum iteration reached : {} \n'.format(max_iteration))
    filetxt.write('\n')
    
    filetxt.write('>> total computational time reached (.s) : {} \n'.format(total_computational_time))
    filetxt.write('\n')

    filetxt.close()

    #print("Exporting max iteration and time reached by the simulation (resolution convergence data)...")

    return