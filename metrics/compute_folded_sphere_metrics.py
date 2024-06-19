import sys, os
import argparse 
import fenics
import pyvista
#import meshio
import numpy as np

sys.path.append(sys.path[0])

from metrics import brain_external_area, brain_volume, gyrification_index, laplace_beltrami_operator_decomposition
from utils.converters import convert_meshformats

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='braingrowthFEniCS sphere Fgt: launch series of simulations to test parameters')

    """
    parser.add_argument('-i', '--initialmesh', help='Initial mesh path (xdmf)', type=str, required=False, 
                        default='./data/sphere/meshlab_netgen/sphere_257Ktets_20Kfaces_refined5.xdmf')  
    """                          

    parser.add_argument('-p', '--parametername', help='Input parameter name', type=str, required=False, 
                        default='H0')  
    
    parser.add_argument('-pv', '--parametervalue', help='Value of the parameter to investigate', type=float, required=False, 
                        default=0.06)  

    parser.add_argument('-s', '--simulationresult', help='Simulation results file path (xdmf)', type=str, required=False, 
                        default='./results_sphere_Fgt/H0/growth_simulation.xdmf')
    
    parser.add_argument('-t', '--time', help='Time to consider of the folded mesh', type=float, required=False, 
                        default=0.65)
    
    parser.add_argument('-o', '--output', help='Output folder path', type=str, required=False, 
                        default='metrics/sphere_Fgt_simulations/H0/0_06/')  
    
    parser.add_argument('-m', '--metricsfilename', help='Output file name where to write metrics (.txt)', type=str, required=False, 
                        default='metrics_H006.json')  # metrics.txt
  
    
    args = parser.parse_args() 

    ##########
    # OUTPUT #
    ##########
    output_folder_path = args.output
    try:
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
    except OSError:
        print ('Error: Creating directory. ' + output_folder_path) 
    

    ##########
    # MESHES #
    ##########
    
    """
    https://fenicsproject.discourse.group/t/time-series-data-in-pyvista-xdmfreader/11126
    https://stackoverflow.com/questions/73490842/find-cells-from-vertices-in-pyvista-polydata-mesh
    """
    
    simulation_results_file_path = args.simulationresult
    reader = pyvista.get_reader(simulation_results_file_path)
    
    # initial mesh
    ##############
    # pyvista
    # -------
    reader.set_active_time_value(0.0)

    initial_mesh_pyvista = reader.read() # grid
    coordinates_0 = np.float64(initial_mesh_pyvista.points) # float32 by default 
    tets_0 = initial_mesh_pyvista.cells_dict[10]

    # write .xdmf
    convert_meshformats.mesh_to_xml_xdmf(os.path.join(output_folder_path, "initial_mesh.xdmf"), coordinates_0, tets_0) 

    # initial boundary mesh
    initial_boundarymesh_pyvista = initial_mesh_pyvista.extract_surface()
    

    # FEniCS 
    # ------
    # FEniCS mesh
    initial_mesh_FEniCS = fenics.Mesh()
    with fenics.XDMFFile(os.path.join(output_folder_path, "initial_mesh.xdmf")) as infile:
        infile.read(initial_mesh_FEniCS)
    
    # FEniCS boundarymesh
    initial_bmesh_FEniCS = fenics.BoundaryMesh(initial_mesh_FEniCS, "exterior") 
    faces_coordinates_0 = initial_bmesh_FEniCS.coordinates()
    faces_0 = initial_bmesh_FEniCS.cells()
    
    #initial_mesh.point_cell_ids
    
    # folded mesh (growth_simulation.xdmf at tf)
    ##############    
    # pyvista
    # -------
    folded_time_to_analyze = args.time
    reader.set_active_time_value(folded_time_to_analyze)

    folded_mesh_pyvista = reader.read() # grid
    coordinates = np.float64(folded_mesh_pyvista.points) # float32 by default
    tets = folded_mesh_pyvista.cells_dict[10]

    # write .xdmf
    convert_meshformats.mesh_to_xml_xdmf(os.path.join(output_folder_path, "folded_mesh.xdmf"), coordinates, tets) 
    
    # initial boundary mesh
    folded_boundarymesh_pyvista = folded_mesh_pyvista.extract_surface()

    # FEniCS
    # ------
    # FEniCS mesh
    folded_mesh_FEniCS = fenics.Mesh()
    with fenics.XDMFFile(os.path.join(output_folder_path, "folded_mesh.xdmf")) as infile:
        infile.read(folded_mesh_FEniCS)
    
    # FEniCS boundarymesh
    folded_bmesh_FEniCS = fenics.BoundaryMesh(folded_mesh_FEniCS, "exterior") 
    faces_coordinates = folded_bmesh_FEniCS.coordinates()
    faces = folded_bmesh_FEniCS.cells()

    # write boundary mesh
    #meshio.write(os.path.join(output_folder_path, "./folded_boundarymesh.xdmf"), meshio.Mesh(points=faces_coordinates, cells=[("triangle", faces)]))
    
    # folded_mesh_pyvista.point_cell_ids
    # folded_mesh_pyvista.point_data 
    #displacement = folded_mesh_pyvista.point_data["Displacement"] # folded_mesh.point_data["dgTAN"], etc.
    
    ###########
    # METRICS #
    ###########
    
    # cortical surface of folded mesh
    #################################
    cortex_area = brain_external_area.compute_mesh_external_surface(folded_bmesh_FEniCS) # mm²
    
    # cortical volume
    ##################
    volume = brain_volume.compute_mesh_volume(folded_mesh_FEniCS) # mm³

    # GI
    ####
    folded_mesh_FEniCS = gyrification_index.rescale_folded_mesh_to_initial_smooth_mesh(initial_mesh_FEniCS, folded_mesh_FEniCS)
    GI = gyrification_index.compute_gyrification_index(initial_bmesh_FEniCS, folded_bmesh_FEniCS)

    # principal wavelength
    ######################
    # Number of modes used to decompose the folded mesh curvature on
    num_modes = 1500

    # Initial mesh .stl and .gi
    # -------------------------
    # generate .stl initial sphere mesh / See to compute boundarymesh normals and be sure they are well oriented before saving .stl https://github.com/pyvista/pyvista/discussions/4944
    initial_boundarymesh_pyvista.compute_normals(flip_normals=True, inplace=True) # compute face normals / See https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.polydatafilters.compute_normals#pyvista.PolyDataFilters.compute_normals
    face_normals_0 = initial_boundarymesh_pyvista["Normals"] # tag normals. Can also be computed by: normals = initial_boundarymesh_pyvista.face_normals # to get face normals

    initial_mesh_stl_path = os.path.join(output_folder_path, "initial_mesh.stl")
    pyvista.save_meshio(initial_mesh_stl_path, initial_boundarymesh_pyvista)
    #convert_meshformats.mesh_to_stl(initial_mesh_stl_path, faces_coordinates_0, faces_0) # create .stl mesh file TODO: check faces are not inverted
    
    # generate .gii initial sphere mesh
    initial_mesh_gii_path = os.path.join(output_folder_path, "initial_mesh.gii")
    convert_meshformats.stl_to_gii(initial_mesh_stl_path, initial_mesh_gii_path) # create .gii mesh file

    #Folded mesh .stl and .gi
    # -----------------------
    # generate .stl folded sphere mesh / See to compute boundarymesh normals and be sure they are well oriented before saving .stl https://github.com/pyvista/pyvista/discussions/4944
    folded_boundarymesh_pyvista.compute_normals(flip_normals=True, inplace=True)  # compute face normals / See https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.polydatafilters.compute_normals#pyvista.PolyDataFilters.compute_normals
    face_normals = folded_boundarymesh_pyvista["Normals"] # tag normals. Can also be computed by: normals = initial_boundarymesh_pyvista.face_normals # to get face normals

    folded_mesh_stl_path = os.path.join(output_folder_path, "folded_mesh.stl")
    pyvista.save_meshio(folded_mesh_stl_path, folded_boundarymesh_pyvista)
    #convert_meshformats.mesh_to_stl(folded_mesh_stl_path, faces_coordinates, faces) # create .stl mesh file TODO: check faces are not inverted
    
    # generate .gii folded sphere mesh
    folded_mesh_gii_path = os.path.join(output_folder_path, "folded_mesh.gii")
    convert_meshformats.stl_to_gii(folded_mesh_stl_path, folded_mesh_gii_path) # create .gii mesh file
    
    # Save geometries paths
    Geometries = {}
    Geometries['initial_sphere'] = initial_mesh_gii_path
    Geometries['folded_sphere'] = folded_mesh_gii_path

    # compute LBO decomposition of the folded mesh curvature (at tf)
    visualization = False
    wavelenghts_array, power_spectrum_array = laplace_beltrami_operator_decomposition.analyze_mesh_folding_pattern(Geometries, num_modes, visualization) # len(wavelenghts_array) = len(power_spectrum_array) = 1499, eigen mode 0 is removed.
        
    # extract principal wavelength of the foled mesh
    wavelenghts_array_without_mode0 = wavelenghts_array[1:]
    power_spectrum_array_without_mode0 = power_spectrum_array[1:]
    principal_wavelength = [ wavelenghts_array_without_mode0[i] for i in range(len(power_spectrum_array_without_mode0)) if power_spectrum_array_without_mode0[i] == np.max(power_spectrum_array_without_mode0) ]
    
    ##########################
    # EXPORT METRICS (.json) #
    ##########################
    import json
    import io
    # https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file

    output_file_name = args.metricsfilename
    output_path = os.path.join(output_folder_path, output_file_name)

    # Define data
    data = {'parameters': {args.parametername: args.parametervalue},
            't': folded_time_to_analyze,
            'cortical_area': cortex_area,
            'volume': volume,
            'gyrification_index': GI,
            'principal_wavelength': np.float64(principal_wavelength[0]),
            'max_power_spectrum': np.max(power_spectrum_array_without_mode0)
            }

    # Write JSON file
    with io.open(output_path, 'w', encoding='utf8') as outfile:
        str_ = json.dumps(data,
                          indent=4, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(str_)

    """
    # Read JSON file
    with open(output_path) as data_file:
        data_loaded = json.load(data_file)

    print(data == data_loaded)
    """

    """
    #########################
    # EXPORT METRICS (.txt) #
    #########################
    output_file_name = args.metricsfilename
    output_path = os.path.join(output_folder_path, output_file_name)
        
    filetxt = open(output_path, "w")
    
    filetxt.write('#########################\n')
    filetxt.write('# FOLDED SPHERE METRICS #\n')
    filetxt.write('#########################\n')

    filetxt.write('>> tf: {}\n'.format(folded_time_to_analyze))
    filetxt.write('\n')

    filetxt.write('>> cortical area: {} mm²\n'.format(cortex_area))
    filetxt.write('\n')
    
    filetxt.write('>> volume: {} mm³\n'.format(volume))
    filetxt.write('\n')

    filetxt.write('>> GI: {}\n'.format(GI))
    filetxt.write('\n')
    """
    """    
    filetxt.write('LBO Analysis of the curvature of the folded sphere mesh\n')
    filetxt.write('# -----------------------------------------------------\n')

    filetxt.write('>> num_modes :{}\n'.format(num_modes))
    filetxt.write('\n')
    
    #filetxt.write('>> wavelength (1499) : {} \n'.format(wavelenghts_array))
    filetxt.write('>> wavelength mode1 = {} mm \n'.format(wavelenghts_array_without_mode0[0]))
    filetxt.write('>> wavelength mode2 = {} mm \n'.format(wavelenghts_array_without_mode0[1]))
    filetxt.write('>> wavelength mode2 = {} mm \n'.format(wavelenghts_array_without_mode0[2]))
    filetxt.write('>> wavelength mode3 = {} mm \n'.format(wavelenghts_array_without_mode0[3]))
    filetxt.write('>> wavelength last mode = {} mm \n'.format(wavelenghts_array_without_mode0[-1]))
    filetxt.write('\n')
    
    #filetxt.write('>> power spectrum values (1499) : {} \n'.format(power_spectrum_array))
    filetxt.write('>> power spectrum value mode1 = {} mm \n'.format(power_spectrum_array_without_mode0[0]))
    filetxt.write('>> power spectrum value mode2 = {} mm \n'.format(power_spectrum_array_without_mode0[1]))
    filetxt.write('>> power spectrum value mode3 = {} mm \n'.format(power_spectrum_array_without_mode0[2]))
    filetxt.write('>> power spectrum value mode4 = {} mm \n'.format(power_spectrum_array_without_mode0[3]))
    filetxt.write('>> power spectrum value last mode = {} mm \n'.format(power_spectrum_array_without_mode0[-1]))
    filetxt.write('\n')
    """
    """
    filetxt.write('>> max power spectrum value = {} (-) \n'.format(np.max(power_spectrum_array_without_mode0)))
    filetxt.write('>> principal (associated) wavelength = {} mm \n'.format(principal_wavelength[0]))
    filetxt.write('\n')
    
    filetxt.close()
    """