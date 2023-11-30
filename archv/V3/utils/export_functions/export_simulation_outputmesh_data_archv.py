import fenics
import meshio
import numpy as np
import os 

import sys
sys.path.append(".")

import preprocessing
from metrics import brain_volume, brain_external_area

def export_resultmesh_data(output_folder_path,
                           inputmesh_vtk_file_path,
                           max_simulation_time,
                           number_of_iterations,
                           total_computational_time,
                           exportTXTfile_name): 
    """
    First import the .xdmf result-file in Paraview, place at time of interest (e.g. tF) and export data in .vtk.  
    
    Export mesh details all along the simulation, and especially at its end.
    -> n_nodes, n_tets
    -> coordinates
    -> center of gravity
    -> min, mean and max mesh spacing
    -> mesh volume
    -> mesh surface
    """

    # output folder path
    ####################
    try:
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
    except OSError:
        print ('Error: Creating directory. ' + output_folder_path) 

    # folded mesh file
    ##################
    #print("Importing the .vtk file of the brain growth simulation results (output folded mesh)...")
    # .vtk file
    mesh_vtk_file = meshio.read(inputmesh_vtk_file_path) 

    # convert .vtk into .xml
    for cell in mesh_vtk_file.cells:
        if cell.type == "tetra": 
            # write output mesh (.xml)
            mesh_xml_file = meshio.Mesh(points=mesh_vtk_file.points, cells={"tetra": cell.data})
            xml_mesh_file_name = inputmesh_vtk_file_path.split('.vtk')[0]+ ".xml"
            meshio.write(xml_mesh_file_name, mesh_xml_file)
            #print('output file (with tetra cells) well written down in XML format')

    # FEniCS mesh
    mesh = fenics.Mesh(xml_mesh_file_name)
    bmesh = fenics.BoundaryMesh(mesh, "exterior")

    # Data to export
    ################
    #print("Computing the folded mesh data to export...")

    # mesh characteristics, COG and mesh spacing
    characteristics = preprocessing.compute_geometrical_characteristics(mesh, bmesh)
    cog_coordinates = preprocessing.compute_center_of_gravity(characteristics)
    min_mesh_spacing, mean_mesh_spacing, max_mesh_spacing = preprocessing.compute_mesh_spacing(mesh)

    # volume
    mesh_volume = brain_volume.compute_mesh_volume(mesh, characteristics)

    # surface
    cortex_area = brain_external_area.compute_mesh_external_surface(bmesh)

    # aspect ratio of the whole mesh (quality metrics)
    from metrics import mesh_quality
    quality_measure, [minval, meanval, maxval], qualitymeasure_pyvistagrid, mesh_surf_edges = mesh_quality.tets_quality_computer(inputmesh_vtk_file_path, quality_measure='aspect_ratio')

    # write .txt file
    #################
    completeName = os.path.join(output_folder_path, exportTXTfile_name)
    filetxt = open(completeName, "w")

    filetxt.write('Brain Growth simulation at simu time {}, iteration {} and total computational time (.s) {}: \n'.format(max_simulation_time, number_of_iterations, int(total_computational_time)))
    filetxt.write('*******************************************************************************************\n')
    filetxt.write('\n')

    filetxt.write('>> mesh characteristics:\n{} \n'.format(characteristics))
    filetxt.write('\n')

    filetxt.write('>> center of gravity coords:{} \n'.format(cog_coordinates))
    filetxt.write('\n')

    filetxt.write('>> min mesh spacing = {} \n'.format(min_mesh_spacing))
    filetxt.write('>> mean mesh spacing = {} \n'.format(mean_mesh_spacing))
    filetxt.write('>> max mesh spacing = {} \n'.format(max_mesh_spacing))
    filetxt.write('\n')

    filetxt.write('>> cortical area = {:.2f} mm2 \n'.format(cortex_area))
    filetxt.write('\n')

    filetxt.write('>> volume = {:.2f} mm3 \n'.format(mesh_volume))
    filetxt.write('\n')

    filetxt.write('>> min mesh tetrahedron aspect ratio = {} \n'.format(minval))
    filetxt.write('>> mean mesh tetrahedron aspect ratio = {} \n'.format(meanval))
    filetxt.write('>> max mesh tetrahedron aspect ratio= {} \n'.format(maxval))
    filetxt.write('\n')

    filetxt.close()

    #print("Exporting detailed information about the output sphere mesh, folded by braingrowthFEnICS...")

    return

if __name__ == '__main__':
    # folder path where to export the data linked to the result-mesh folded by the braingrowthFEniCS simulation 
    output_folder_path = "./simulation_results/sphere_Ptot_alpha4_nsteps100_newtonabs3rel2relax1_gmres_sor/"

    # path to the result-folded mesh in .vtk format
    inputmesh_vtk_file_path = "./simulation_results/sphere_Ptot_alpha4_nsteps100_newtonabs3rel2relax1_gmres_sor/sphere_Ptot_alpha4_nsteps100_newtonabs3rel2relax1_gmres_sor_iteration76_100_volume.vtk" # vtk

    # convergence of the simulation
    computational_time = 0.76
    number_of_iterations = 76
    total_computational_time = 14400 # ~4h * 3600s
    Nsteps = 100

    # export .txt file name
    exportTXTfile_name = "final_folded_mesh_DATA_{}over{}steps.txt".format(number_of_iterations, Nsteps)  

    # export data
    export_resultmesh_data(output_folder_path,
                           inputmesh_vtk_file_path,
                           computational_time,
                           number_of_iterations,
                           total_computational_time,
                           exportTXTfile_name)