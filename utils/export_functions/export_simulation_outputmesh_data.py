import fenics
import meshio
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))  

from braingrowth_3D.phenomenological_dynamic_3D.FEM_biomechanical_model.preprocessing import compute_geometrical_characteristics, compute_center_of_gravity, compute_mesh_spacing
from metrics import brain_volume, brain_external_area

def export_resultmesh_data(output_folder_path,
                           inputmesh_vtk_file_path,
                           numerical_time,
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
    characteristics = compute_geometrical_characteristics(mesh, bmesh)
    cog_coordinates = compute_center_of_gravity(characteristics)
    min_mesh_spacing, mean_mesh_spacing, max_mesh_spacing = compute_mesh_spacing(mesh)

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

    filetxt.write('Brain Growth simulation at simu time {}, iteration {} and total computational time at tmax (.s) {}: \n'.format(numerical_time, number_of_iterations, int(total_computational_time)))
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

    import argparse
    
    parser = argparse.ArgumentParser(description='Export characteristical data from the folded mesh (by braingrowthFEniCS)')

    #parser.add_argument('-i', '--inputVTKfoldedmeshpath', help='Input mesh (.vtk) path', type=list, required=False, 
    #default=["./simulations/series_of_nsteps/nsteps2000/sphere_Ptot_alpha3_nsteps2000_newtonabs8rel7relax1_gmres_sor_iteration1700over2000.vtk",
    #         ]) 
    
    parser.add_argument('-t', '--numericaltimes', help='numerical time (between 0. and 1.)', type=int, nargs='+', required=True, 
    default=[0.4, 0.6, 0.8, 0.85, 1.0]) 
    
    #parser.add_argument('-it', '--step', help='step reached', type=int, required=True, 
    #default=1700) 
    
    parser.add_argument('-n', '--nsteps', help='nsteps goal', type=int, required=True, 
    default=100) 
    
    parser.add_argument('-o', '--output', help='Path to output folder', type=str, required=True, 
                        default='results')

    args = parser.parse_args()

    # folder path where to export the data linked to the result-mesh folded by the braingrowthFEniCS simulation 
    output_folder_path = args.output

    for numerical_time in args.numericaltimes:
        # path to the result-folded mesh in .vtk format
        inputmesh_vtk_file_path = "./results/brainmesh_nsteps{}_time{}_volume.vtk".format(args.nsteps, str(numerical_time).split('.')[0] + "_" + str(numerical_time).split('.')[-1])

        # convergence of the simulation
        number_of_iterations = 0 #args.step
        Nsteps = args.nsteps
        total_computational_time = 0 #args.totalcomputationaltime

        # export .txt file name
        exportTXTfile_name = "foldedspheremesh_nsteps{}_DATA_at_time{}.txt".format(args.nsteps, str(numerical_time).split('.')[0] + "_" + str(numerical_time).split('.')[-1])  

        # export data
        export_resultmesh_data(output_folder_path,
                               inputmesh_vtk_file_path,
                               numerical_time,
                               number_of_iterations,
                               total_computational_time,
                               exportTXTfile_name)