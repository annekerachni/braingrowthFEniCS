import pyvista 
import numpy as np

# sources for Pyvista functions and code:
# https://docs.pyvista.org/examples/01-filter/mesh-quality.html#mesh-quality-example
# https://docs.pyvista.org/examples/01-filter/compute-volume.html

def compute_cells_quality_criterion(mesh_path_VTK, quality_criterion): 
    """
    Options for cell quality measure:
    - ``'aspect_ratio'``  
    - ``'radius_ratio'`` 
    - ``'jacobian'``  // (negative values in the case of braingrowth mesh)
    - ``'scaled_jacobian'``  // (negative values in the case of braingrowth mesh)
    - ``'volume'``  // (negative values in the case of braingrowth mesh)

    Returns:
    - a pyvista grid containing numpy array with quality criterion values for cells
    - mesh surface edges
    """

    print('\nChoosen quality measure: {}'.format(quality_criterion))
    print('Computing quality of the mesh tetrahedrons...')

    # Get pyvista dataset 
    dataset = pyvista.read(mesh_path_VTK) #"dataset" type: unstructured grid https://docs.pyvista.org/api/core/_autosummary/pyvista.UnstructuredGrid.html

    # Extract mesh surface
    mesh_surf = dataset.extract_surface()
    mesh_surf_edges = mesh_surf.extract_all_edges()

    # Compute the choosen quality measure values for tetrahedrons. 
    quality_criterion_cells_values_pyvistagrid = dataset.compute_cell_quality(quality_criterion) # "qual" type: unstructured grid (size = number of mesh tetrahedrons).
    # The qual pyvista grid contains a numpy array (size: number of mesh tetrahedrons) providing the quality criterion values (qual.array_names --> returns key entries 'CellQuality')

    # Analyze quality measure values
    minval, maxval = quality_criterion_cells_values_pyvistagrid.get_data_range()
    quality_criterion_cells_values_array = quality_criterion_cells_values_pyvistagrid.get_array('CellQuality') # return the quality criterions values (numpy array)
    meanval = np.mean(quality_criterion_cells_values_array)
    print("{} min value = {} \n{} max value = {} \n{} mean value = {}".format(quality_criterion, minval, quality_criterion, maxval, quality_criterion, meanval))

    return quality_criterion_cells_values_pyvistagrid, mesh_surf_edges
    
def display_surface_cells_quality_criterion(quality_criterion, quality_criterion_cells_values_pyvistagrid): 
    """
    Mesh surface quality analysis.
    Display only the surface tetrahedrons quality
    """
    print('\nDisplaying {} for surface mesh tetrahedrons...'.format(quality_criterion))

    # Surface analysis: Plot the quality value for all faces of the mesh
    plotter = pyvista.Plotter()
    text = str(quality_criterion) + " values of surface mesh tethradrons"
    plotter.add_text(text, font_size=8)
    plotter.add_mesh(quality_criterion_cells_values_pyvistagrid, show_edges=True) 
    plotter.show_grid()
    plotter.show()

    return

def display_volume_cells_filtered_by_quality_criterion_percent(quality_criterion, quality_criterion_cells_values_pyvistagrid): 
    """
    Mesh volume quality analysis.
    Filter the cells (tetrahedrons) to display basing on their quality criterion value and display them.

    - percent: float between 0 and 1
    - For quality criterions: "jacobian", "scaled_jacobians", "volume": invert=True -> values below given percent are displayed
    - For quality criterions: "aspect_ratio", "radius_ratio": invert=False -> values above given percent are displayed
    """

    # 3D analysis: Plot mesh elements (tets) for which the value of the quality measure belongs to the selected values percent  
    threshed_percent = quality_criterion_cells_values_pyvistagrid.threshold_percent(percent=0.7, invert=False)

    plotter = pyvista.Plotter()
    text = str(quality_criterion) + " values of percent-threshed tethradrons"
    plotter.add_text(text, font_size=8)
    plotter.add_mesh(threshed_percent, show_edges=True) 
    plotter.show_grid()
    plotter.show()

    return

def display_volume_cells_filtered_by_quality_criterion_interval(quality_criterion, quality_criterion_cells_values_pyvistagrid, mesh_surf_edges): 
    """
    Mesh volume quality analysis.
    Filter the cells (tetrahedrons) to display basing on their quality criterion value and display them.
    """

    command = "yes"

    while command != "no":
        print('\nChoose the {} values interval in which you want to visualize mesh tetrahedrons: '.format(quality_criterion))
        min_quality = input("Enter interval min value (float): ")  
        max_quality = input("Enter interval max value (float): ") 
        min_qualityval = float(min_quality)
        max_qualityval = float(max_quality)
        
        # 3D analysis: Plot mesh elements (tets) for which the value of the quality measure is part of the selected values interval
        print('Displaying the threshed tetrahedrons in the choosen interval range...')
        threshed_values = quality_criterion_cells_values_pyvistagrid.threshold((min_qualityval, max_qualityval))
        #bounded_threshed_values = threshed_values.merge(mesh_surf_edges, main_has_priority=True) 

        plotter = pyvista.Plotter()
        text = str(quality_criterion) + " values of interval-threshed tethradrons"
        plotter.add_text(text, font_size=8)
        plotter.add_mesh(threshed_values, show_edges=True) 
        plotter.show_grid()
        plotter.show()

        command = input("Do you want to compute a new threshold analysis? (yes / no): ")  

    return


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyse Mesh Quality')
    parser.add_argument('-i', '--input', help='Input mesh (.vtk) path ', type=str, required=False, default='/home/latim/FEniCS/Simulation/data_test/ellipsoid.vtk')
    parser.add_argument('-qc', '--qualitycriterion', help='quality criterion', type=str, required=False, default='aspect_ratio') # 'radius_ratio'; 'jacobian'; 'scaled_jacobian'; 'volume'

    args = parser.parse_args()

    input_mesh_path = args.input # mesh in .vtk
    quality_criterion = args.qualitycriterion

    quality_criterion_cells_values_pyvistagrid, mesh_surf_edges = compute_cells_quality_criterion(input_mesh_path, quality_criterion) 
    display_surface_cells_quality_criterion(quality_criterion, quality_criterion_cells_values_pyvistagrid)
    #display_volume_cells_filtered_by_quality_criterion_percent(quality_criterion, quality_criterion_cells_values_pyvistagrid)
    display_volume_cells_filtered_by_quality_criterion_interval(quality_criterion, quality_criterion_cells_values_pyvistagrid, mesh_surf_edges)


