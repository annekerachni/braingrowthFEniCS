import pyvista 
import numpy as np

# Before using the program, the whole mesh from the 'displacement.xdmf' file needs to be exported as .vtk 
# To do so:
# 1. open 'displacement.xdmf' wtih Paraview
# 2. Place at final simulation time
# 3. Save data in .vtk

# In order to compare with the surface element aspect ratio, compute aspect ratio analysis of the surface mesh:
# 1. open 'displacement.xdmf' wtih Paraview
# 2. Apply filter "Extract surface"
# 3. Place at final simulation time
# 4. Save data in .vtk

def tets_quality_computer(meshpath_vtk, quality_measure='aspect_ratio'): 
    """
    Options for cell quality measure:
      > tested for BrainGrowth meshes:
        - ``'aspect_ratio'``  
        - ``'radius_ratio'`` 
        - ``'jacobian'``  // (negative values for BrainGrowth)
        - ``'scaled_jacobian'``  // (negative values for BrainGrowth)
        - ``'volume'``  // (negative values for BrainGrowth)
    """

    print('\nChoosen quality measure: {}'.format(quality_measure))
    print('Computing quality of the mesh tetrahedrons...')

    # Get pyvista dataset 
    dataset = pyvista.read(meshpath_vtk) #"dataset" type: unstructured grid https://docs.pyvista.org/api/core/_autosummary/pyvista.UnstructuredGrid.html

    # Extract mesh surface
    mesh_surf = dataset.extract_surface()
    mesh_surf_edges = mesh_surf.extract_all_edges()

    # Compute the choosen cell quality measure values. 
    qualitymeasure_pyvistagrid = dataset.compute_cell_quality(quality_measure) # "qual" type: unstructured grid (size = number of mesh tetrahedrons).
    # The qual pyvista grid contains the quality measure numpy array associated to the key 'CellQuality' (qual.array_names --> returns key entries for a pyvista grid array)

    # Analyze quality measure values
    minval, maxval = qualitymeasure_pyvistagrid.get_data_range()
    qualitymeasure_array = qualitymeasure_pyvistagrid.get_array('CellQuality') # return the quality measure numpy array
    meanval = np.mean(qualitymeasure_array)
    print("{} min value = {} \n{} max value = {} \n{} mean value = {}".format(quality_measure, minval, quality_measure, maxval, quality_measure, meanval))

    return quality_measure, [minval, meanval, maxval], qualitymeasure_pyvistagrid, mesh_surf_edges
    
def surface_tets_quality(quality_measure, qualitymeasure_pyvistagrid): 
    """Display of surface tetrahedrons quality only"""
    print('\nDisplaying {} for surface mesh tetrahedrons...'.format(quality_measure))

    # Surface analysis: Plot the quality value for all faces of the mesh
    plotter = pyvista.Plotter()
    text = str(quality_measure) + " values of surface mesh tethradrons"
    plotter.add_text(text, font_size=8)
    plotter.add_mesh(qualitymeasure_pyvistagrid, show_edges=True) 
    plotter.show_grid()
    plotter.show()

    return

def percent_filtered_tets_quality(quality_measure, qualitymeasure_pyvistagrid): 
    """
    Volume analysis : filter the tetrahedrons to display basing on their quality measure value

    percent: float between 0 and 1
    For quality measures "jacobian", "scaled_jacobians", "volume": invert=True -> values below given percent are displayed
    For quality measures "aspect_ratio", "radius_ratio": invert=False -> values above given percent are displayed
    """

    # 3D analysis: Plot mesh elements (tets) for which the value of the quality measure belongs to the selected values percent  
    threshed_percent = qualitymeasure_pyvistagrid.threshold_percent(percent=0.7, invert=False)

    plotter = pyvista.Plotter()
    text = str(quality_measure) + " values of percent-threshed tethradrons"
    plotter.add_text(text, font_size=8)
    plotter.add_mesh(threshed_percent, show_edges=True) 
    plotter.show_grid()
    plotter.show()

    return

def interval_filtered_tets_quality(quality_measure, qualitymeasure_pyvistagrid, mesh_surf_edges): 
    """Volume analysis : filter the tetrahedrons to display basing on their quality measure value"""

    command = "yes"

    while command != "no":
        print('\nChoose the {} values interval in which you want to visualize mesh tetrahedrons: '.format(quality_measure))
        min_quality = input("Enter interval min value (float): ")  
        max_quality = input("Enter interval max value (float): ") 
        min_qualityval = float(min_quality)
        max_qualityval = float(max_quality)
        
        # 3D analysis: Plot mesh elements (tets) for which the value of the quality measure is part of the selected values interval
        print('Displaying the threshed tetrahedrons in the choosen interval range...')
        threshed_values = qualitymeasure_pyvistagrid.threshold((min_qualityval, max_qualityval))
        #bounded_threshed_values = threshed_values.merge(mesh_surf_edges, main_has_priority=True) 

        plotter = pyvista.Plotter()
        text = str(quality_measure) + " values of interval-threshed tethradrons"
        plotter.add_text(text, font_size=8)
        plotter.add_mesh(threshed_values, show_edges=True) 
        plotter.show_grid()
        plotter.show()

        command = input("Do you want to compute a new threshold analysis ? (yes / no): ")  

    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyse folded mesh quality ')
    
    parser.add_argument('-i', '--input', help='Path to folded mesh file (.vtk)', type=str, required=True, 
                        default='./results/growth_simulation.vtk')
    
    args = parser.parse_args()
    
    # Import the whole mesh in .vtk format
    # ------------------------------------
    meshfile_vtk = args.input

    # Computing min, mean and max mesh elements aspect ratio
    # ------------------------------------------------------
    quality_measure, [minval, meanval, maxval], qualitymeasure_pyvistagrid, mesh_surf_edges = tets_quality_computer(meshfile_vtk, quality_measure='aspect_ratio')

    # Display surface elements aspect ratio
    # -------------------------------------
    #surface_tets_quality(quality_measure, qualitymeasure_pyvistagrid)