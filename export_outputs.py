import fenics
from PIL import Image
import glob

# File-writing functions
########################
def export_XMLfile(output_folderpath, name, mesh):

    fenics.File( str(output_folderpath + name + '.xml') ) << mesh
    print('tetra_{}.xml was written'.format(name))

    return 

def export_PVDfile(output_folderpath, name, geometry_entity):

    fenics.File( str(output_folderpath + name + '.pvd') ).write(geometry_entity)
    print('{}.pvd was written'.format(name))

    return 

def export_XDMFfile(output_folderpath, value):
    # Set name and options for elastodynamics solution XDMF file export
    results_file_path = str(output_folderpath) + str(value) + ".xdmf"
    xdmf_file = fenics.XDMFFile(results_file_path)
    xdmf_file.parameters["flush_output"] = True
    xdmf_file.parameters["functions_share_mesh"] = True
    xdmf_file.parameters["rewrite_function_mesh"] = False

    return xdmf_file

def make_gif(screenshotsPNG_path, gifname):
    """
    Make gif from sery of (PNG) simulation results screenshots.
    Code source: https://pythonprogramming.altervista.org/png-to-gif/?doing_wp_cron=1659355206.1907370090484619140625
    """
    # Create the frames
    images = []
    
    imgs = glob.glob( str(screenshotsPNG_path + '/*.png') )
    for i in sorted(imgs):
        new_image = Image.open(i)
        images.append(new_image)
    
    # Save into a GIF file 
    outputpath = str(screenshotsPNG_path + '/' + gifname + '.gif')
    images[0].save( outputpath, 
                    format='GIF',
                    append_images=images[1:],
                    save_all=True,
                    optimize=False,
                    duration=400)

    print('\nGIF file "{}" has been written'.format(outputpath))

    return