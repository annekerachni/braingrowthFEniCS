import fenics
from PIL import Image
import glob
#import matplotlib.pyplot as plt

def make_gif(screenshotsPNG_path, gifname, duration, loop_mode):
    """
    Make gif from sery of (PNG) simulation results screenshots.
    Code source: https://pythonprogramming.altervista.org/png-to-gif/?doing_wp_cron=1659355206.1907370090484619140625
    """
    # Create the frames
    images = []
    
    imgs = glob.glob( str(screenshotsPNG_path + '*.png') )
    for i in sorted(imgs):
        new_image = Image.open(i)
        images.append(new_image)
    
    # Save into a GIF file 
    if loop_mode == 0:
        outputpath = str(screenshotsPNG_path + gifname + '.gif')
        images[0].save( outputpath, 
                        format='GIF',
                        append_images=images[1:],
                        save_all=True,
                        optimize=False,
                        duration=duration,
                        loop=loop_mode) # default duration: 400
    
    elif loop_mode == None:
        outputpath = str(screenshotsPNG_path + gifname + '.gif')
        images[0].save( outputpath, 
                        format='GIF',
                        append_images=images[1:],
                        save_all=True,
                        optimize=False,
                        duration=duration) # default duration: 400

    print('\nGIF file "{}" has been written'.format(outputpath))

    return


if __name__ == '__main__':

    screenshotsPNG_path = './results/braingrowth_simulations_png/gif/'
    gifname = 'braingrowth'
    duration = 400 # 200 (600 for looping GIF)
    loop_mode = 0 # None (0 for infini looping GIF)

    make_gif(screenshotsPNG_path, gifname, duration, loop_mode)