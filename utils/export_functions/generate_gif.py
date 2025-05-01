from PIL import Image
import glob
#import matplotlib.pyplot as plt
import imageio
import numpy as np

def make_GIF_LowQuality(screenshotsPNG_path, gifname, duration, loop_mode):
    """
    Make gif from sery of (PNG) simulation results screenshots.
    Code source: https://pythonprogramming.altervista.org/png-to-gif/?doing_wp_cron=1659355206.1907370090484619140625
    """
    # Create the frames
    images = []
    
    imgs = glob.glob( str(screenshotsPNG_path + '*.png') )
    for i in sorted(imgs):
        im = Image.open(i)
        new_image = im.convert('P', palette=Image.ADAPTIVE, colors=256) # new_image = Image.open(i)
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



def make_GIF_HighQuality(screenshotsPNG_path, animation_name, duration, loop_mode):
    """Other formats"""

    # Create the frames
    images = []
    
    imgs = glob.glob( str(screenshotsPNG_path + '*.png') )
    for i in sorted(imgs):
        im = Image.open(i)
        new_image = im.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=256) 
        images.append(new_image) 

    #imgs = sorted(glob.glob("/home/administrateur/Bureau/gif_github/biais/*.png"))
    # print(images)

    outputpath = str(screenshotsPNG_path + animation_name + '.gif') 
    images[0].save( outputpath, 
                    format='GIF',
                    append_images=images[1:],
                    save_all=True,
                    optimize=False,
                    duration=duration,
                    loop=loop_mode) # default duration: 400

    """
    frames = [imageio.imread(f) for f in images]

    outputpath = str(screenshotsPNG_path + mp4name + '.mp4') 
    imageio.mimsave(outputpath, frames, fps=10)
    """

    # Other code to improve GIF quality
    ###################################
    # imgs = sorted(glob.glob(str(screenshotsPNG_path + '*.png')))
    # print(imgs)
    # frames = [imageio.imread(f) for f in imgs]
    # outputpath = str(screenshotsPNG_path + gifname + '.png')
    # imageio.mimsave(outputpath, frames, duration=duration, format='PNG-PIL')

    print('\nGIF file "{}" has been written'.format(outputpath))

    return 

def make_GIF_HighQuality_Transparent(screenshotsPNG_path, animation_name, duration, loop_mode):
    """Other formats"""

    # Create the frames
    images = []
    
    imgs = glob.glob( str(screenshotsPNG_path + '*.png') )
    for i in sorted(imgs):

        im = Image.open(i).convert('RGBA')
        alpha = im.getchannel('A')

        new_image = im.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=255)  # Convertir en mode P (palette) avec 255 couleurs (la 256e sera la transparence)
        
        mask = Image.eval(alpha, lambda a: 255 if a <=128 else 0) # Créer un masque : pixels avec alpha < 128 seront transparents
        new_image.paste(255, mask) # Utiliser l’index 255 de la palette pour la transparence

        images.append(new_image) 


    #imgs = sorted(glob.glob("/home/administrateur/Bureau/gif_github/biais/*.png"))
    # print(images)

    outputpath = str(screenshotsPNG_path + animation_name + '.gif') 
    images[0].save( outputpath, 
                    format='GIF',
                    append_images=images[1:],
                    save_all=True,
                    optimize=False,
                    duration=duration,
                    loop=loop_mode,
                    transparency=255) # default duration: 400

    """
    frames = [imageio.imread(f) for f in images]

    outputpath = str(screenshotsPNG_path + mp4name + '.mp4') 
    imageio.mimsave(outputpath, frames, fps=10)
    """

    # Other code to improve GIF quality
    ###################################
    # imgs = sorted(glob.glob(str(screenshotsPNG_path + '*.png')))
    # print(imgs)
    # frames = [imageio.imread(f) for f in imgs]
    # outputpath = str(screenshotsPNG_path + gifname + '.png')
    # imageio.mimsave(outputpath, frames, duration=duration, format='PNG-PIL')

    print('\nGIF file "{}" has been written'.format(outputpath))

    return 

if __name__ == '__main__':

    screenshotsPNG_path = '/home/administrateur/Bureau/gif_github/biais/' #'./results/braingrowth_simulations_png/gif/'
    animation_name = 'braingrowth_duration100'
    duration = 100 # 500 (100 to 600 for longer GIF)
    loop_mode = 0 # None for one loop, 0 for infini looping GIF

    #make_GIF_LowQuality(screenshotsPNG_path, gifname, duration, loop_mode)
    #make_GIF_HighQuality(screenshotsPNG_path, animation_name, duration, loop_mode)
    make_GIF_HighQuality_Transparent(screenshotsPNG_path, animation_name, duration, loop_mode)