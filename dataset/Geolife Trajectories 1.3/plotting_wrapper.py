import io
import imageio
import numpy as np
from PIL import Image
from IPython import display
import matplotlib.pyplot as plt

def show_file_image(filename):

    plt.figure(figsize=(12,12))
    plt.imshow(plt.imread(filename))
    plt.axis('off')

def read_pil_image_from_plt(plt):

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

def create_gif(img_generator, gif_name="./.__gif_sample.gif", fps=10):

    fig = plt.figure(figsize=(4,4))
    with imageio.get_writer(gif_name, mode='I', fps=fps) as writer:

        for img in img_generator():
            # Clear canvas
            plt.gca().cla()
            ax = plt.gca()
            # Draw
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            # Wait to draw - only for online visualization
            display.clear_output(wait=True)
            display.display(plt.gcf())
            # Append to GIF
            img = read_pil_image_from_plt(plt)
            writer.append_data(np.array(img))
        plt.clf()
    return
