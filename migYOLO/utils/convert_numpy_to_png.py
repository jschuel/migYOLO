import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from io import BytesIO

def numpy_to_png(numpy_array,vmin,vmax,cmap='jet'):

    # Create a list to hold the in-memory PNGs
    png_images = []

    # Define the colormap and normalization
    try: #For version of matplotlib before 3.9
        colormap = cm.get_cmap(cmap)  # You can choose any colormap you prefer
    except: #for versions after 3.9
        colormap = plt.get_cmap(cmap)  # You can choose any colormap you prefer
    norm = plt.Normalize(vmin=vmin, vmax=vmax)  # Fixed colorscale

    '''For MIGDAL our we have 200 images'''
    for i in range(numpy_array.shape[0]):
        # Apply the colormap and normalization
        colored_image = colormap(norm(numpy_array[i]))
        # Convert to 8-bit unsigned integer
        colored_image = (colored_image[:, :, :3] * 255).astype('uint8')  #Use [:, :, :3] if you want to drop the alpha channel
        # Convert the NumPy array to a PIL image
        img = Image.fromarray(colored_image)
        # Save the PIL image to memory as PNG
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        # Get the byte value of the buffer
        png_images.append(buffer.getvalue())
        buffer.close()
    return png_images
