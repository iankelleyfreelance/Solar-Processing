# Colorization
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm, AsinhNorm, LinearSegmentedColormap
import sunpy.visualization.colormaps
import easygui
import imageio.v3 as iio

# Hydrogen-Alpha Colormap

halpha_red = LinearSegmentedColormap.from_list(
    "halpha_red",
    [(0, 0, 0),
     (0.4, 0.0, 0.0),
     (0.8, 0.1, 0.1),
     (1.0, 0.9, 0.9)]
)

# open file select menu
import easygui
file_path = easygui.fileopenbox()

# import tiff
import imageio.v3 as iio
data = iio.imread(file_path)

# check data
print("Loaded Data: ", data.shape, data.dtype)

# get name
name = input("\nEnter file name (YYYY-MM-DDTHH_MM_SSNXXX): ").strip()

# pick normalization
n = input("\nEnter 1 for PowerNorm, 2 for AsinhNorm, or 3 for Linear: ")
if n=="1":
    g = float(input("\nEnter desired gamma value for normalization (0-1): ").strip())
    norming = PowerNorm(gamma=g)
elif n=="2":
    g = float(input("\nEnter desired linear_width for normalization (0-1):").strip())
    norming = AsinhNorm(linear_width=g)
elif n == "3":
    norming=None
else:
    input("\nInvalid input. Please try again. Default .6 gamma and .05 linear width. ")
    quit()


# pick colormap    
c = input("\nEnter 1 for gist_heat, 2 for halpha color map, and 3 for 'hinodesotintensity'")
if c == "1":
    cmap = plt.colormaps["gist_heat"]
elif c == "2":
    cmap = halpha_red
elif c == "3": 
    cmap = plt.colormaps['hinodesotintensity']
else:
    cmap = plt.colormaps["grayscale"]


# output specifically the ground truth to save the data with similar format
iio.imwrite(
    name + "_linear.tif",
    data.astype("uint16")
)

# save colored image
rgb = cmap(norming(data))  # data -> normalized -> RGBA
plt.imsave(name + "_display.png", rgb)

plt.figure(figsize=(6, 6))
plt.imshow(data, cmap=cmap, norm=norming)
plt.colorbar()
plt.title("Preview")
plt.show()
