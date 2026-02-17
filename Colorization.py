# Colorization
# import
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm, AsinhNorm, LinearSegmentedColormap
import sunpy.visualization.colormaps
import easygui
import tifffile
import json
from datetime import datetime, timezone
from pngmeta import PngMeta

# Hydrogen-Alpha Colormap

halpha_red = LinearSegmentedColormap.from_list(
    "halpha_red",
    [(0, 0, 0),
     (0.4, 0.0, 0.0),
     (0.8, 0.1, 0.1),
     (1.0, 0.9, 0.9)]
)

# open file select menu
file_path = easygui.fileopenbox()

# import tiff
data = tifffile.imread(file_path)

# check data
print("\nLoaded Data: ", data.shape, data.dtype)

# get name
name = input("\nEnter file name (YYYY-MM-DDTHH_MM_SSNXXX): ").strip()

# pick normalization
n = input("\nEnter 1 for PowerNorm, 2 for AsinhNorm, or 3 for Linear: ")
if n=="1":
    g = float(input("\nEnter desired gamma value for normalization (0-1): ").strip())
    norming = PowerNorm(gamma=g)
elif n=="2":
    g = float(input("\nEnter desired linear_width for normalization (0-.1 or so):").strip())
    norming = AsinhNorm(linear_width=g)
elif n == "3":
    norming=None
else:
    input("\nInvalid input. Please try again. Default .6 gamma and .05 linear width. ")
    quit()


# pick colormap    
c = input("\nEnter 1 for gist_heat, 2 for halpha color map, and 3 for 'hinodesotintensity': ")
if c == "1":
    cmap = plt.colormaps["gist_heat"]
elif c == "2":
    cmap = halpha_red
elif c == "3": 
    cmap = plt.colormaps['hinodesotintensity']
else:
    cmap = plt.colormaps["gray"]


# output specifically the ground truth to save the data with similar format.
# Also, apply metadata explaining instrument, normalization used, colormap, and when made.

if norming is None:
    norm_name = "linear"
    norm_param = None
elif isinstance(norming, PowerNorm):
    norm_name = "PowerNorm"
    norm_param = {"gamma": norming.gamma}
elif isinstance(norming, AsinhNorm):
    norm_name = "AsinhNorm"
    norm_param = {"linear_width": norming.linear_width}


tiff_metadata = {
    "instrument": "H-alpha solar telescope",
    "normalization": norm_name,
    "norm_param": norm_param,
    "colormap": cmap.name,

    "created_utc": datetime.now(timezone.utc).isoformat() + "Z",
    "software": "Custom H-alpha pipeline (Python)"
}

tifffile.imwrite(
    name + "_linear.tif",
    data.astype("uint16"),
    description=json.dumps(tiff_metadata, indent=2)
)

# save colored image
if norming is None:
    rgb = cmap(data)
else:
    rgb = cmap(norming(data))  # data -> normalized -> RGBA
    
plt.imsave(name + "_display.png", rgb)

# go ahead and add metadata with what options were used

meta = PngMeta(name + "_display.png")

meta["Title"] = "H-alpha Solar Image"
meta["Normalization"] = (
    "linear" if norming is None
    else type(norming).__name__
)
meta["Colormap"] = cmap.name
meta["Comment"] = json.dumps(tiff_metadata, indent=2)
meta["Software"] = "Custom H-alpha pipeline (Python)"

meta.save()

plt.figure(figsize=(6, 6))
plt.imshow(data, cmap=cmap, norm=norming)
plt.colorbar()
plt.title("Preview")
plt.show()

