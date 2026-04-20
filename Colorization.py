# Colorization
# import
from random import choice

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm, AsinhNorm, LinearSegmentedColormap
import sunpy.visualization.colormaps
import easygui
import tifffile
import json
from datetime import datetime, timezone
from pngmeta import PngMeta
import os
import re

# Hydrogen-Alpha Colormap

halpha_red = LinearSegmentedColormap.from_list(
    "halpha_red",
    [(0, 0, 0),
     (0.4, 0.0, 0.0),
     (0.8, 0.1, 0.1),
     (1.0, 0.9, 0.9)]
)




def COL_GEN(file_path,data, choice):
    # get name
    number = input("\nEnter processing number (start at 000): ").strip()

    # Extract date and time from file path using regex
    # Look for pattern like 2026-04-07\Sun\AS_P10\16_57_00
    datetime_match = re.search(r'(\d{4}-\d{2}-\d{2}).*?(\d{2})_(\d{2})_(\d{2})', file_path)
    if datetime_match:
        date_str, hour, minute, second = datetime_match.groups()
        dt = datetime(int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10]), 
                     int(hour), int(minute), int(second))
    else:
        # Fallback to file creation time if date not found in path
        ts = os.path.getctime(file_path)
        dt = datetime.fromtimestamp(ts, timezone.utc)

    # Format datetime without milliseconds
    name = (f'{dt.astimezone().isoformat().split(".")[0].replace(":", "_")}N{number}')

    if choice=="1":
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

    elif choice=="2":
        norming = PowerNorm(gamma=0.6)
        cmap = plt.colormaps['hinodesotintensity']


    tiff_metadata = {
        "instrument": "H-alpha solar telescope",
        "software": "Custom H-alpha pipeline (Python)"
    }

    save_path = os.path.join(os.path.dirname(file_path), name + "_linear.tif")

    tifffile.imwrite(
        save_path,
        data.astype("uint16"),
        description=json.dumps(tiff_metadata, indent=2)
    )

    print(f"Saved linear TIFF to: {os.path.abspath(save_path)}")

    # save colored image
    if norming is None:
        rgb = cmap(data)
    else:
        rgb = cmap(norming(data))  # data -> normalized -> RGBA
        
    plt.imsave(os.path.join(os.path.dirname(file_path), name + "_display.png"), rgb)

    # go ahead and add metadata with what options were used

    meta = PngMeta(os.path.join(os.path.dirname(file_path), name + "_display.png"))

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
    if choice == "2":
        return
    else:
        plt.show()

def IMA_TES(data):
    print("This may take some time")
    for i in range(1,17):
        plt.subplot(4,4,i)
        plt.imshow(plt.colormaps['gist_heat'](PowerNorm(gamma=float(i/16))(data)))
        plt.title(f'Gamma: {i/16}')
        print(f'{i}/16')

    plt.show()

    for i in range(1,17):
        plt.subplot(4,4,i)
        plt.imshow(plt.colormaps['gist_heat'](AsinhNorm(linear_width=float(i/16))(data)))
        plt.title(f'Linear Width: {i/16}')
        print(f'{i}/16')

    plt.show()

def main():
    # open file select menu - allow multiple files
    paths = easygui.fileopenbox(multiple=True)
    
    if not paths:
        print("No files selected.")
        return
    
    # If only one file selected, paths will be a string, convert to list
    if isinstance(paths, str):
        paths = [paths]
    
    print(f"Selected {len(paths)} files to process.")
    
    # Process each file
    for path in paths:
        print(f"\nProcessing: {os.path.basename(path)}")

        # import tiff
        image = tifffile.imread(path)

        # check data
        print("Loaded Data: ", image.shape, image.dtype)

        # select between seeing options or processing images.
        choice = input("Please type 1 to continue, 2 for default,or press enter for test mode (changes will not be saved): ")
        if choice == "1" or choice == "2":
            COL_GEN(path, image, choice)
        else:
            IMA_TES(image)

if __name__ == "__main__":
    main()