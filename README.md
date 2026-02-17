# Solar Processing Readme
This is a repository for the processing of images taken with a Hydrogen-Alpha telescope. 

## Colorization Guide 

To run colorization, download the python script and run the following commands (typically within a venv). 
```
pip install numpy matplotlib sunpy[all] tifffile pngmeta easygui
python Colorization.py
```

### Here are the options explained:

#### Normalization
- Perhaps the biggest thing that impacts the quality of the image.
- PowerNorm - Applies intensity to some specified power. Stretches dark end and compresses bright end. Good for structure.
-   Gamma should be between 0 and 1 ideally.
- Asinh - Applies inverse sinh function. Linear_width is the cutoff below which there is linear and above which there is log. Prominences, disk, local. Good to see everything.
-   Linear_width is usually between 0 and .1.
- None - Linear. This is the most basic (and therefore should be saved for TIFF files for later processing!), but the dynamic range is so vast that details can be hard to pull out.

#### Colormaps
- H-Alpha is one I made that's basically just red. Simulates the H-alpha of the telescope.
- Gist_Heat is a very nice looking colormap that works well.
- https://docs.sunpy.org/en/stable/reference/visualization.html#module-sunpy.visualization.colormaps see this for the sunpy colormap. I quite look the look of "hinodesotintensity"

