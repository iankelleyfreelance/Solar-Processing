# Solar Processing Readme
This is a repository for the processing of images taken with a Hydrogen-Alpha telescope. 

## Colorization Guide 

To run colorization, download the python script and run the following commands (typically within a venv). 
```
python -m pip install numpy matplotlib sunpy[all] tifffile pngmeta easygui
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

## Coordinate Guide
To run image comparison (to see how the Sun changed in a day):
```
python -m pip install matplotlib tkinter
python compare_images.py
```
Select the image **in the folder containing your TIF image files (rotation corrected!) and your JPG files.**

Close once done.

To turn tracked point pairs into a differential-rotation scatter plot:
```
python -m pip install numpy matplotlib astropy sunpy
python solar_rotation_analysis.py
```

This script assumes:
- the solar disk is centered in the image,
- the solar radius is 900 px,
- solar north is up,
- `points.csv` contains `BX,BY,EX,EY` for the start and end positions.

For pooled tracks collected across many dates, add per-row time columns. Any of these pairs will work:
- `start_obstime,end_obstime`
- `start_time,end_time`
- `start_image,end_image`

If no time columns are present, the script now falls back to a pooled sunspot mode:
- it reads the dated images in `Images/`,
- estimates one effective `B0` tilt from the folder dates,
- applies sunspot-focused quality cuts,
- and fits the rotation curve to latitude-bin medians.

Example header:
```
BX,BY,EX,EY,start_image,end_image
```

Useful options:
- `--start-obstime 2026-04-16T17:35:23 --end-obstime 2026-04-17T17:35:23`
- `--center-x 1024 --center-y 1024 --radius-px 900`
- `--max-cmd 45` to reject more near-limb points when the scatter is too noisy
- `--max-delta-lat 0.5` to keep only tracks whose heliographic latitude changes very little over the day
- `--effective-b0-deg -7.2` to override the pooled effective tilt
- `--b0-selection mean` or `--b0-selection median` to choose the pooled tilt directly from folder ephemeris
- `--y-unit deg_day` to plot sidereal rotation in deg/day instead of nHz

Outputs:
- `solar_rotation_analysis.png` with absolute heliographic latitude on the x-axis and sidereal rotation on the y-axis
- `solar_rotation_measurements.csv` with the deprojected latitude, longitude change, rate, and quality-cut flags for each track

For a longer walkthrough of the geometry, pooled approximation, and quality cuts, see `sunspot_rotation_analysis_explained.md`.
