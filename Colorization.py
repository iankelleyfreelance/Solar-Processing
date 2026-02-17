# Color Correction
import numpy as np
from matplotlib import pyplot as plt


from matplotlib.colors import LinearSegmentedColormap

halpha_red = LinearSegmentedColormap.from_list(
    "halpha_red",
    [(0, 0, 0),
     (0.4, 0.0, 0.0),
     (0.8, 0.1, 0.1),
     (1.0, 0.9, 0.9)]
)


import easygui

file_path = easygui.fileopenbox()

# image = np.asarray(Image.open(file_path))
I = plt.imread(file_path)

imgplot = plt.imshow(I, cmap="gist_heat")
plt.show()

name = input("Enter file name (YYYY-MM-DDTHH_MM_SSNXXX): ")
plt.savefig(name + ".png")