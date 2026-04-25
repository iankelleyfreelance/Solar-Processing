import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataframe = pd.read_csv('points.csv')

# dataframe structure: Begin X, Begin Y, End X, End Y, Magnitude
# Calculate the displacement vectors
dataframe['DX'] = dataframe['EX'] - dataframe['BX']
dataframe['DY'] = dataframe['EY'] - dataframe['BY']

# Calculate magnitude if not already present
if 'Magnitude' not in dataframe.columns:
    dataframe['Magnitude'] = np.sqrt(dataframe['DX']**2 + dataframe['DY']**2)

# Create the vector field plot
# Create figure with 2048x2048 pixel size (approx 21.3x21.3 inches at 96 DPI)
fig, ax = plt.subplots(figsize=(21.33, 21.33), dpi=96)

# Plot vectors using quiver
# quiver(X, Y, U, V, C) where (X,Y) is the starting point, (U,V) is the direction
quiver = ax.quiver(
    dataframe['BX'], 
    dataframe['BY'], 
    dataframe['DX'], 
    dataframe['DY'],
    dataframe['Magnitude'],  # Color by magnitude
    cmap='viridis',
    scale=1,
    scale_units='xy',
    angles='xy',
    width=0.004
)

# Add colorbar to show magnitude scale
cbar = plt.colorbar(quiver, ax=ax)
cbar.set_label('Magnitude', rotation=270, labelpad=20)

# Labels and title
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_xbound(0, 2048)
ax.set_ybound(0, 2048)
ax.set_title('Solar Vector Field')
ax.grid(True, alpha=0.3)

# Set equal aspect ratio so vectors aren't distorted
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()

# Save as 2048x2048 image
output_filename = 'solar_vector_field.png'
plt.savefig(output_filename, dpi=96, bbox_inches='tight', pad_inches=0.1)
print(f"Vector field saved to: {output_filename}")

plt.show()

plt.scatter(dataframe['BY'], dataframe['Magnitude'], cmap='viridis', s=10)
plt.show()