import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

dataframe = pd.read_csv('points.csv')

# dataframe structure: Begin X, Begin Y, End X, End Y, Magnitude
# Calculate the displacement vectors
dataframe['DX'] = dataframe['EX'] - dataframe['BX']
dataframe['DY'] = dataframe['EY'] - dataframe['BY']

# Calculate magnitude (velocity/rotation rate)
if 'Magnitude' not in dataframe.columns:
    dataframe['Magnitude'] = np.sqrt(dataframe['DX']**2 + dataframe['DY']**2)

# Convert pixel Y coordinate to latitude
# Assuming 2048x2048 image: Y=0 at south pole, Y=2048 at north pole
# Latitude ranges from -90 (south) to +90 (north)
dataframe['Latitude'] = (dataframe['BY'] / 2048.0 - 0.5) * 180  # -90 to +90

# Rotation rate (could be normalized or scaled as needed)
dataframe['RotationRate'] = dataframe['Magnitude']

# Define a fitting function (differential rotation model)
# Common model: ω(lat) = A - B*sin²(lat) - C*sin⁴(lat)
def rotation_model(latitude, A, B, C):
    lat_rad = np.radians(latitude)
    return A - B * np.sin(lat_rad)**2 - C * np.sin(lat_rad)**4

# Prepare data for fitting (remove any NaN values)
valid_data = dataframe.dropna(subset=['Latitude', 'RotationRate'])
latitudes = valid_data['Latitude'].values
rotation_rates = valid_data['RotationRate'].values

# Fit the curve
try:
    # Initial guess for parameters
    popt, _ = curve_fit(rotation_model, latitudes, rotation_rates, 
                        p0=[1, 0.1, 0.01], maxfev=5000)
    A, B, C = popt
    
    # Calculate R² (coefficient of determination)
    predictions = rotation_model(latitudes, A, B, C)
    ss_res = np.sum((rotation_rates - predictions)**2)  # Residual sum of squares
    ss_tot = np.sum((rotation_rates - np.mean(rotation_rates))**2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)
    
    # Create smooth curve for plotting
    lat_smooth = np.linspace(-90, 90, 200)
    rotation_smooth = rotation_model(lat_smooth, A, B, C)
    
    # Create scatter plot with fitted curve
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    scatter = ax.scatter(latitudes, rotation_rates, alpha=0.6, s=30, c='blue', label='Observed data')
    
    # Fitted curve
    ax.plot(lat_smooth, rotation_smooth, 'r-', linewidth=2, label='Fitted curve')
    
    # Labels and title
    ax.set_xlabel('Latitude (degrees)', fontsize=12)
    ax.set_ylabel('Rotation Rate', fontsize=12)
    ax.set_title('Solar Rotation vs. Latitude', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add equation and R² to plot
    equation_text = f'ω(lat) = {A:.4f} - {B:.4f}sin²(lat) - {C:.4f}sin⁴(lat)\nR² = {r_squared:.4f}'
    ax.text(0.05, 0.95, equation_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('solar_rotation_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to: solar_rotation_analysis.png")
    print(f"\nFitted parameters:")
    print(f"A = {A:.6f}")
    print(f"B = {B:.6f}")
    print(f"C = {C:.6f}")
    print(f"\nGoodness of fit:")
    print(f"R² = {r_squared:.6f}")
    print(f"\nEquation: ω(lat) = {A:.4f} - {B:.4f}*sin²(lat) - {C:.4f}*sin⁴(lat)")
    
    plt.show()
    
except Exception as e:
    print(f"Error during curve fitting: {e}")
    print("Make sure your CSV has the required columns: BX, BY, EX, EY, Magnitude")
