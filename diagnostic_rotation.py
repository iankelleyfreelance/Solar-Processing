import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import sunpy.map
from sunpy.util.metadata import MetaDict
from sunpy.coordinates import frames

# Read the data
dataframe = pd.read_csv('points.csv')

print("=" * 70)
print("DIAGNOSTIC: Comparing rotation rate approaches")
print("=" * 70)

# Create solar map metadata
solar_radius_arcsec = 960
solar_radius_px = 900
pixel_scale = solar_radius_arcsec / solar_radius_px

meta = MetaDict({
    'cdelt1': pixel_scale,
    'cdelt2': pixel_scale,
    'crpix1': 1024,
    'crpix2': 1024,
    'crval1': 0,
    'crval2': 0,
    'ctype1': 'HPLN-TAN',
    'ctype2': 'HPLT-TAN',
    'cunit1': 'arcsec',
    'cunit2': 'arcsec',
    'date-obs': '2026-02-24T12:00:00',
    'obsrvtry': 'GONG',
    'dsun_obs': 1.5e11,
})

data = np.zeros((2048, 2048))
solar_map = sunpy.map.Map(data, meta)

# Approach 1: Pixel X displacement
print("\nAPPROACH 1: Pixel X Displacement")
print("-" * 70)
dx_pixels = dataframe['EX'].values - dataframe['BX'].values
print(f"  Min X displacement: {dx_pixels.min():.2f} px")
print(f"  Max X displacement: {dx_pixels.max():.2f} px")
print(f"  Mean X displacement: {dx_pixels.mean():.2f} px")

# Approach 2: Pixel distance
print("\nAPPROACH 2: Total Pixel Distance")
print("-" * 70)
pixel_distance = np.sqrt((dataframe['EX'].values - dataframe['BX'].values)**2 + 
                         (dataframe['EY'].values - dataframe['BY'].values)**2)
print(f"  Min pixel distance: {pixel_distance.min():.2f} px")
print(f"  Max pixel distance: {pixel_distance.max():.2f} px")
print(f"  Mean pixel distance: {pixel_distance.mean():.2f} px")

# Approach 3: Heliographic angular separation (current method)
print("\nAPPROACH 3: Heliographic Angular Separation (Current)")
print("-" * 70)
angular_separations = []
begin_latitudes = []
begin_longitudes = []
end_latitudes = []
end_longitudes = []

for idx, row in dataframe.iterrows():
    begin_x, begin_y = row['BX'], row['BY']
    end_x, end_y = row['EX'], row['EY']
    
    try:
        begin_coord = solar_map.pixel_to_world(begin_x * u.pix, begin_y * u.pix)
        end_coord = solar_map.pixel_to_world(end_x * u.pix, end_y * u.pix)
        
        begin_helio = begin_coord.transform_to(frames.HeliographicStonyhurst(obstime=solar_map.date))
        end_helio = end_coord.transform_to(frames.HeliographicStonyhurst(obstime=solar_map.date))
        
        separation = begin_coord.separation(end_coord)
        angular_sep_deg = separation.to(u.degree).value
        angular_separations.append(angular_sep_deg)
        
        begin_latitudes.append(begin_helio.lat.to(u.degree).value)
        begin_longitudes.append(begin_helio.lon.to(u.degree).value)
        end_latitudes.append(end_helio.lat.to(u.degree).value)
        end_longitudes.append(end_helio.lon.to(u.degree).value)
        
    except Exception as e:
        angular_separations.append(np.nan)
        begin_latitudes.append(np.nan)
        begin_longitudes.append(np.nan)
        end_latitudes.append(np.nan)
        end_longitudes.append(np.nan)

angular_separations = np.array(angular_separations)
begin_latitudes = np.array(begin_latitudes)
begin_longitudes = np.array(begin_longitudes)
end_latitudes = np.array(end_latitudes)
end_longitudes = np.array(end_longitudes)

# Remove NaNs
valid_mask = ~np.isnan(angular_separations)
angular_separations = angular_separations[valid_mask]
begin_latitudes_clean = begin_latitudes[valid_mask]
begin_longitudes_clean = begin_longitudes[valid_mask]
end_latitudes_clean = end_latitudes[valid_mask]
end_longitudes_clean = end_longitudes[valid_mask]

print(f"  Min angular separation: {angular_separations.min():.6f}°")
print(f"  Max angular separation: {angular_separations.max():.6f}°")
print(f"  Mean angular separation: {angular_separations.mean():.6f}°")

# Approach 4: Longitude displacement at begin latitude
print("\nAPPROACH 4: Longitude Displacement (with latitude correction)")
print("-" * 70)
longitude_displacement = end_longitudes_clean - begin_longitudes_clean
# Correct for discontinuity at ±180°
longitude_displacement = np.where(longitude_displacement > 180, longitude_displacement - 360,
                                  np.where(longitude_displacement < -180, longitude_displacement + 360,
                                           longitude_displacement))

# Account for latitude: rotational displacement on sphere
# At latitude φ, a longitude change of Δλ corresponds to angular distance Δλ*cos(φ)
# But for differential rotation fitting, we want the longitude displacement
latitude_rad = np.radians(begin_latitudes_clean)
adjusted_lon_displacement = longitude_displacement / np.cos(latitude_rad)

print(f"  Raw longitude displacement:")
print(f"    Min: {longitude_displacement.min():.6f}°")
print(f"    Max: {longitude_displacement.max():.6f}°")
print(f"    Mean: {longitude_displacement.mean():.6f}°")

print(f"  Latitude-corrected displacement:")
print(f"    Min: {adjusted_lon_displacement.min():.6f}°")
print(f"    Max: {adjusted_lon_displacement.max():.6f}°")
print(f"    Mean: {adjusted_lon_displacement.mean():.6f}°")

# Fit all approaches against latitude
print("\n" + "=" * 70)
print("FITTING RESULTS")
print("=" * 70)

def rotation_model(latitude_deg, A, B, C):
    lat_rad = np.radians(latitude_deg)
    return A - B * np.sin(lat_rad)**2 - C * np.sin(lat_rad)**4

def calc_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

# Fit 1: X displacement vs latitude
try:
    dx_valid = dx_pixels[valid_mask]
    popt1, _ = curve_fit(rotation_model, begin_latitudes_clean, dx_valid, 
                         p0=[0.05, 0.05, 0.05], maxfev=5000)
    pred1 = rotation_model(begin_latitudes_clean, *popt1)
    r2_1 = calc_r_squared(dx_valid, pred1)
    print(f"\n1. Pixel X Displacement vs Latitude:")
    print(f"   R² = {r2_1:.6f}")
    print(f"   Fitted A = {popt1[0]:.6f}, B = {popt1[1]:.6f}, C = {popt1[2]:.6f}")
except Exception as e:
    print(f"\n1. Pixel X Displacement: ERROR - {e}")

# Fit 2: Total pixel distance vs latitude
try:
    pixel_dist_valid = pixel_distance[valid_mask]
    popt2, _ = curve_fit(rotation_model, begin_latitudes_clean, pixel_dist_valid,
                         p0=[0.05, 0.05, 0.05], maxfev=5000)
    pred2 = rotation_model(begin_latitudes_clean, *popt2)
    r2_2 = calc_r_squared(pixel_dist_valid, pred2)
    print(f"\n2. Total Pixel Distance vs Latitude:")
    print(f"   R² = {r2_2:.6f}")
    print(f"   Fitted A = {popt2[0]:.6f}, B = {popt2[1]:.6f}, C = {popt2[2]:.6f}")
except Exception as e:
    print(f"\n2. Total Pixel Distance: ERROR - {e}")

# Fit 3: Angular separation vs latitude (current method)
try:
    popt3, _ = curve_fit(rotation_model, begin_latitudes_clean, angular_separations,
                         p0=[0.05, 0.05, 0.05], maxfev=5000)
    pred3 = rotation_model(begin_latitudes_clean, *popt3)
    r2_3 = calc_r_squared(angular_separations, pred3)
    print(f"\n3. Angular Separation vs Latitude (Current):")
    print(f"   R² = {r2_3:.6f}")
    print(f"   Fitted A = {popt3[0]:.6f}, B = {popt3[1]:.6f}, C = {popt3[2]:.6f}")
except Exception as e:
    print(f"\n3. Angular Separation: ERROR - {e}")

# Fit 4: Longitude displacement vs latitude
try:
    popt4, _ = curve_fit(rotation_model, begin_latitudes_clean, longitude_displacement,
                         p0=[0.05, 0.05, 0.05], maxfev=5000)
    pred4 = rotation_model(begin_latitudes_clean, *popt4)
    r2_4 = calc_r_squared(longitude_displacement, pred4)
    print(f"\n4. Longitude Displacement vs Latitude:")
    print(f"   R² = {r2_4:.6f}")
    print(f"   Fitted A = {popt4[0]:.6f}, B = {popt4[1]:.6f}, C = {popt4[2]:.6f}")
except Exception as e:
    print(f"\n4. Longitude Displacement: ERROR - {e}")

# Fit 5: Latitude-corrected displacement vs latitude
try:
    # Filter out poles where cos(lat) ≈ 0
    pole_mask = np.abs(begin_latitudes_clean) < 80
    lat_clean_poles = begin_latitudes_clean[pole_mask]
    adj_disp_clean = adjusted_lon_displacement[pole_mask]
    
    popt5, _ = curve_fit(rotation_model, lat_clean_poles, adj_disp_clean,
                         p0=[0.05, 0.05, 0.05], maxfev=5000)
    pred5 = rotation_model(lat_clean_poles, *popt5)
    r2_5 = calc_r_squared(adj_disp_clean, pred5)
    print(f"\n5. Latitude-Corrected Displacement vs Latitude (|lat|<80°):")
    print(f"   R² = {r2_5:.6f}")
    print(f"   Fitted A = {popt5[0]:.6f}, B = {popt5[1]:.6f}, C = {popt5[2]:.6f}")
except Exception as e:
    print(f"\n5. Latitude-Corrected: ERROR - {e}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("If pixel X displacement (R²≈0.27) >> Angular separation (R²≈0.024),")
print("then the heliographic coordinate conversion is destroying the signal.")
print("This suggests issues with:")
print("  - Reference frame/date mismatch for multi-month data")
print("  - Using angular separation instead of longitude displacement")
print("  - WCS/pixel scale calibration")
