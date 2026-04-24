import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image
import datetime

def find_matching_image(tif_filename):
    """
    Given a TIF filename (e.g., 2026-02-20T16_05_02_rotated.tif),
    find the corresponding next-day image (JPG).
    Returns the JPG filename if found, None otherwise.
    """
    import os
    import re
    
    # Parse the TIF filename to get the date and time
    match = re.match(r'(\d{4})-(\d{2})-(\d{2})T(\d{2})_(\d{2})_(\d{2})_rotated\.tif', tif_filename)
    if not match:
        return None
    
    year, month, day, hour, minute, second = map(int, match.groups())
    tif_date = datetime.datetime(year, month, day, hour, minute, second)
    
    # Add 1 day
    next_day = tif_date + datetime.timedelta(days=1)
    
    # Look for JPG files that match this next_day
    # Try different channel suffixes
    channels = ['Ch', 'Bh', 'Lh', 'Uh', 'Th']
    
    for channel in channels:
        # Try files with same hour/minute
        jpg_filename = next_day.strftime(f"%Y-%m-%dT%H_%M_%S_{channel}.jpg")
        if os.path.exists(jpg_filename):
            return jpg_filename
    
    return None

def compare_images(tif_path, jpg_path):
    """
    Display two images side by side with pixel grids for detailed comparison.
    """
    # Load images
    tif_img = Image.open(tif_path)
    jpg_img = Image.open(jpg_path)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Display TIF on left
    ax1 = axes[0]
    ax1.imshow(tif_img)
    ax1.set_title(f'TIF: {tif_path}', fontsize=12)
    ax1.grid(True, which='both', color='red', linewidth=0.5, alpha=0.5)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax1.grid(True, which='minor', color='gray', linewidth=0.2, alpha=0.3)
    
    # Display JPG on right
    ax2 = axes[1]
    ax2.imshow(jpg_img)
    ax2.set_title(f'JPG: {jpg_path}', fontsize=12)
    ax2.grid(True, which='both', color='red', linewidth=0.5, alpha=0.5)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax2.grid(True, which='minor', color='gray', linewidth=0.2, alpha=0.3)
    
    # Enable interactive zoom and pan
    fig.suptitle('Use mouse to zoom/pan. Scroll to zoom in/out.', fontsize=10, color='blue')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    import os
    import glob
    
    # Find all TIF files
    tif_files = sorted(glob.glob("*_rotated.tif"))
    
    if not tif_files:
        print("No TIF files found!")
    else:
        # Use the last TIF file
        latest_tif = tif_files[-1]
        print(f"Found TIF: {latest_tif}")
        
        # Find matching JPG
        matching_jpg = find_matching_image(latest_tif)
        
        if matching_jpg:
            print(f"Found matching JPG: {matching_jpg}")
            compare_images(latest_tif, matching_jpg)
        else:
            print(f"No matching JPG found for {latest_tif}")
            print("\nAvailable JPG files:")
            jpgs = glob.glob("*.jpg")
            for jpg in sorted(jpgs):
                print(f"  - {jpg}")
