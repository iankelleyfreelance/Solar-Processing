import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image
import datetime

def find_matching_image(tif_filename):
    """
    Given a TIF filename (e.g., 2026-02-20T16_05_02_rotated.tif or Images\2026-04-16T17_35_23_rotated.tif),
    find the corresponding next-day image (JPG).
    Ignores seconds - matches on date, hour, and minute only.
    Returns the JPG filename if found, None otherwise.
    """
    import os
    import re
    import glob
    
    # Parse the TIF filename to get the date and time
    match = re.match(r'(\d{4})-(\d{2})-(\d{2})T(\d{2})_(\d{2})_(\d{2})_rotated\.tif', os.path.basename(tif_filename))
    if not match:
        return None
    
    year, month, day, hour, minute, second = map(int, match.groups())
    tif_date = datetime.datetime(year, month, day, hour, minute, second)
    
    # Add 1 day
    next_day = tif_date + datetime.timedelta(days=1)
    
    # Get the directory of the TIF file
    tif_dir = os.path.dirname(tif_filename) or "."
    
    # Create pattern for matching JPG files (ignoring seconds)
    # Pattern: YYYY-MM-DDTHH_MM_* to match any seconds and channel
    search_pattern = os.path.join(tif_dir, next_day.strftime("%Y-%m-%dT%H_%M_") + "*.jpg")
    
    matching_files = glob.glob(search_pattern)
    
    # If not found in TIF directory, search current directory
    if not matching_files:
        search_pattern = next_day.strftime("%Y-%m-%dT%H_%M_") + "*.jpg"
        matching_files = glob.glob(search_pattern)
    
    if matching_files:
        # Return the full path to the match
        return matching_files[0]
    
    return None

def compare_images(tif_path, jpg_path):
    """
    Display two images side by side for detailed comparison with cursor coordinates.
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
    ax1.set_xlabel('X pixel')
    ax1.set_ylabel('Y pixel')
    
    # Display JPG on right
    ax2 = axes[1]
    ax2.imshow(jpg_img)
    ax2.set_title(f'JPG: {jpg_path}', fontsize=12)
    ax2.set_xlabel('X pixel')
    ax2.set_ylabel('Y pixel')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    import os
    import glob
    from tkinter import Tk, filedialog
    
    # Hide the root tkinter window
    root = Tk()
    root.withdraw()
    
    # Open file dialog to select a TIF file
    tif_file = filedialog.askopenfilename(
        title="Select a TIF file to compare",
        filetypes=[("TIFF files", "*.tif"), ("All files", "*.*")],
        initialdir="Images" if os.path.exists("Images") else "."
    )
    
    if not tif_file:
        print("No file selected.")
    else:
        print(f"Selected TIF: {tif_file}")
        
        # Find matching JPG
        matching_jpg = find_matching_image(tif_file)
        
        if matching_jpg:
            print(f"Found matching JPG: {matching_jpg}")
            compare_images(tif_file, matching_jpg)
        else:
            print(f"No matching JPG found for {tif_file}")
            print("\nSearched for images matching the date pattern one day after the TIF.")

