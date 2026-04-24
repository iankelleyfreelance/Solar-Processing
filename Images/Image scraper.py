import numpy
import datetime
from PIL import Image
import urllib.request
import ssl

# Create unverified SSL context to bypass certificate verification
ssl_context = ssl._create_unverified_context()

# TIF file dates (one day after each, we'll add 1 day to these)
tif_dates = [
    datetime.datetime(2026, 1, 26, 17, 21, 57),
    datetime.datetime(2026, 2, 6, 20, 7, 35),
    datetime.datetime(2026, 2, 7, 18, 53, 47),
    datetime.datetime(2026, 2, 9, 20, 1, 37),
    datetime.datetime(2026, 2, 12, 17, 32, 30),
    datetime.datetime(2026, 2, 13, 17, 37, 33),
    datetime.datetime(2026, 2, 16, 17, 7, 2),
    datetime.datetime(2026, 2, 20, 16, 0, 57),
    datetime.datetime(2026, 2, 20, 16, 5, 2),
    datetime.datetime(2026, 2, 20, 16, 10, 57),
    datetime.datetime(2026, 2, 21, 12, 12, 2),
    datetime.datetime(2026, 2, 21, 12, 30, 20),
    datetime.datetime(2026, 2, 21, 12, 35, 26),
    datetime.datetime(2026, 2, 21, 12, 58, 45),
    datetime.datetime(2026, 2, 27, 18, 56, 30),
    datetime.datetime(2026, 2, 28, 18, 38, 25),
    datetime.datetime(2026, 3, 17, 22, 5, 16),
    datetime.datetime(2026, 3, 19, 18, 13, 21),
    datetime.datetime(2026, 3, 22, 19, 31, 41),
    datetime.datetime(2026, 3, 23, 20, 16, 51),
    datetime.datetime(2026, 3, 24, 16, 56, 55),
    datetime.datetime(2026, 4, 7, 21, 22, 42),
    datetime.datetime(2026, 4, 7, 21, 23, 42),
    datetime.datetime(2026, 4, 7, 21, 34, 42),
    datetime.datetime(2026, 4, 7, 21, 45, 59),
    datetime.datetime(2026, 4, 7, 21, 57, 0),
    datetime.datetime(2026, 4, 9, 20, 12, 2),
    datetime.datetime(2026, 4, 16, 17, 35, 23),
]

# Add 1 day to each date
dates = [d + datetime.timedelta(days=1) for d in tif_dates]

# Open log file for failed downloads
failed_log = open("download_failures.log", "w")

for i in range(0, len(dates)):
    date_str = dates[i].strftime("%Y%m%d")
    month_str = dates[i].strftime("%Y%m")
    hour_minute = dates[i].strftime("%H%M")
    
    # Try different seconds: 02, 42, 52, 22
    seconds_to_try = ["02", "42", "52", "22"]
    channels = ["Ch", "Bh", "Lh", "Uh"]
    downloaded = False
    
    for seconds in seconds_to_try:
        if downloaded:
            break
        
        for channel in channels:
            if downloaded:
                break
            
            url_filename = f"{date_str}{hour_minute}{seconds}{channel}.jpg"
            current_link = f"https://gong2.nso.edu/HA/hag/{month_str}/{date_str}/{url_filename}"
            
            # Create ISO 8601 format filename for saving locally
            iso_filename = dates[i].strftime("%Y-%m-%dT%H_%M_%S") + f"_{channel}.jpg"
            
            print(f"Trying: {current_link}")
            
            try:
                with urllib.request.urlopen(current_link, context=ssl_context) as response:
                    with open(iso_filename, 'wb') as out_file:
                        out_file.write(response.read())
                print(f"Downloaded: {iso_filename}")
                downloaded = True
            except Exception as e:
                print(f"Not found ({channel}, {seconds}s): {e}")
    
    if not downloaded:
        error_msg = f"Could not find image for {dates[i].strftime('%Y-%m-%d %H:%M:%S')} (one day after {tif_dates[i].strftime('%Y-%m-%d %H:%M:%S')})"
        print(error_msg)
        failed_log.write(error_msg + "\n")

failed_log.close()
print("\nLog file saved to: download_failures.log")

