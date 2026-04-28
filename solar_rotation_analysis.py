# ============================================================================
# SOLAR ROTATION ANALYSIS: Track sunspot features and measure differential rotation
# ============================================================================
# This script converts pixel-based tracked feature points into heliographic coordinates
# and analyzes solar differential rotation by fitting the rotation rate as a function
# of latitude using the Carrington/Newton-Nunn model:
#   ω(latitude) = A - B*sin²(latitude) - C*sin⁴(latitude)
# 
# Two modes:
# 1. EXACT: Uses per-track observation times for precise coordinate transformations
# 2. POOLED: When no times available, uses a single effective B0 angle for all tracks
# ============================================================================

import argparse          # Command-line argument parsing
import csv              # Reading/writing CSV files
import math             # Basic mathematical functions
import re               # Regular expressions for timestamp parsing
from datetime import timedelta  # Time arithmetic
from pathlib import Path        # File path handling

import astropy.units as u              # Unit handling (degrees, arcseconds, etc.)
import matplotlib
import numpy as np                     # Numerical arrays and operations
from astropy.coordinates import SkyCoord  # Coordinate system transformations
from astropy.time import Time           # High-precision time handling
from scipy.optimize import least_squares  # Curve fitting with constraints
from sunpy.coordinates import frames, sun  # Solar coordinate frames and solar ephemeris
from sunpy.coordinates.ephemeris import get_earth  # Earth position relative to Sun

matplotlib.use("Agg")  # Use non-interactive backend for server/batch processing
import matplotlib.pyplot as plt  # Plotting library


# ============================================================================
# CONSTANTS
# ============================================================================
# Regex pattern to parse ISO 8601 timestamps from filenames
# Format: YYYY-MM-DDTHH[-_]MM[-_]SS (allows dashes or underscores as separators)
TIMESTAMP_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})T(?P<hour>\d{2})[-_](?P<minute>\d{2})[-_](?P<second>\d{2})"
)

# Carrington sidereal rotation rate: The Sun rotates 360° every 25.38 days in a fixed frame
CARRINGTON_SIDEREAL_RATE_DEG_PER_DAY = 360.0 / 25.38

# Earth's orbital rate: Earth moves ~360°/365.2422 days around the Sun
# This is needed to convert Synodic rates (Earth-relative) to Sidereal rates (Sun-fixed)
EARTH_ORBITAL_RATE_DEG_PER_DAY = 360.0 / 365.2422

# Conversion factor from deg/day to nanoHertz (nHz)
# 1 nHz = 1e-9 Hz; rotation rate in nHz is useful for period analysis
DEG_PER_DAY_TO_NHZ = 1e9 / (360.0 * 86400.0)

# Reference sunspot rotation curve (Newton-Nunn model, historical data)
# These are standard reference values from solar physics literature
REFERENCE_SUNSPOT_A = 14.38      # Equatorial rotation rate (deg/day)
REFERENCE_SUNSPOT_B = 2.96       # Sin²(latitude) coefficient
REFERENCE_SUNSPOT_C = 0.0        # Sin⁴(latitude) coefficient


# ============================================================================
# COMMAND-LINE ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert tracked solar-image points into heliographic coordinates and "
            "make a sunspot differential-rotation plot."
        )
    )
    parser.add_argument("--points-csv", default="points.csv", help="CSV with BX,BY,EX,EY and optional time columns.")
    parser.add_argument("--images-dir", default="Images", help="Directory of the dated source images.")
    parser.add_argument("--start-obstime", help="Global start observation time, e.g. 2026-04-16T17:35:23")
    parser.add_argument("--end-obstime", help="Global end observation time, e.g. 2026-04-17T17:35:23")
    parser.add_argument("--start-image", help="Global start image filename or path to parse the start time.")
    parser.add_argument("--end-image", help="Global end image filename or path to parse the end time.")
    parser.add_argument("--center-x", type=float, default=1024.0, help="Disk center X in pixels.")
    parser.add_argument("--center-y", type=float, default=1024.0, help="Disk center Y in pixels.")
    parser.add_argument("--radius-px", type=float, default=900.0, help="Solar disk radius in pixels.")
    parser.add_argument(
        "--effective-b0-deg",
        type=float,
        help="Optional override for the pooled no-date effective B0 tilt in degrees.",
    )
    parser.add_argument(
        "--b0-selection",
        choices=["auto", "mean", "median"],
        default="auto",
        help="How to choose the effective B0 when the point file has no per-row dates.",
    )
    parser.add_argument(
        "--max-cmd",
        type=float,
        default=45.0,
        help="Maximum allowed central-meridian distance in degrees.",
    )
    parser.add_argument(
        "--max-lat",
        type=float,
        default=35.0,
        help="Maximum absolute latitude included in the final fit.",
    )
    parser.add_argument(
        "--max-delta-lat",
        type=float,
        default=0.5,
        help="Maximum allowed one-day change in heliographic latitude for a tracked spot.",
    )
    parser.add_argument(
        "--min-sidereal-rate",
        type=float,
        default=13.2,
        help="Minimum accepted sidereal rotation rate in deg/day.",
    )
    parser.add_argument(
        "--max-sidereal-rate",
        type=float,
        default=15.4,
        help="Maximum accepted sidereal rotation rate in deg/day.",
    )
    parser.add_argument("--bin-size", type=float, default=5.0, help="Latitude-bin width in degrees.")
    parser.add_argument(
        "--min-bin-count",
        type=int,
        default=3,
        help="Minimum points required before a latitude bin is plotted or fitted.",
    )
    parser.add_argument(
        "--y-unit",
        choices=["nHz", "deg_day"],
        default="nHz",
        help="Vertical-axis unit for the final plot.",
    )
    parser.add_argument(
        "--hide-reference-curve",
        action="store_true",
        help="Hide the Newton-Nunn-style reference sunspot curve.",
    )
    parser.add_argument("--output-plot", default="solar_rotation_analysis.png", help="Output plot filename.")
    parser.add_argument(
        "--output-csv",
        default="solar_rotation_measurements.csv",
        help="Output CSV containing per-track deprojection results and quality flags.",
    )
    return parser.parse_args()


# ============================================================================
# TIMESTAMP AND TIME RESOLUTION
# ============================================================================

def parse_timestamp_from_name(name):
    """Extract ISO 8601 timestamp from filename.
    
    Looks for pattern YYYY-MM-DDTHH[-_]MM[-_]SS in filename.
    Example: 2026-04-16T17_35_23 → Time object for 2026-04-16 17:35:23
    """
    match = TIMESTAMP_RE.search(Path(name).name)
    if not match:
        raise ValueError(f"Could not parse a timestamp from: {name}")

    parts = match.groupdict()
    return Time(f"{parts['date']}T{parts['hour']}:{parts['minute']}:{parts['second']}")


def normalize_row(raw_row):
    """Normalize CSV row keys to lowercase for case-insensitive column lookup."""
    return {str(key).strip().lower(): str(value).strip() for key, value in raw_row.items() if key is not None}


def get_first_value(row_lookup, names):
    """Get first non-empty value from row_lookup for any key in names list.
    
    Useful for finding optional columns that may have different names.
    """
    for name in names:
        value = row_lookup.get(name)
        if value:
            return value
    return None


def resolve_global_times(args):
    """Resolve global start/end observation times from command-line arguments.
    
    Supports:
    - Direct ISO 8601 times (--start-obstime, --end-obstime)
    - Filenames to parse (--start-image, --end-image)
    - If only one provided, assumes 1-day interval and computes the other
    """
    start_time = None
    end_time = None

    # Try direct time specifications first
    if args.start_obstime:
        start_time = Time(args.start_obstime)
    elif args.start_image:
        start_time = parse_timestamp_from_name(args.start_image)

    if args.end_obstime:
        end_time = Time(args.end_obstime)
    elif args.end_image:
        end_time = parse_timestamp_from_name(args.end_image)

    # If only one time specified, assume 1-day interval
    if start_time is not None and end_time is None:
        end_time = Time(start_time.to_datetime() + timedelta(days=1))
    elif end_time is not None and start_time is None:
        start_time = Time(end_time.to_datetime() - timedelta(days=1))

    return start_time, end_time


def resolve_row_times(row_lookup, global_start, global_end):
    """Resolve per-row start/end observation times.
    
    Priority:
    1. Row-specific times (columns: start_obstime, start_time, start_image, etc.)
    2. Global times from command-line arguments
    3. If only one provided, assume 1-day interval
    
    Returns: (start_time, end_time, source_string) where source indicates whether
    times came from row data or global args
    """
    start_value = get_first_value(row_lookup, ["start_obstime", "start_time", "start_image"])
    end_value = get_first_value(row_lookup, ["end_obstime", "end_time", "end_image"])

    start_time = global_start  # Default to global times
    end_time = global_end
    source = "global"

    # Override with row-specific times if present
    if start_value is not None:
        if "image" in next(name for name in ["start_obstime", "start_time", "start_image"] if row_lookup.get(name) == start_value):
            start_time = parse_timestamp_from_name(start_value)  # Parse filename
        else:
            start_time = Time(start_value)  # Parse ISO 8601 string
        source = "row"

    if end_value is not None:
        if "image" in next(name for name in ["end_obstime", "end_time", "end_image"] if row_lookup.get(name) == end_value):
            end_time = parse_timestamp_from_name(end_value)
        else:
            end_time = Time(end_value)
        source = "row"

    # Fill in missing time if only one provided
    if start_time is not None and end_time is None:
        end_time = Time(start_time.to_datetime() + timedelta(days=1))
    elif end_time is not None and start_time is None:
        start_time = Time(end_time.to_datetime() - timedelta(days=1))

    return start_time, end_time, source


def load_rows(csv_path):
    """Load tracked feature points from CSV file.
    
    Required columns: BX, BY, EX, EY (pixel coordinates of feature start and end)
    Optional columns: start_obstime, end_obstime, start_image, end_image (observation times)
    
    Returns list of dicts with pixel coordinates and raw CSV data.
    """
    rows = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, raw_row in enumerate(reader, start=1):
            lookup = normalize_row(raw_row)  # Normalize to lowercase for case-insensitive matching
            # Verify required columns exist
            missing = [name for name in ["bx", "by", "ex", "ey"] if not lookup.get(name)]
            if missing:
                raise ValueError(
                    f"Row {row_index} in {csv_path} is missing required columns: {', '.join(missing)}"
                )

            rows.append(
                {
                    "row_index": row_index,
                    "BX": float(lookup["bx"]),  # Beginning X pixel coordinate
                    "BY": float(lookup["by"]),  # Beginning Y pixel coordinate
                    "EX": float(lookup["ex"]),  # End X pixel coordinate
                    "EY": float(lookup["ey"]),  # End Y pixel coordinate
                    "raw_lookup": lookup,       # Keep raw data for time resolution
                }
            )
    return rows


def pixel_to_frames_exact(x_px, y_px, obstime, center_x, center_y, radius_px):
    """Convert pixel coordinates to heliographic coordinates using exact ephemeris.
    
    This is the most accurate method: it uses the actual solar radius at the
    observation time and Earth's precise position relative to the Sun.
    
    Process:
    1. Convert pixels to arcseconds using solar radius at obstime
    2. Create 3D Helioprojective coordinate (apparent Sun position as seen from Earth)
    3. Transform to HeliographicStonyhurst (latitude/longitude on Sun, Earth-based)
    4. Transform to HeliographicCarrington (fixed rotating frame, sidereal)
    
    Returns: (hgs, hgc) coordinate objects
    """
    # Get solar radius in arcseconds at observation time
    # This accounts for Earth-Sun distance variations throughout the year
    scale = sun.angular_radius(obstime).to_value(u.arcsec) / radius_px
    
    # Convert pixel offsets to arcseconds
    # x increases to the right, y increases downward in images but upward in astronomy
    tx = (x_px - center_x) * scale * u.arcsec
    ty = (center_y - y_px) * scale * u.arcsec

    # Get Earth's position relative to Sun at obstime
    observer = get_earth(obstime)
    
    # Create 3D Helioprojective coordinate
    # This is the coordinate as it appears from Earth at a specific time
    hpc = SkyCoord(
        tx,
        ty,
        frame=frames.Helioprojective,
        obstime=obstime,
        observer=observer,
    ).make_3d()

    # Transform to HeliographicStonyhurst (HGS):
    # - Latitude/longitude system on the Sun's surface
    # - Stonyhurst: longitude = 0 on the meridian visible from Earth on Jan 1, 1900
    # - "Synodic" frame - rotates with Earth's view
    hgs = hpc.transform_to(frames.HeliographicStonyhurst(obstime=obstime))
    
    # Transform to HeliographicCarrington (HGC):
    # - Fixed rotating frame: longitude = 0 at fixed point on Sun
    # - "Sidereal" frame - fixed to Sun's rotation, not Earth's motion
    # - Best for tracking solar rotation over time
    hgc = hpc.transform_to(frames.HeliographicCarrington(obstime=obstime, observer=observer))
    
    return hgs, hgc


def pixel_to_normalized(x_px, y_px, center_x, center_y, radius_px):
    """Convert pixel coordinates to normalized disk coordinates.
    
    Normalized coordinates: [-1, 1] × [-1, 1] where center is (0,0) and
    the solar disk edge is at radius 1. Used for deprojection calculations.
    """
    return (x_px - center_x) / radius_px, (center_y - y_px) / radius_px


def invert_normalized_point(x_norm, y_norm, b0_rad):
    """Deproject normalized disk point to heliographic latitude/longitude.
    
    This is the inverse of the projection: takes a point on the 2D solar disk image
    and converts it to 3D latitude/longitude coordinates on the Sun's surface.
    
    The B0 angle accounts for the tilt of the Sun's north pole relative to our view:
    - B0 = 0: Sun's north pole points directly at us (no tilt)
    - B0 > 0: North pole tilted toward Earth
    - B0 < 0: North pole tilted away from Earth
    
    Raises ValueError if point is outside the disk (rho² ≥ 1).
    """
    rho_sq = x_norm * x_norm + y_norm * y_norm  # Distance from disk center
    if rho_sq >= 1.0:
        raise ValueError("Point lies outside the normalized solar disk.")

    # cos_c is the cosine of the angle from disk center to the point
    # This gives the 3D distance along the line of sight
    cos_c = math.sqrt(max(0.0, 1.0 - rho_sq))
    
    # Deprojection formulas accounting for B0 tilt
    latitude_rad = math.asin(y_norm * math.cos(b0_rad) + cos_c * math.sin(b0_rad))
    longitude_rad = math.atan2(
        x_norm,
        cos_c * math.cos(b0_rad) - y_norm * math.sin(b0_rad),
    )
    return math.degrees(latitude_rad), math.degrees(longitude_rad)


def wrap_delta_deg(delta_deg):
    """Wrap angle difference to [-180, 180) degree range.
    
    Ensures longitude differences are computed via the shortest arc.
    Example: 350° becomes -10°, 180° becomes -180°
    """
    return ((delta_deg + 180.0) % 360.0) - 180.0


# ============================================================================
# RATE/PERIOD CONVERSION UTILITIES
# ============================================================================
# These functions convert between different representations of rotation rates:
# - deg/day: Angular velocity (most intuitive)
# - nanoHertz (nHz): Frequency representation (used in asteroseismology)
# - days: Rotation period (time for one full 360° rotation)

def deg_day_to_nhz(rate_deg_day):
    """Convert angular velocity (deg/day) to nanoHertz (nHz)."""
    return rate_deg_day * DEG_PER_DAY_TO_NHZ


def nhz_to_deg_day(rate_nhz):
    """Convert frequency (nHz) to angular velocity (deg/day)."""
    return rate_nhz / DEG_PER_DAY_TO_NHZ


def rate_to_period_days(rate_deg_day):
    """Convert angular velocity (deg/day) to rotation period (days).
    
    Example: 14 deg/day → 360/14 ≈ 25.7 days
    """
    rate_deg_day = np.asarray(rate_deg_day, dtype=float)
    return 360.0 / rate_deg_day


def period_days_to_nhz(period_days):
    """Convert rotation period (days) to frequency (nHz)."""
    period_days = np.asarray(period_days, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return deg_day_to_nhz(360.0 / period_days)


def nhz_to_period_days(rate_nhz):
    """Convert frequency (nHz) to rotation period (days)."""
    rate_nhz = np.asarray(rate_nhz, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return rate_to_period_days(nhz_to_deg_day(rate_nhz))


# ============================================================================
# DIFFERENTIAL ROTATION MODEL
# ============================================================================
# Solar rotation varies with latitude following this empirical model:
#   ω(lat) = A - B*sin²(lat) - C*sin⁴(lat)
# where:
#   A = equatorial rotation rate (deg/day)
#   B = latitude dependence coefficient
#   C = higher-order latitude dependence
# Higher B means the equator rotates faster than poles (differential rotation)

def reference_sunspot_rate(latitude_deg):
    """Calculate reference sunspot rotation rate at given latitude.
    
    Uses historical Newton-Nunn model parameters.
    """
    sin_lat_sq = np.sin(np.deg2rad(latitude_deg)) ** 2
    return REFERENCE_SUNSPOT_A - REFERENCE_SUNSPOT_B * sin_lat_sq - REFERENCE_SUNSPOT_C * sin_lat_sq**2


def rotation_model_deg_day(latitude_deg, coefficients):
    """Evaluate differential rotation model at given latitude with fit coefficients.
    
    Args:
        latitude_deg: Heliographic latitude (scalar or array)
        coefficients: [A, B, C] model parameters
    
    Returns: Rotation rate(s) in deg/day
    """
    sin_lat_sq = np.sin(np.deg2rad(latitude_deg)) ** 2
    return coefficients[0] - coefficients[1] * sin_lat_sq - coefficients[2] * sin_lat_sq**2


def fit_constrained_sunspot_curve(bin_latitudes_deg, bin_rates_deg_day, bin_counts):
    """Fit differential rotation model to binned data with constraints.
    
    Uses least_squares with:
    - Weighting by sqrt(bin_counts): More points = more weight
    - Constraints: B ≥ 0 and C ≥ 0 (ensures physical differential rotation)
    - Initial guess based on data max
    
    Args:
        bin_latitudes_deg: Latitude values at bin centers
        bin_rates_deg_day: Median rotation rate in each bin
        bin_counts: Number of points in each bin (for weighting)
    
    Returns: Dict with coefficients, predictions, and R² (or None if too few bins)
    """
    if len(bin_latitudes_deg) < 3:
        return None  # Need at least 3 bins to constrain 3-parameter model

    # Residuals weighted by point count: bins with more data get more influence
    def residuals(params):
        return np.sqrt(bin_counts) * (rotation_model_deg_day(bin_latitudes_deg, params) - bin_rates_deg_day)

    # Initial guess: A = max observed rate, B = typical value, C = small
    initial = np.array([np.max(bin_rates_deg_day), 2.5, 0.1], dtype=float)
    
    # Fit with constraints: A is unbounded, but B ≥ 0 and C ≥ 0
    # This enforces that equatorial rotation is faster than poles
    result = least_squares(
        residuals,
        x0=initial,
        bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf]),  # B and C must be non-negative
        max_nfev=5000,
    )
    
    coefficients = result.x
    predictions = rotation_model_deg_day(bin_latitudes_deg, coefficients)
    
    # Calculate R² (coefficient of determination)
    # R² = 1 - (SS_res / SS_tot)
    # Ranges from 0 (no fit) to 1 (perfect fit)
    ss_res = np.sum((bin_rates_deg_day - predictions) ** 2)
    ss_tot = np.sum((bin_rates_deg_day - np.mean(bin_rates_deg_day)) ** 2)
    r_squared = np.nan if ss_tot == 0 else 1.0 - (ss_res / ss_tot)

    return {
        "coefficients": coefficients,
        "predictions": predictions,
        "r_squared": r_squared,
    }


def collect_folder_ephemeris(images_dir):
    """Collect timestamps and ephemeris data (B0, rsun) from image folder.
    
    Used for POOLED mode when individual track timestamps aren't available.
    Scans folder for *_rotated.tif files, parses timestamps from filenames,
    and calculates solar ephemeris (B0 angle and solar radius) at each time.
    
    B0 = heliographic latitude of the Sun's north pole as viewed from Earth
    rsun = solar radius in arcseconds at observation time (varies with Earth-Sun distance)
    """
    images_path = Path(images_dir)
    if not images_path.exists():
        raise ValueError(f"Image directory not found: {images_dir}")

    entries = []
    seen_times = set()
    
    # Find all rotated TIF files (pre-aligned to standard orientation)
    for path in sorted(images_path.glob("*_rotated.tif")):
        timestamp = parse_timestamp_from_name(path.name)
        key = timestamp.isot
        # Skip duplicate timestamps (in case multiple rotations of same time)
        if key in seen_times:
            continue
        seen_times.add(key)
        
        # Get solar ephemeris at this time
        entries.append(
            {
                "filename": path.name,
                "time": timestamp,
                "b0_deg": sun.B0(timestamp).deg,           # Sun's north pole tilt angle
                "rsun_arcsec": sun.angular_radius(timestamp).arcsec,  # Solar radius
            }
        )

    if not entries:
        raise ValueError(
            "No dated rotated TIFF images were found. The pooled no-date mode needs the dated image folder."
        )

    return entries


def compute_exact_row_measurement(row, start_time, end_time, args):
    """Compute rotation rate from exact observation times (EXACT MODE).
    
    This is the most accurate method:
    - Uses precise ephemeris at each observation time
    - Handles both Synodic (Earth-relative) and Sidereal (Sun-fixed) rates
    - Calculates latitude and central meridian distance (CMD)
    - Validates latitude change over tracking interval
    
    Key outputs:
        synodic_rate: How fast feature moves relative to Earth
        sidereal_rate: How fast feature moves relative to Sun's fixed frame
        latitude_deg: Feature's heliographic latitude (averaged)
        cmd: Central meridian distance (longitude, 0 = visible center)
    """
    delta_days = (end_time - start_time).to_value(u.day)
    if delta_days <= 0:
        raise ValueError("End time must be later than start time.")

    # Convert pixel positions to heliographic coordinates at both times
    start_hgs, start_hgc = pixel_to_frames_exact(
        row["BX"], row["BY"], start_time, args.center_x, args.center_y, args.radius_px
    )
    end_hgs, end_hgc = pixel_to_frames_exact(
        row["EX"], row["EY"], end_time, args.center_x, args.center_y, args.radius_px
    )

    # Calculate longitude displacement in both coordinate frames
    # wrap_at(180°) ensures we measure the shortest arc
    synodic_delta_lon_deg = (end_hgs.lon - start_hgs.lon).wrap_at(180 * u.deg).to_value(u.deg)
    carrington_delta_lon_deg = (end_hgc.lon - start_hgc.lon).wrap_at(180 * u.deg).to_value(u.deg)
    
    # Sidereal rate = Carrington sidereal rate + observed rotation
    sidereal_rate_deg_per_day = CARRINGTON_SIDEREAL_RATE_DEG_PER_DAY + (carrington_delta_lon_deg / delta_days)
    
    # Use mean latitude over the tracking interval
    latitude_deg = 0.5 * (start_hgs.lat.to_value(u.deg) + end_hgs.lat.to_value(u.deg))
    
    # Track how much the latitude changed (should be small)
    delta_lat_deg = (end_hgs.lat - start_hgs.lat).to_value(u.deg)
    
    # Central meridian distance: longitude where 0° is center of disk facing us
    cmd_start_deg = start_hgs.lon.to_value(u.deg)
    cmd_end_deg = end_hgs.lon.to_value(u.deg)

    return {
        "time_delta_days": delta_days,
        "latitude_deg": latitude_deg,
        "delta_lat_deg": delta_lat_deg,
        "cmd_start_deg": cmd_start_deg,
        "cmd_end_deg": cmd_end_deg,
        "abs_cmd_deg": max(abs(cmd_start_deg), abs(cmd_end_deg)),
        "synodic_delta_lon_deg": synodic_delta_lon_deg,
        "carrington_delta_lon_deg": carrington_delta_lon_deg,
        "synodic_rate_deg_per_day": synodic_delta_lon_deg / delta_days,
        "sidereal_rate_deg_per_day": sidereal_rate_deg_per_day,
        "sidereal_rate_nhz": deg_day_to_nhz(sidereal_rate_deg_per_day),
        "sidereal_period_days": rate_to_period_days(sidereal_rate_deg_per_day),
    }


def compute_measurements_exact(rows, args, global_start, global_end):
    """Process all rows using EXACT MODE (per-track observation times).
    
    EXACT MODE uses precise observation times for each tracked feature.
    Times can come from:
    1. Per-row columns (start_obstime, end_obstime)
    2. Per-row filenames (start_image, end_image)
    3. Global command-line arguments
    4. Defaults to 1-day interval if only one time specified
    
    Each row produces a measurement dict with derived quantities and quality flags.
    Returns: (measurements, summary_dict)
    """
    measurements = []
    missing_time_rows = 0
    row_time_rows = 0
    global_time_rows = 0

    for row in rows:
        row_lookup = row["raw_lookup"]
        start_time, end_time, time_source = resolve_row_times(row_lookup, global_start, global_end)

        # Initialize measurement record with row data
        measurement = {
            "row_index": row["row_index"],
            "BX": row["BX"],
            "BY": row["BY"],
            "EX": row["EX"],
            "EY": row["EY"],
            "mode": "exact",
            "start_obstime": start_time.isot if start_time is not None else "",
            "end_obstime": end_time.isot if end_time is not None else "",
            "time_source": time_source,
            "conversion_ok": False,
            "accepted": False,
            "reject_reason": "",
            "effective_b0_deg": "",     # Not used in exact mode
            "effective_b0_source": "",
        }

        # Check if times were resolved
        if start_time is None or end_time is None:
            measurement["reject_reason"] = "missing_observation_time"
            measurements.append(measurement)
            missing_time_rows += 1
            continue

        # Track time source statistics
        if time_source == "row":
            row_time_rows += 1
        else:
            global_time_rows += 1

        # Attempt coordinate conversion and rate calculation
        try:
            derived = compute_exact_row_measurement(row, start_time, end_time, args)
        except Exception:
            measurement["reject_reason"] = "coordinate_conversion_failed"
            measurements.append(measurement)
            continue

        # Add derived quantities to measurement
        measurement.update(derived)
        measurement["conversion_ok"] = True
        
        # Apply quality filters (latitude, CMD, rate, etc.)
        apply_quality_cuts(measurement, args)
        measurements.append(measurement)

    summary = {
        "mode": "exact",
        "missing_time_rows": missing_time_rows,
        "row_time_rows": row_time_rows,
        "global_time_rows": global_time_rows,
    }
    return measurements, summary


def provisional_pooled_measurement(row, b0_deg, args):
    """Compute rotation rate from pixel data using POOLED MODE (no individual timestamps).
    
    POOLED MODE is used when observation times aren't available. Instead:
    - Uses a single B0 angle (heliographic latitude of Sun's north pole) for all tracks
    - Deprojects pixel coordinates using simplified geometry
    - Assumes all tracks are from a 1-day interval
    - Converts Synodic rates (Earth-relative) to Sidereal rates (Sun-fixed)
    
    The B0 angle is typically chosen as the median over the data collection period,
    or auto-selected to minimize latitude drift in tracks.
    """
    b0_rad = math.radians(b0_deg)
    
    # Convert pixel coordinates to normalized disk coordinates [-1, 1]
    begin_x, begin_y = pixel_to_normalized(row["BX"], row["BY"], args.center_x, args.center_y, args.radius_px)
    end_x, end_y = pixel_to_normalized(row["EX"], row["EY"], args.center_x, args.center_y, args.radius_px)
    
    # Deproject to get latitude/longitude at beginning and end
    begin_lat_deg, begin_lon_deg = invert_normalized_point(begin_x, begin_y, b0_rad)
    end_lat_deg, end_lon_deg = invert_normalized_point(end_x, end_y, b0_rad)

    # Calculate longitude displacement, handling 360° wrap
    synodic_delta_lon_deg = wrap_delta_deg(end_lon_deg - begin_lon_deg)
    
    # Convert Synodic rate (observed from Earth) to Sidereal rate (fixed to Sun)
    # Sidereal = Synodic + Earth's orbital rate
    sidereal_rate_deg_per_day = synodic_delta_lon_deg + EARTH_ORBITAL_RATE_DEG_PER_DAY
    
    # Use mean latitude over tracking interval
    latitude_deg = 0.5 * (begin_lat_deg + end_lat_deg)
    delta_lat_deg = end_lat_deg - begin_lat_deg

    return {
        "latitude_deg": latitude_deg,
        "delta_lat_deg": delta_lat_deg,
        "cmd_start_deg": begin_lon_deg,
        "cmd_end_deg": end_lon_deg,
        "abs_cmd_deg": max(abs(begin_lon_deg), abs(end_lon_deg)),
        "synodic_delta_lon_deg": synodic_delta_lon_deg,
        "carrington_delta_lon_deg": sidereal_rate_deg_per_day - CARRINGTON_SIDEREAL_RATE_DEG_PER_DAY,
        "synodic_rate_deg_per_day": synodic_delta_lon_deg,
        "sidereal_rate_deg_per_day": sidereal_rate_deg_per_day,
        "sidereal_rate_nhz": deg_day_to_nhz(sidereal_rate_deg_per_day),
        "sidereal_period_days": rate_to_period_days(sidereal_rate_deg_per_day),
    }


def choose_effective_b0(rows, args, ephemeris_entries):
    """Determine the best B0 angle to use for POOLED MODE analysis.
    
    Methods (in priority order):
    1. User override (--effective-b0-deg): Take user's value
    2. User selection (--b0-selection):
       - "mean": Average B0 from all folder images
       - "median": Median B0 from all folder images  
       - "auto": Choose B0 that minimizes latitude drift in tracks
           (features shouldn't change latitude much in 1 day if B0 is right)
    
    Returns: (b0_value_deg, selection_method_string)
    """
    # User override takes precedence
    if args.effective_b0_deg is not None:
        return args.effective_b0_deg, "user_override"

    # Extract B0 values from all folder images
    b0_values = np.array([item["b0_deg"] for item in ephemeris_entries], dtype=float)
    
    # Method 1: Mean B0
    if args.b0_selection == "mean":
        return float(np.mean(b0_values)), "folder_mean"
    
    # Method 2: Median B0
    if args.b0_selection == "median":
        return float(np.median(b0_values)), "folder_median"

    # Method 3: Auto - find B0 that minimizes latitude drift
    # This checks which B0 produces tracks with smallest delta-latitude
    best_score = None
    best_entry = None
    for entry in ephemeris_entries:
        candidate_b0 = entry["b0_deg"]
        delta_latitudes = []
        for row in rows:
            try:
                derived = provisional_pooled_measurement(row, candidate_b0, args)
            except Exception:
                continue

            # Apply loose quality filters to get reasonable subset
            if abs(derived["latitude_deg"]) > 50.0:  # Skip high latitudes
                continue
            if derived["abs_cmd_deg"] > 60.0:  # Skip far-from-center tracks
                continue
            if derived["sidereal_rate_deg_per_day"] < 12.0 or derived["sidereal_rate_deg_per_day"] > 16.0:
                continue  # Skip unrealistic rates
            delta_latitudes.append(abs(derived["delta_lat_deg"]))

        # Need enough points to evaluate this B0
        if len(delta_latitudes) < 10:
            continue

        # Score is (median latitude drift, timestamp) - prefer lower drift
        score = (float(np.median(delta_latitudes)), entry["time"].isot)
        if best_score is None or score < best_score:
            best_score = score
            best_entry = entry

    # If auto selection succeeded, use best B0; otherwise fall back to median
    if best_entry is None:
        return float(np.median(b0_values)), "folder_median_fallback"
    return float(best_entry["b0_deg"]), f"auto_from_{best_entry['filename']}"


def compute_measurements_pooled(rows, args, ephemeris_entries):
    """Process all rows using POOLED MODE (no per-row observation times).
    
    POOLED MODE flow:
    1. Choose effective B0 angle (see choose_effective_b0)
    2. Apply provisional_pooled_measurement to each row with that B0
    3. Apply quality filters
    
    Returns: (measurements, summary_dict) for all rows
    """
    effective_b0_deg, b0_source = choose_effective_b0(rows, args, ephemeris_entries)
    measurements = []

    for row in rows:
        measurement = {
            "row_index": row["row_index"],
            "BX": row["BX"],
            "BY": row["BY"],
            "EX": row["EX"],
            "EY": row["EY"],
            "mode": "pooled",
            "start_obstime": "",      # No times in pooled mode
            "end_obstime": "",
            "time_source": "folder_dates",
            "time_delta_days": 1.0,   # Assume 1-day tracking interval
            "conversion_ok": False,
            "accepted": False,
            "reject_reason": "",
            "effective_b0_deg": effective_b0_deg,
            "effective_b0_source": b0_source,
        }

        try:
            derived = provisional_pooled_measurement(row, effective_b0_deg, args)
        except Exception:
            measurement["reject_reason"] = "coordinate_conversion_failed"
            measurements.append(measurement)
            continue

        measurement.update(derived)
        measurement["conversion_ok"] = True
        apply_quality_cuts(measurement, args)
        measurements.append(measurement)

    # Summarize folder ephemeris for reporting
    b0_values = np.array([item["b0_deg"] for item in ephemeris_entries], dtype=float)
    times = [item["time"] for item in ephemeris_entries]
    summary = {
        "mode": "pooled",
        "missing_time_rows": len(rows),
        "row_time_rows": 0,
        "global_time_rows": 0,
        "folder_date_start": min(times).isot,
        "folder_date_end": max(times).isot,
        "folder_b0_min_deg": float(np.min(b0_values)),
        "folder_b0_max_deg": float(np.max(b0_values)),
        "folder_b0_median_deg": float(np.median(b0_values)),
        "effective_b0_deg": effective_b0_deg,
        "effective_b0_source": b0_source,
    }
    return measurements, summary


def apply_quality_cuts(measurement, args):
    """Apply quality filters to determine if a measurement should be accepted.
    
    Quality cuts prevent systematic biases:
    - off_disk: Tracked features must stay on the solar disk
    - latitude_cut: Limits to mid-latitudes for better accuracy
    - cmd_cut: Restricts to features near Earth-facing disk center (better precision)
    - delta_lat_cut: Features shouldn't move much in latitude (1-day motion)
    - sidereal_rate_cut: Realistic rotation rates for solar features
    
    Sets measurement["accepted"] and measurement["reject_reason"] fields.
    """
    reject_reasons = []
    
    # Calculate how far from disk center the start and end points are
    # Expressed as fraction of solar radius
    start_radius_fraction = np.hypot(measurement["BX"] - args.center_x, measurement["BY"] - args.center_y) / args.radius_px
    end_radius_fraction = np.hypot(measurement["EX"] - args.center_x, measurement["EY"] - args.center_y) / args.radius_px
    measurement["start_radius_fraction"] = start_radius_fraction
    measurement["end_radius_fraction"] = end_radius_fraction

    # Quality check 1: Both points must be inside the solar disk
    if start_radius_fraction > 1.0 or end_radius_fraction > 1.0:
        reject_reasons.append("off_disk")
    
    # Quality check 2: Latitude must be within allowed range
    if abs(measurement["latitude_deg"]) > args.max_lat:
        reject_reasons.append("latitude_cut")
    
    # Quality check 3: Central meridian distance too large
    # (feature too far from Earth-facing center, higher foreshortening error)
    if measurement["abs_cmd_deg"] > args.max_cmd:
        reject_reasons.append("cmd_cut")
    
    # Quality check 4: Latitude shouldn't change much over 1 day
    if abs(measurement["delta_lat_deg"]) > args.max_delta_lat:
        reject_reasons.append("delta_lat_cut")
    
    # Quality check 5: Rotation rate must be physically reasonable
    if (
        measurement["sidereal_rate_deg_per_day"] < args.min_sidereal_rate
        or measurement["sidereal_rate_deg_per_day"] > args.max_sidereal_rate
    ):
        reject_reasons.append("sidereal_rate_cut")

    # Set acceptance status
    measurement["accepted"] = len(reject_reasons) == 0
    measurement["reject_reason"] = ";".join(reject_reasons)  # Multiple reasons separated by semicolon


def summarize_bins(measurements, args):
    """Create latitude bins and compute median rotation rate in each bin.
    
    Binning strategy:
    - Use absolute latitude (combine north and south by symmetry)
    - Bins of size --bin-size degrees
    - Only keep bins with at least --min-bin-count points
    - Use medians (robust to outliers) not means
    
    Returns: (bin_centers, bin_medians, bin_counts) arrays
    """
    # Only use accepted measurements
    accepted = [item for item in measurements if item["accepted"]]
    if not accepted:
        return np.array([]), np.array([]), np.array([])

    # Extract absolute latitude and sidereal rates
    abs_latitudes = np.array([abs(item["latitude_deg"]) for item in accepted], dtype=float)
    sidereal_rates = np.array([item["sidereal_rate_deg_per_day"] for item in accepted], dtype=float)

    # Create bin edges: 0, bin_size, 2*bin_size, ..., max_lat
    edges = np.arange(0.0, args.max_lat + args.bin_size, args.bin_size)
    
    centers = []
    medians = []
    counts = []
    
    # Process each bin
    for lower, upper in zip(edges[:-1], edges[1:]):
        # Find all points in this bin
        in_bin = (abs_latitudes >= lower) & (abs_latitudes < upper)
        count = int(np.count_nonzero(in_bin))
        
        # Skip bins with too few points (unreliable)
        if count < args.min_bin_count:
            continue
        
        # Record bin center, median rate, and count
        centers.append(0.5 * (lower + upper))
        medians.append(float(np.median(sidereal_rates[in_bin])))
        counts.append(count)

    return np.array(centers), np.array(medians), np.array(counts)


def convert_plot_y(rate_deg_day, y_unit):
    """Convert rotation rate to desired plot units.
    
    Supports:
    - "deg_day": Degrees per day (intuitive, shows angular velocity)
    - "nHz": Nanohertz (frequency representation, easier for period axis)
    """
    if y_unit == "nHz":
        return deg_day_to_nhz(rate_deg_day)
    return rate_deg_day


def y_axis_label(y_unit):
    """Generate y-axis label text based on unit choice."""
    if y_unit == "nHz":
        return "Sidereal rotation rate (nHz)"
    return "Sidereal rotation rate (deg/day)"


def write_measurements_csv(measurements, output_csv):
    """Write detailed per-track measurement results to CSV file.
    
    Includes all measurements (accepted and rejected) with quality flags,
    coordinates, derived quantities, and rejection reasons.
    Useful for:
    - Auditing individual tracks
    - Post-processing analysis
    - Understanding why tracks were rejected
    """
    fieldnames = [
        "row_index",
        "BX", "BY", "EX", "EY",                    # Pixel coordinates
        "mode", "start_obstime", "end_obstime",   # Time info
        "time_source", "time_delta_days",         # Time provenance
        "conversion_ok",                            # Did coordinate conversion succeed?
        "accepted",                                 # Passed quality cuts?
        "reject_reason",                            # Why rejected (if applicable)
        "effective_b0_deg", "effective_b0_source",  # B0 info (pooled mode)
        "start_radius_fraction", "end_radius_fraction",  # Disk position
        "latitude_deg", "delta_lat_deg",           # Latitude measurements
        "cmd_start_deg", "cmd_end_deg", "abs_cmd_deg",  # Central meridian distance
        "synodic_delta_lon_deg", "carrington_delta_lon_deg",  # Longitude changes
        "synodic_rate_deg_per_day", "sidereal_rate_deg_per_day",  # Rotation rates
        "sidereal_rate_nhz", "sidereal_period_days",  # Alternative rate representations
    ]

    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for measurement in measurements:
            writer.writerow(measurement)


def build_plot(measurements, bin_centers, bin_medians_deg_day, bin_counts, fit_result, args, summary):
    """Create comprehensive solar rotation analysis plot.
    
    The plot shows:
    1. Rejected measurements (gray dots) - why they failed quality cuts
    2. Accepted measurements (blue dots) - high-quality data points
    3. Bin medians (black line with circles) - median rate in each latitude bin
    4. Fitted curve (red line) - best-fit differential rotation model
    5. Reference curve (gray dashed) - historical Newton-Nunn model for comparison
    6. Secondary y-axis (right) - rotation periods in days (when units are nHz)
    7. Info box - shows analysis parameters and results
    """
    # Separate accepted and rejected measurements
    accepted = [item for item in measurements if item["accepted"]]
    rejected = [item for item in measurements if item["conversion_ok"] and not item["accepted"]]

    figure, axis = plt.subplots(figsize=(10.5, 6.8))

    # Plot rejected measurements in light gray
    if rejected:
        rejected_lat = np.array([abs(item["latitude_deg"]) for item in rejected], dtype=float)
        rejected_rate_deg_day = np.array([item["sidereal_rate_deg_per_day"] for item in rejected], dtype=float)
        axis.scatter(
            rejected_lat,
            convert_plot_y(rejected_rate_deg_day, args.y_unit),
            s=24,
            color="#b9b9b9",
            alpha=0.45,
            label="Rejected by quality cuts",
        )

    # Plot accepted measurements in blue
    accepted_lat = np.array([abs(item["latitude_deg"]) for item in accepted], dtype=float)
    accepted_rate_deg_day = np.array([item["sidereal_rate_deg_per_day"] for item in accepted], dtype=float)
    axis.scatter(
        accepted_lat,
        convert_plot_y(accepted_rate_deg_day, args.y_unit),
        s=30,
        color="#4c78a8",
        alpha=0.78,
        label="Accepted sunspot tracks",
    )

    # Plot bin medians as a black line
    if len(bin_centers) > 0:
        axis.plot(
            bin_centers,
            convert_plot_y(bin_medians_deg_day, args.y_unit),
            color="black",
            linewidth=1.6,
            marker="o",
            markersize=5,
            label=f"{int(args.bin_size)}-deg median bins",
        )

    # Plot fitted differential rotation curve in red
    if fit_result is not None:
        lat_grid = np.linspace(0.0, args.max_lat, 300)  # Smooth latitude grid for curve
        fit_deg_day = rotation_model_deg_day(lat_grid, fit_result["coefficients"])
        axis.plot(
            lat_grid,
            convert_plot_y(fit_deg_day, args.y_unit),
            color="#e45756",
            linewidth=2.3,
            label="Constrained fit to bin medians",
        )

    # Plot reference sunspot curve for comparison (optional)
    if not args.hide_reference_curve:
        lat_grid = np.linspace(0.0, args.max_lat, 300)
        reference_deg_day = reference_sunspot_rate(lat_grid)
        axis.plot(
            lat_grid,
            convert_plot_y(reference_deg_day, args.y_unit),
            color="#6f6f6f",
            linewidth=1.3,
            linestyle="--",
            alpha=0.9,
            label="Reference sunspot curve",
        )

    # Configure axes
    axis.set_xlim(0.0, args.max_lat)
    axis.set_xlabel("Absolute heliographic latitude (deg)")
    axis.set_ylabel(y_axis_label(args.y_unit))
    axis.set_title("Sunspot differential rotation from tracked points")
    axis.set_ylim(
        convert_plot_y(args.min_sidereal_rate - 0.35, args.y_unit),
        convert_plot_y(args.max_sidereal_rate + 0.35, args.y_unit),
    )
    axis.grid(True, alpha=0.25)
    axis.legend(loc="lower left", fontsize=9)

    # Add secondary y-axis for period (days) when using nHz
    if args.y_unit == "nHz":
        secondary = axis.secondary_yaxis("right", functions=(nhz_to_period_days, period_days_to_nhz))
        secondary.set_ylabel("Sidereal period (days)")

    # Build info box with analysis parameters and results
    info_lines = []
    if summary["mode"] == "pooled":
        info_lines.append("Mode: pooled no-date approximation")
        info_lines.append(f"Folder dates: {summary['folder_date_start'][:10]} to {summary['folder_date_end'][:10]}")
        info_lines.append(
            f"B0 range: {summary['folder_b0_min_deg']:.2f} to {summary['folder_b0_max_deg']:.2f} deg"
        )
        info_lines.append(
            f"Chosen B0: {summary['effective_b0_deg']:.2f} deg ({summary['effective_b0_source']})"
        )
    else:
        info_lines.append("Mode: exact per-track observation times")

    # Quality cut parameters
    info_lines.extend(
        [
            f"Accepted points = {len(accepted)}",
            f"|CMD| <= {args.max_cmd:.0f} deg",
            f"|delta latitude| <= {args.max_delta_lat:.2f} deg/day",
            f"Sidereal rate window = {args.min_sidereal_rate:.1f} to {args.max_sidereal_rate:.1f} deg/day",
            f"Bin threshold = {args.min_bin_count} points",
        ]
    )

    # Fitted model equation and quality metrics
    if fit_result is not None:
        coeff_a, coeff_b, coeff_c = fit_result["coefficients"]
        info_lines.append(
            f"omega = {coeff_a:.3f} - {coeff_b:.3f} sin^2(lat) - {coeff_c:.3f} sin^4(lat) deg/day"
        )
        if not np.isnan(fit_result["r_squared"]):
            info_lines.append(f"Bin-median R^2 = {fit_result['r_squared']:.3f}")

    # Place info box in upper right corner
    axis.text(
        0.98,
        0.98,
        "\n".join(info_lines),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    plt.tight_layout()
    figure.savefig(args.output_plot, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main():
    """Main analysis pipeline.
    
    Workflow:
    1. Parse command-line arguments
    2. Load tracked feature points from CSV
    3. Resolve global observation times from arguments/images
    4. Attempt EXACT MODE analysis (per-row times if available)
    5. If all rows are missing times, fall back to POOLED MODE
    6. Apply quality cuts to filter unreliable measurements
    7. Bin data by latitude and compute medians
    8. Fit differential rotation model to bin medians
    9. Create publication-quality plot
    10. Write detailed CSV of all measurements
    11. Print summary statistics
    """
    # Parse all command-line arguments
    args = parse_args()
    
    # Load tracked feature points from CSV
    rows = load_rows(args.points_csv)
    
    # Try to determine global observation times from arguments
    global_start, global_end = resolve_global_times(args)
    
    # Attempt EXACT MODE analysis using per-row times
    exact_measurements, exact_summary = compute_measurements_exact(rows, args, global_start, global_end)

    # If ALL rows are missing times, use POOLED MODE instead
    if exact_summary["missing_time_rows"] == len(rows):
        ephemeris_entries = collect_folder_ephemeris(args.images_dir)
        measurements, summary = compute_measurements_pooled(rows, args, ephemeris_entries)
    else:
        measurements, summary = exact_measurements, exact_summary

    # Check if we have enough accepted data
    accepted = [item for item in measurements if item["accepted"]]
    if len(accepted) < args.min_bin_count:
        raise ValueError(
            "Too few accepted points remained after the sunspot quality cuts. "
            "Loosen the filters or provide more tracked spots."
        )

    # Bin data by latitude and compute median rates
    bin_centers, bin_medians_deg_day, bin_counts = summarize_bins(measurements, args)
    
    # Fit differential rotation model to bin medians
    fit_result = fit_constrained_sunspot_curve(bin_centers, bin_medians_deg_day, bin_counts)

    # Generate publication-quality plot
    build_plot(measurements, bin_centers, bin_medians_deg_day, bin_counts, fit_result, args, summary)
    
    # Write detailed CSV with all measurements
    write_measurements_csv(measurements, args.output_csv)

    # Print summary to console
    print("Solar rotation analysis complete.")
    print(f"  Mode:                    {summary['mode']}")
    print(f"  Input rows:              {len(rows)}")
    print(f"  Accepted points:         {len(accepted)}")
    print(f"  Plotted bins:            {len(bin_centers)}")
    if summary["mode"] == "pooled":
        print(f"  Folder date start:       {summary['folder_date_start']}")
        print(f"  Folder date end:         {summary['folder_date_end']}")
        print(
            "  Folder B0 range:         "
            f"{summary['folder_b0_min_deg']:.5f} to {summary['folder_b0_max_deg']:.5f} deg"
        )
        print(
            "  Chosen effective B0:     "
            f"{summary['effective_b0_deg']:.5f} deg ({summary['effective_b0_source']})"
        )
    else:
        print(f"  Rows using per-row times:{summary['row_time_rows']}")
        print(f"  Rows using global times: {summary['global_time_rows']}")
    if fit_result is not None:
        coeff_a, coeff_b, coeff_c = fit_result["coefficients"]
        print(
            "  Constrained fit:         "
            f"omega = {coeff_a:.5f} - {coeff_b:.5f} sin^2(lat) - {coeff_c:.5f} sin^4(lat) deg/day"
        )
        if not np.isnan(fit_result["r_squared"]):
            print(f"  Bin-median R^2:          {fit_result['r_squared']:.5f}")
    print(f"  Plot saved to:           {args.output_plot}")
    print(f"  Table saved to:          {args.output_csv}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
