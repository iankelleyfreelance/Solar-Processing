import argparse
import csv
import math
import re
from datetime import timedelta
from pathlib import Path

import astropy.units as u
import matplotlib
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from scipy.optimize import least_squares
from sunpy.coordinates import frames, sun
from sunpy.coordinates.ephemeris import get_earth

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TIMESTAMP_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})T(?P<hour>\d{2})[-_](?P<minute>\d{2})[-_](?P<second>\d{2})"
)
CARRINGTON_SIDEREAL_RATE_DEG_PER_DAY = 360.0 / 25.38
EARTH_ORBITAL_RATE_DEG_PER_DAY = 360.0 / 365.2422
DEG_PER_DAY_TO_NHZ = 1e9 / (360.0 * 86400.0)
REFERENCE_SUNSPOT_A = 14.38
REFERENCE_SUNSPOT_B = 2.96
REFERENCE_SUNSPOT_C = 0.0


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


def parse_timestamp_from_name(name):
    match = TIMESTAMP_RE.search(Path(name).name)
    if not match:
        raise ValueError(f"Could not parse a timestamp from: {name}")

    parts = match.groupdict()
    return Time(f"{parts['date']}T{parts['hour']}:{parts['minute']}:{parts['second']}")


def normalize_row(raw_row):
    return {str(key).strip().lower(): str(value).strip() for key, value in raw_row.items() if key is not None}


def get_first_value(row_lookup, names):
    for name in names:
        value = row_lookup.get(name)
        if value:
            return value
    return None


def resolve_global_times(args):
    start_time = None
    end_time = None

    if args.start_obstime:
        start_time = Time(args.start_obstime)
    elif args.start_image:
        start_time = parse_timestamp_from_name(args.start_image)

    if args.end_obstime:
        end_time = Time(args.end_obstime)
    elif args.end_image:
        end_time = parse_timestamp_from_name(args.end_image)

    if start_time is not None and end_time is None:
        end_time = Time(start_time.to_datetime() + timedelta(days=1))
    elif end_time is not None and start_time is None:
        start_time = Time(end_time.to_datetime() - timedelta(days=1))

    return start_time, end_time


def resolve_row_times(row_lookup, global_start, global_end):
    start_value = get_first_value(row_lookup, ["start_obstime", "start_time", "start_image"])
    end_value = get_first_value(row_lookup, ["end_obstime", "end_time", "end_image"])

    start_time = global_start
    end_time = global_end
    source = "global"

    if start_value is not None:
        if "image" in next(name for name in ["start_obstime", "start_time", "start_image"] if row_lookup.get(name) == start_value):
            start_time = parse_timestamp_from_name(start_value)
        else:
            start_time = Time(start_value)
        source = "row"

    if end_value is not None:
        if "image" in next(name for name in ["end_obstime", "end_time", "end_image"] if row_lookup.get(name) == end_value):
            end_time = parse_timestamp_from_name(end_value)
        else:
            end_time = Time(end_value)
        source = "row"

    if start_time is not None and end_time is None:
        end_time = Time(start_time.to_datetime() + timedelta(days=1))
    elif end_time is not None and start_time is None:
        start_time = Time(end_time.to_datetime() - timedelta(days=1))

    return start_time, end_time, source


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, raw_row in enumerate(reader, start=1):
            lookup = normalize_row(raw_row)
            missing = [name for name in ["bx", "by", "ex", "ey"] if not lookup.get(name)]
            if missing:
                raise ValueError(
                    f"Row {row_index} in {csv_path} is missing required columns: {', '.join(missing)}"
                )

            rows.append(
                {
                    "row_index": row_index,
                    "BX": float(lookup["bx"]),
                    "BY": float(lookup["by"]),
                    "EX": float(lookup["ex"]),
                    "EY": float(lookup["ey"]),
                    "raw_lookup": lookup,
                }
            )
    return rows


def pixel_to_frames_exact(x_px, y_px, obstime, center_x, center_y, radius_px):
    scale = sun.angular_radius(obstime).to_value(u.arcsec) / radius_px
    tx = (x_px - center_x) * scale * u.arcsec
    ty = (center_y - y_px) * scale * u.arcsec

    observer = get_earth(obstime)
    hpc = SkyCoord(
        tx,
        ty,
        frame=frames.Helioprojective,
        obstime=obstime,
        observer=observer,
    ).make_3d()

    hgs = hpc.transform_to(frames.HeliographicStonyhurst(obstime=obstime))
    hgc = hpc.transform_to(frames.HeliographicCarrington(obstime=obstime, observer=observer))
    return hgs, hgc


def pixel_to_normalized(x_px, y_px, center_x, center_y, radius_px):
    return (x_px - center_x) / radius_px, (center_y - y_px) / radius_px


def invert_normalized_point(x_norm, y_norm, b0_rad):
    rho_sq = x_norm * x_norm + y_norm * y_norm
    if rho_sq >= 1.0:
        raise ValueError("Point lies outside the normalized solar disk.")

    cos_c = math.sqrt(max(0.0, 1.0 - rho_sq))
    latitude_rad = math.asin(y_norm * math.cos(b0_rad) + cos_c * math.sin(b0_rad))
    longitude_rad = math.atan2(
        x_norm,
        cos_c * math.cos(b0_rad) - y_norm * math.sin(b0_rad),
    )
    return math.degrees(latitude_rad), math.degrees(longitude_rad)


def wrap_delta_deg(delta_deg):
    return ((delta_deg + 180.0) % 360.0) - 180.0


def deg_day_to_nhz(rate_deg_day):
    return rate_deg_day * DEG_PER_DAY_TO_NHZ


def nhz_to_deg_day(rate_nhz):
    return rate_nhz / DEG_PER_DAY_TO_NHZ


def rate_to_period_days(rate_deg_day):
    rate_deg_day = np.asarray(rate_deg_day, dtype=float)
    return 360.0 / rate_deg_day


def period_days_to_nhz(period_days):
    period_days = np.asarray(period_days, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return deg_day_to_nhz(360.0 / period_days)


def nhz_to_period_days(rate_nhz):
    rate_nhz = np.asarray(rate_nhz, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return rate_to_period_days(nhz_to_deg_day(rate_nhz))


def reference_sunspot_rate(latitude_deg):
    sin_lat_sq = np.sin(np.deg2rad(latitude_deg)) ** 2
    return REFERENCE_SUNSPOT_A - REFERENCE_SUNSPOT_B * sin_lat_sq - REFERENCE_SUNSPOT_C * sin_lat_sq**2


def rotation_model_deg_day(latitude_deg, coefficients):
    sin_lat_sq = np.sin(np.deg2rad(latitude_deg)) ** 2
    return coefficients[0] - coefficients[1] * sin_lat_sq - coefficients[2] * sin_lat_sq**2


def fit_constrained_sunspot_curve(bin_latitudes_deg, bin_rates_deg_day, bin_counts):
    if len(bin_latitudes_deg) < 3:
        return None

    def residuals(params):
        return np.sqrt(bin_counts) * (rotation_model_deg_day(bin_latitudes_deg, params) - bin_rates_deg_day)

    initial = np.array([np.max(bin_rates_deg_day), 2.5, 0.1], dtype=float)
    result = least_squares(
        residuals,
        x0=initial,
        bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf]),
        max_nfev=5000,
    )
    coefficients = result.x
    predictions = rotation_model_deg_day(bin_latitudes_deg, coefficients)
    ss_res = np.sum((bin_rates_deg_day - predictions) ** 2)
    ss_tot = np.sum((bin_rates_deg_day - np.mean(bin_rates_deg_day)) ** 2)
    r_squared = np.nan if ss_tot == 0 else 1.0 - (ss_res / ss_tot)

    return {
        "coefficients": coefficients,
        "predictions": predictions,
        "r_squared": r_squared,
    }


def collect_folder_ephemeris(images_dir):
    images_path = Path(images_dir)
    if not images_path.exists():
        raise ValueError(f"Image directory not found: {images_dir}")

    entries = []
    seen_times = set()
    for path in sorted(images_path.glob("*_rotated.tif")):
        timestamp = parse_timestamp_from_name(path.name)
        key = timestamp.isot
        if key in seen_times:
            continue
        seen_times.add(key)
        entries.append(
            {
                "filename": path.name,
                "time": timestamp,
                "b0_deg": sun.B0(timestamp).deg,
                "rsun_arcsec": sun.angular_radius(timestamp).arcsec,
            }
        )

    if not entries:
        raise ValueError(
            "No dated rotated TIFF images were found. The pooled no-date mode needs the dated image folder."
        )

    return entries


def compute_exact_row_measurement(row, start_time, end_time, args):
    delta_days = (end_time - start_time).to_value(u.day)
    if delta_days <= 0:
        raise ValueError("End time must be later than start time.")

    start_hgs, start_hgc = pixel_to_frames_exact(
        row["BX"], row["BY"], start_time, args.center_x, args.center_y, args.radius_px
    )
    end_hgs, end_hgc = pixel_to_frames_exact(
        row["EX"], row["EY"], end_time, args.center_x, args.center_y, args.radius_px
    )

    synodic_delta_lon_deg = (end_hgs.lon - start_hgs.lon).wrap_at(180 * u.deg).to_value(u.deg)
    carrington_delta_lon_deg = (end_hgc.lon - start_hgc.lon).wrap_at(180 * u.deg).to_value(u.deg)
    sidereal_rate_deg_per_day = CARRINGTON_SIDEREAL_RATE_DEG_PER_DAY + (carrington_delta_lon_deg / delta_days)
    latitude_deg = 0.5 * (start_hgs.lat.to_value(u.deg) + end_hgs.lat.to_value(u.deg))
    delta_lat_deg = (end_hgs.lat - start_hgs.lat).to_value(u.deg)
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
    measurements = []
    missing_time_rows = 0
    row_time_rows = 0
    global_time_rows = 0

    for row in rows:
        row_lookup = row["raw_lookup"]
        start_time, end_time, time_source = resolve_row_times(row_lookup, global_start, global_end)

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
            "effective_b0_deg": "",
            "effective_b0_source": "",
        }

        if start_time is None or end_time is None:
            measurement["reject_reason"] = "missing_observation_time"
            measurements.append(measurement)
            missing_time_rows += 1
            continue

        if time_source == "row":
            row_time_rows += 1
        else:
            global_time_rows += 1

        try:
            derived = compute_exact_row_measurement(row, start_time, end_time, args)
        except Exception:
            measurement["reject_reason"] = "coordinate_conversion_failed"
            measurements.append(measurement)
            continue

        measurement.update(derived)
        measurement["conversion_ok"] = True
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
    b0_rad = math.radians(b0_deg)
    begin_x, begin_y = pixel_to_normalized(row["BX"], row["BY"], args.center_x, args.center_y, args.radius_px)
    end_x, end_y = pixel_to_normalized(row["EX"], row["EY"], args.center_x, args.center_y, args.radius_px)
    begin_lat_deg, begin_lon_deg = invert_normalized_point(begin_x, begin_y, b0_rad)
    end_lat_deg, end_lon_deg = invert_normalized_point(end_x, end_y, b0_rad)

    synodic_delta_lon_deg = wrap_delta_deg(end_lon_deg - begin_lon_deg)
    sidereal_rate_deg_per_day = synodic_delta_lon_deg + EARTH_ORBITAL_RATE_DEG_PER_DAY
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
    if args.effective_b0_deg is not None:
        return args.effective_b0_deg, "user_override"

    b0_values = np.array([item["b0_deg"] for item in ephemeris_entries], dtype=float)
    if args.b0_selection == "mean":
        return float(np.mean(b0_values)), "folder_mean"
    if args.b0_selection == "median":
        return float(np.median(b0_values)), "folder_median"

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

            if abs(derived["latitude_deg"]) > 50.0:
                continue
            if derived["abs_cmd_deg"] > 60.0:
                continue
            if derived["sidereal_rate_deg_per_day"] < 12.0 or derived["sidereal_rate_deg_per_day"] > 16.0:
                continue
            delta_latitudes.append(abs(derived["delta_lat_deg"]))

        if len(delta_latitudes) < 10:
            continue

        score = (float(np.median(delta_latitudes)), entry["time"].isot)
        if best_score is None or score < best_score:
            best_score = score
            best_entry = entry

    if best_entry is None:
        return float(np.median(b0_values)), "folder_median_fallback"
    return float(best_entry["b0_deg"]), f"auto_from_{best_entry['filename']}"


def compute_measurements_pooled(rows, args, ephemeris_entries):
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
            "start_obstime": "",
            "end_obstime": "",
            "time_source": "folder_dates",
            "time_delta_days": 1.0,
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
    reject_reasons = []
    start_radius_fraction = np.hypot(measurement["BX"] - args.center_x, measurement["BY"] - args.center_y) / args.radius_px
    end_radius_fraction = np.hypot(measurement["EX"] - args.center_x, measurement["EY"] - args.center_y) / args.radius_px
    measurement["start_radius_fraction"] = start_radius_fraction
    measurement["end_radius_fraction"] = end_radius_fraction

    if start_radius_fraction > 1.0 or end_radius_fraction > 1.0:
        reject_reasons.append("off_disk")
    if abs(measurement["latitude_deg"]) > args.max_lat:
        reject_reasons.append("latitude_cut")
    if measurement["abs_cmd_deg"] > args.max_cmd:
        reject_reasons.append("cmd_cut")
    if abs(measurement["delta_lat_deg"]) > args.max_delta_lat:
        reject_reasons.append("delta_lat_cut")
    if (
        measurement["sidereal_rate_deg_per_day"] < args.min_sidereal_rate
        or measurement["sidereal_rate_deg_per_day"] > args.max_sidereal_rate
    ):
        reject_reasons.append("sidereal_rate_cut")

    measurement["accepted"] = len(reject_reasons) == 0
    measurement["reject_reason"] = ";".join(reject_reasons)


def summarize_bins(measurements, args):
    accepted = [item for item in measurements if item["accepted"]]
    if not accepted:
        return np.array([]), np.array([]), np.array([])

    abs_latitudes = np.array([abs(item["latitude_deg"]) for item in accepted], dtype=float)
    sidereal_rates = np.array([item["sidereal_rate_deg_per_day"] for item in accepted], dtype=float)

    edges = np.arange(0.0, args.max_lat + args.bin_size, args.bin_size)
    centers = []
    medians = []
    counts = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (abs_latitudes >= lower) & (abs_latitudes < upper)
        count = int(np.count_nonzero(in_bin))
        if count < args.min_bin_count:
            continue
        centers.append(0.5 * (lower + upper))
        medians.append(float(np.median(sidereal_rates[in_bin])))
        counts.append(count)

    return np.array(centers), np.array(medians), np.array(counts)


def convert_plot_y(rate_deg_day, y_unit):
    if y_unit == "nHz":
        return deg_day_to_nhz(rate_deg_day)
    return rate_deg_day


def y_axis_label(y_unit):
    if y_unit == "nHz":
        return "Sidereal rotation rate (nHz)"
    return "Sidereal rotation rate (deg/day)"


def write_measurements_csv(measurements, output_csv):
    fieldnames = [
        "row_index",
        "BX",
        "BY",
        "EX",
        "EY",
        "mode",
        "start_obstime",
        "end_obstime",
        "time_source",
        "time_delta_days",
        "conversion_ok",
        "accepted",
        "reject_reason",
        "effective_b0_deg",
        "effective_b0_source",
        "start_radius_fraction",
        "end_radius_fraction",
        "latitude_deg",
        "delta_lat_deg",
        "cmd_start_deg",
        "cmd_end_deg",
        "abs_cmd_deg",
        "synodic_delta_lon_deg",
        "carrington_delta_lon_deg",
        "synodic_rate_deg_per_day",
        "sidereal_rate_deg_per_day",
        "sidereal_rate_nhz",
        "sidereal_period_days",
    ]

    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for measurement in measurements:
            writer.writerow(measurement)


def build_plot(measurements, bin_centers, bin_medians_deg_day, bin_counts, fit_result, args, summary):
    accepted = [item for item in measurements if item["accepted"]]
    rejected = [item for item in measurements if item["conversion_ok"] and not item["accepted"]]

    figure, axis = plt.subplots(figsize=(10.5, 6.8))

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

    if fit_result is not None:
        lat_grid = np.linspace(0.0, args.max_lat, 300)
        fit_deg_day = rotation_model_deg_day(lat_grid, fit_result["coefficients"])
        axis.plot(
            lat_grid,
            convert_plot_y(fit_deg_day, args.y_unit),
            color="#e45756",
            linewidth=2.3,
            label="Constrained fit to bin medians",
        )

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

    if args.y_unit == "nHz":
        secondary = axis.secondary_yaxis("right", functions=(nhz_to_period_days, period_days_to_nhz))
        secondary.set_ylabel("Sidereal period (days)")

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

    info_lines.extend(
        [
            f"Accepted points = {len(accepted)}",
            f"|CMD| <= {args.max_cmd:.0f} deg",
            f"|delta latitude| <= {args.max_delta_lat:.2f} deg/day",
            f"Sidereal rate window = {args.min_sidereal_rate:.1f} to {args.max_sidereal_rate:.1f} deg/day",
            f"Bin threshold = {args.min_bin_count} points",
        ]
    )

    if fit_result is not None:
        coeff_a, coeff_b, coeff_c = fit_result["coefficients"]
        info_lines.append(
            f"omega = {coeff_a:.3f} - {coeff_b:.3f} sin^2(lat) - {coeff_c:.3f} sin^4(lat) deg/day"
        )
        if not np.isnan(fit_result["r_squared"]):
            info_lines.append(f"Bin-median R^2 = {fit_result['r_squared']:.3f}")

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
    args = parse_args()
    rows = load_rows(args.points_csv)
    global_start, global_end = resolve_global_times(args)
    exact_measurements, exact_summary = compute_measurements_exact(rows, args, global_start, global_end)

    if exact_summary["missing_time_rows"] == len(rows):
        ephemeris_entries = collect_folder_ephemeris(args.images_dir)
        measurements, summary = compute_measurements_pooled(rows, args, ephemeris_entries)
    else:
        measurements, summary = exact_measurements, exact_summary

    accepted = [item for item in measurements if item["accepted"]]
    if len(accepted) < args.min_bin_count:
        raise ValueError(
            "Too few accepted points remained after the sunspot quality cuts. "
            "Loosen the filters or provide more tracked spots."
        )

    bin_centers, bin_medians_deg_day, bin_counts = summarize_bins(measurements, args)
    fit_result = fit_constrained_sunspot_curve(bin_centers, bin_medians_deg_day, bin_counts)

    build_plot(measurements, bin_centers, bin_medians_deg_day, bin_counts, fit_result, args, summary)
    write_measurements_csv(measurements, args.output_csv)

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


if __name__ == "__main__":
    main()
