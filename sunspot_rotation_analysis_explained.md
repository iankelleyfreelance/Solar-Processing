# Sunspot Rotation Analysis Explained

This repository now supports two related analyses:

1. `exact` mode
   Uses a real observation time for each start and end point.
2. `pooled` mode
   Uses the dated images in `Images/` to estimate one effective solar tilt `B0` for the whole point set when the point file itself has no dates.

Your current `points.csv` has no per-row dates, so the script uses `pooled` mode by default.

## What The Folder Dates Tell Us

The dated rotated TIFF files in `Images/` run from:

- `2026-01-26T17:21:57`
- through `2026-04-16T17:35:23`

For those dates, the solar tilt angle `B0` only varies from about:

- `-7.21 deg`
- to `-5.50 deg`

That is a narrow enough range that a pooled approximation is possible, even though it is not as exact as using a true date for every tracked point.

## Coordinate Geometry

All input images are assumed to be:

- centered on the solar disk,
- scaled to `900 px` radius,
- rotated so solar north is up.

That lets us convert image pixels into normalized solar-disk coordinates:

```text
x = (X - center_x) / radius_px
y = (center_y - Y) / radius_px
```

The sign flip in `y` is important because image rows increase downward while heliographic latitude increases upward.

For a chosen solar tilt `B0`, each point is mapped onto the visible solar sphere with:

```text
rho^2 = x^2 + y^2
cos(c) = sqrt(1 - rho^2)
phi = asin(y cos(B0) + cos(c) sin(B0))
lambda = atan2(x, cos(c) cos(B0) - y sin(B0))
```

where:

- `phi` is heliographic latitude,
- `lambda` is Stonyhurst longitude measured from the central meridian.

## Rotation Rate

Each point pair is assumed to span one day.

The measured longitude change is:

```text
delta_lambda_syn = wrapped(lambda_end - lambda_start)
```

The sidereal rate is then approximated by adding the Earth's mean orbital rate:

```text
omega_sid = delta_lambda_syn + 0.9856 deg/day
```

This is a good approximation for a pooled one-day sunspot dataset because the orbital correction changes very little across the folder date range.

## Why `B0` Matters

Without dates, the biggest missing ephemeris term is `B0`, the tilt of the Sun's north pole toward the observer.

If `B0` is wrong, apparent latitude changes are introduced by projection and high-latitude points can look artificially fast.

The pooled analysis handles this by:

1. reading the real folder dates,
2. computing the corresponding candidate `B0` values,
3. choosing the candidate `B0` that minimizes the median one-day `|delta latitude|` for plausible tracks.

For the current dataset, the selected effective tilt is:

- `B0 = -7.21348 deg`

This came from the folder image:

- `2026-02-28T18_38_25_rotated.tif`

## Quality Cuts

Only geometrically and physically reasonable tracks are kept for the final trend fit.

The current default cuts are:

- on disk at both times
- `|latitude| <= 35 deg`
- `|CMD| <= 45 deg`
- `|delta latitude| <= 0.50 deg/day`
- `13.2 <= sidereal rate <= 15.4 deg/day`

These cuts are chosen to keep:

- points away from the limb,
- spots whose inferred latitude is nearly constant over one day,
- rates in the expected sunspot range,
- enough points to still build latitude bins.

## Binning And Fitting

The accepted points are binned by absolute latitude using `5 deg` bins.

For each populated bin, the script stores:

- bin center,
- median sidereal rate,
- number of accepted points.

The fitted curve is:

```text
omega(phi) = A - B sin^2(phi) - C sin^4(phi)
```

with the constraints:

- `B >= 0`
- `C >= 0`

That forces a physically non-rigid, non-increasing sunspot-style profile.

## What The Reported `R^2` Means

The reported `R^2` is computed on the latitude-bin medians, not on the raw points.

That matters because:

- raw sunspot tracks are noisy,
- the bins are the actual rotation profile estimate,
- the fit is intended to describe the latitude trend, not every single track.

For the current pooled dataset, the script produced:

```text
omega(phi) = 14.44992 - 2.16130 sin^2(phi) deg/day
R^2 = 0.90539
```

This is a defensible, textbook-like sunspot rotation curve for the pooled no-date data, but it is still an approximation.

## Reference Curve

The plot also shows a dashed reference sunspot curve:

```text
omega_ref(phi) = 14.38 - 2.96 sin^2(phi) deg/day
```

This is included only as a visual reference. Your fitted curve is driven by your accepted points and bins.

## How To Read The Output CSV

`solar_rotation_measurements.csv` includes, for each track:

- inferred latitude,
- inferred one-day latitude drift,
- central-meridian distance,
- sidereal rate,
- accepted/rejected flag,
- rejection reason,
- effective `B0` used.

This makes it possible to audit exactly why any individual track was kept or removed.

## Commands

Run the pooled no-date analysis:

```text
python solar_rotation_analysis.py
```

Run the exact timed analysis:

```text
python solar_rotation_analysis.py --start-image Images\2026-04-16T17_35_23_rotated.tif --end-image Images\2026-04-17T17_35_23_Bh.jpg
```

If you later add per-row date columns such as `start_image,end_image` or `start_obstime,end_obstime`, the script will automatically use the exact mode for those rows instead of the pooled approximation.
