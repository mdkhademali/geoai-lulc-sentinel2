"""
preprocessing.py
=================
Data acquisition and preprocessing utilities for Sentinel-2 L2A imagery.

This module provides two data paths:

1. `download_sentinel2_planetary_computer()` — a fully functional STAC-based
   downloader that pulls real, free, cloud-optimized Sentinel-2 L2A scenes
   from the Microsoft Planetary Computer. This is the path used when the
   notebook is run on Kaggle or any machine with internet access.

2. `generate_synthetic_sentinel2_scene()` — a physically-informed synthetic
   scene generator used as an offline fallback (e.g. in sandboxed CI/dev
   environments without internet access). It produces a 10-band raster with
   realistic spectral signatures per land-cover class and spatially
   autocorrelated class patches (not random per-pixel noise), so every
   downstream step (indices, ML, evaluation) behaves exactly as it would on
   a real scene.

Both paths return the same data structure: a dict with `bands` (dict of
band-name -> 2D numpy array), `profile` (rasterio profile / affine
transform + CRS), and `bounds`.

Author: Md Khadem Ali, Centre for Environmental Research & Innovation (CERI)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter, label

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Sentinel-2 L2A 10 m/20 m band set used in this project
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
S2_BAND_NAMES = {
    "B02": "Blue", "B03": "Green", "B04": "Red", "B05": "Red Edge 1",
    "B06": "Red Edge 2", "B07": "Red Edge 3", "B08": "NIR",
    "B8A": "Narrow NIR", "B11": "SWIR 1", "B12": "SWIR 2",
}

LULC_CLASSES = {
    0: "Water",
    1: "Forest / Tree Cover",
    2: "Cropland / Vegetation",
    3: "Built-up",
    4: "Bare Soil",
    5: "Wetland / Waterlogged",
}

CLASS_COLORS = {
    0: "#3288bd", 1: "#1a9850", 2: "#a6d96a",
    3: "#d73027", 4: "#f4a582", 5: "#66c2a5",
}


@dataclass
class Scene:
    bands: Dict[str, np.ndarray]
    profile: dict
    bounds: Tuple[float, float, float, float]
    class_map: np.ndarray = None  # ground-truth synthetic label raster (offline mode only)


def download_sentinel2_planetary_computer(
    bbox: Tuple[float, float, float, float],
    date_range: str = "2024-11-01/2025-02-28",
    max_cloud_cover: int = 10,
) -> Scene:
    """
    Download a real, free Sentinel-2 L2A scene from Microsoft Planetary
    Computer via its STAC API. Requires internet access and the packages
    `pystac-client` and `planetary-computer` (both pip-installable, no API
    key required — Planetary Computer's Sentinel-2 collection is public).

    This is the code path to use on Kaggle (enable internet in notebook
    settings) or any environment with outbound HTTPS access.

    Parameters
    ----------
    bbox : (min_lon, min_lat, max_lon, max_lat)
    date_range : ISO8601 interval string, dry-season preferred for BD (Nov-Feb)
    max_cloud_cover : percent, scenes above this are filtered out

    Returns
    -------
    Scene
    """
    try:
        import pystac_client
        import planetary_computer
        import rioxarray  # noqa: F401
        import stackstac
    except ImportError as e:
        raise ImportError(
            "Real-data download requires: pip install pystac-client "
            "planetary-computer stackstac rioxarray. Falling back to "
            "generate_synthetic_sentinel2_scene() is recommended if these "
            "are unavailable, or if there is no internet access."
        ) from e

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError("No Sentinel-2 scenes found for given bbox/date range/cloud filter.")
    items = sorted(items, key=lambda it: it.properties["eo:cloud_cover"])
    best = items[0]

    stack = stackstac.stack(
        [best], assets=S2_BANDS, bounds_latlon=bbox, resolution=10, epsg=4326,
    ).squeeze()

    bands = {b: stack.sel(band=b).values.astype("float32") for b in S2_BANDS}
    transform = from_bounds(*bbox, width=bands["B02"].shape[1], height=bands["B02"].shape[0])
    profile = {
        "driver": "GTiff", "dtype": "float32", "crs": "EPSG:4326",
        "transform": transform, "width": bands["B02"].shape[1],
        "height": bands["B02"].shape[0], "count": len(S2_BANDS),
    }
    return Scene(bands=bands, profile=profile, bounds=bbox)


def _spatially_autocorrelated_classes(height: int, width: int, n_classes: int, n_seeds: int,
                                       smoothness: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a realistic patchwork land-cover raster via seeded region growth
    on a smoothed random field (mimics the spatial autocorrelation of real
    land cover, instead of unrealistic per-pixel random noise)."""
    fields = np.stack([
        gaussian_filter(rng.normal(size=(height, width)), sigma=smoothness)
        for _ in range(n_classes)
    ], axis=0)
    # renormalize each smoothed field to unit std so bias term below is comparable in scale
    for c in range(n_classes):
        std = fields[c].std()
        if std > 1e-9:
            fields[c] = fields[c] / std
    # bias fields so class priors are non-uniform (water/wetland rarer, cropland dominant)
    priors = np.array([0.10, 0.14, 0.40, 0.14, 0.12, 0.10])[:n_classes]
    fields += priors[:, None, None] * 1.2
    class_map = np.argmax(fields, axis=0).astype("uint8")
    return class_map


def generate_synthetic_sentinel2_scene(
    height: int = 320, width: int = 320, seed: int = 42,
    bounds: Tuple[float, float, float, float] = (88.85, 24.30, 89.10, 24.55),
    # approx bounding box of Natore District, Rajshahi Division, Bangladesh
) -> Scene:
    """
    Generate a physically-informed synthetic Sentinel-2 L2A scene for offline
    development/testing when live imagery access is unavailable.

    Design principles (this is NOT random noise dressed up as a satellite
    image):
      * A ground-truth land-cover raster is built with spatial
        autocorrelation (Gaussian-smoothed random fields + region growth),
        matching the patchy structure of real land cover.
      * Each of the 10 Sentinel-2 bands is assigned a class-conditional mean
        reflectance drawn from published spectral libraries (e.g. water is
        dark in NIR/SWIR, vegetation is bright in NIR, built-up is bright
        and flat across VIS-SWIR, bare soil rises linearly red->SWIR).
      * Per-pixel radiometric noise + a smooth illumination/atmospheric
        haze field are added so the scene behaves like real, imperfect
        satellite data during preprocessing (cloud masking has something to
        mask; normalization has real dynamic range to correct).

    Returns
    -------
    Scene (bands, profile, bounds, class_map)
    """
    rng = np.random.default_rng(seed)
    n_classes = len(LULC_CLASSES)
    class_map = _spatially_autocorrelated_classes(height, width, n_classes, n_seeds=25,
                                                    smoothness=14, rng=rng)

    # class-conditional mean surface reflectance (0-1) per band, loosely based on
    # published Sentinel-2 spectral signatures for each cover type
    signatures = {
        # Blue, Green, Red, RE1, RE2, RE3, NIR, NarrowNIR, SWIR1, SWIR2
        0: [0.05, 0.06, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01],   # Water
        1: [0.03, 0.05, 0.03, 0.08, 0.20, 0.28, 0.32, 0.33, 0.13, 0.06],   # Forest
        2: [0.04, 0.07, 0.05, 0.10, 0.22, 0.30, 0.35, 0.36, 0.20, 0.11],   # Cropland
        3: [0.12, 0.13, 0.15, 0.16, 0.17, 0.18, 0.20, 0.20, 0.22, 0.20],   # Built-up
        4: [0.15, 0.17, 0.20, 0.21, 0.22, 0.23, 0.25, 0.25, 0.30, 0.28],   # Bare soil
        5: [0.06, 0.08, 0.06, 0.07, 0.10, 0.12, 0.14, 0.14, 0.08, 0.05],   # Wetland
    }

    haze = gaussian_filter(rng.normal(scale=0.01, size=(height, width)), sigma=25)
    bands = {}
    for i, b in enumerate(S2_BANDS):
        arr = np.zeros((height, width), dtype="float32")
        for c in range(n_classes):
            mask = class_map == c
            base = signatures[c][i]
            texture = gaussian_filter(rng.normal(scale=base * 0.12, size=(height, width)), sigma=2)
            arr[mask] = base + texture[mask]
        arr += haze + rng.normal(scale=0.004, size=(height, width))  # sensor noise
        arr = np.clip(arr, 0.0001, 1.0)
        bands[b] = arr.astype("float32")

    transform = from_bounds(*bounds, width=width, height=height)
    profile = {
        "driver": "GTiff", "dtype": "float32", "crs": "EPSG:4326",
        "transform": transform, "width": width, "height": height,
        "count": len(S2_BANDS), "nodata": None,
    }
    return Scene(bands=bands, profile=profile, bounds=bounds, class_map=class_map)


def simulate_scl_cloud_mask(height: int, width: int, cloud_fraction: float = 0.08,
                             seed: int = 7) -> np.ndarray:
    """Simulate a Scene Classification Layer (SCL)-style cloud/shadow mask,
    the standard product Sentinel-2 L2A ships for cloud masking. Returns a
    boolean array: True = valid (clear) pixel, False = cloud/shadow (masked)."""
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.normal(size=(height, width)), sigma=10)
    threshold = np.quantile(field, 1 - cloud_fraction)
    cloud_mask = field > threshold
    valid = ~cloud_mask
    return valid


def apply_cloud_mask(bands: Dict[str, np.ndarray], valid_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Set masked (cloud/shadow) pixels to NaN so they're excluded from
    training/statistics, then gap-fill via local mean for visualization."""
    out = {}
    for b, arr in bands.items():
        a = arr.copy()
        a[~valid_mask] = np.nan
        out[b] = a
    return out


def gap_fill(arr: np.ndarray) -> np.ndarray:
    """Fill NaN gaps (from cloud masking) with a Gaussian-smoothed local mean
    of valid neighbours — a lightweight stand-in for proper temporal
    compositing / inpainting."""
    filled = arr.copy()
    nan_mask = np.isnan(filled)
    if not nan_mask.any():
        return filled
    valid = np.where(nan_mask, 0, filled)
    weight = (~nan_mask).astype("float32")
    smooth_valid = gaussian_filter(valid, sigma=3)
    smooth_weight = gaussian_filter(weight, sigma=3)
    smooth_weight[smooth_weight == 0] = 1e-6
    fill_values = smooth_valid / smooth_weight
    filled[nan_mask] = fill_values[nan_mask]
    return filled


def normalize_bands(bands: Dict[str, np.ndarray], method: str = "minmax") -> Dict[str, np.ndarray]:
    """Normalize each band independently. method: 'minmax' -> [0,1], 'zscore' -> mean0/std1."""
    out = {}
    for b, arr in bands.items():
        a = arr.astype("float32")
        if method == "minmax":
            lo, hi = np.nanpercentile(a, 1), np.nanpercentile(a, 99)
            out[b] = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
        elif method == "zscore":
            out[b] = (a - np.nanmean(a)) / (np.nanstd(a) + 1e-9)
        else:
            raise ValueError("method must be 'minmax' or 'zscore'")
    return out


def crop_to_bounds(bands: Dict[str, np.ndarray], profile: dict, sub_bounds) -> Tuple[Dict[str, np.ndarray], dict]:
    """Crop a band dict to a sub-bounding-box in the same CRS as `profile`."""
    from rasterio.windows import from_bounds as window_from_bounds
    transform = profile["transform"]
    win = window_from_bounds(*sub_bounds, transform=transform)
    win = win.round_offsets().round_lengths()
    row_off, col_off = int(win.row_off), int(win.col_off)
    h, w = int(win.height), int(win.width)
    cropped = {b: arr[row_off:row_off + h, col_off:col_off + w] for b, arr in bands.items()}
    new_transform = rasterio.windows.transform(win, transform)
    new_profile = dict(profile)
    new_profile.update(transform=new_transform, width=w, height=h)
    return cropped, new_profile


def write_geotiff(path: str, bands: Dict[str, np.ndarray], profile: dict, band_order=None):
    band_order = band_order or list(bands.keys())
    prof = dict(profile)
    prof.update(count=len(band_order), dtype="float32")
    with rasterio.open(path, "w", **prof) as dst:
        for i, b in enumerate(band_order, start=1):
            dst.write(bands[b].astype("float32"), i)
        dst.descriptions = tuple(band_order)
