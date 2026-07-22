"""
feature_engineering.py
=======================
Spectral index computation for Sentinel-2 L2A imagery and tabular
feature-matrix assembly for machine learning.

All indices follow standard published formulas. Band references use
Sentinel-2 naming (B02=Blue, B03=Green, B04=Red, B08=NIR, B11=SWIR1,
B12=SWIR2, B8A=Narrow NIR).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

EPS = 1e-9


def ndvi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """Normalized Difference Vegetation Index — vegetation vigor/greenness."""
    nir, red = bands["B08"], bands["B04"]
    return (nir - red) / (nir + red + EPS)


def ndwi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """Normalized Difference Water Index (McFeeters) — open water detection."""
    green, nir = bands["B03"], bands["B08"]
    return (green - nir) / (green + nir + EPS)


def mndwi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """Modified NDWI (Xu 2006) — water detection robust to built-up noise."""
    green, swir1 = bands["B03"], bands["B11"]
    return (green - swir1) / (green + swir1 + EPS)


def ndbi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """Normalized Difference Built-up Index — impervious/urban surfaces."""
    swir1, nir = bands["B11"], bands["B08"]
    return (swir1 - nir) / (swir1 + nir + EPS)


def savi(bands: Dict[str, np.ndarray], L: float = 0.5) -> np.ndarray:
    """Soil Adjusted Vegetation Index — NDVI corrected for soil background."""
    nir, red = bands["B08"], bands["B04"]
    return ((nir - red) / (nir + red + L + EPS)) * (1 + L)


def evi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """Enhanced Vegetation Index — improved sensitivity in dense canopy, atmosphere-corrected."""
    nir, red, blue = bands["B08"], bands["B04"], bands["B02"]
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + EPS)


def bsi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """Bare Soil Index — highlights exposed/bare soil surfaces."""
    swir1, red, nir, blue = bands["B11"], bands["B04"], bands["B08"], bands["B02"]
    return ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + EPS)


def gndvi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """Green NDVI — chlorophyll sensitivity via green band instead of red."""
    nir, green = bands["B08"], bands["B03"]
    return (nir - green) / (nir + green + EPS)


def nbr(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """Normalized Burn Ratio — useful for degraded vegetation / burn scars."""
    nir, swir2 = bands["B08"], bands["B12"]
    return (nir - swir2) / (nir + swir2 + EPS)


def rendvi(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """Red-Edge NDVI — leverages Sentinel-2's red-edge bands for crop vigor."""
    re2, re1 = bands["B06"], bands["B05"]
    return (re2 - re1) / (re2 + re1 + EPS)


INDEX_FUNCS = {
    "NDVI": ndvi, "NDWI": ndwi, "MNDWI": mndwi, "NDBI": ndbi,
    "SAVI": savi, "EVI": evi, "BSI": bsi, "GNDVI": gndvi,
    "NBR": nbr, "RENDVI": rendvi,
}


def compute_all_indices(bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute the full spectral index stack. Returns dict name -> 2D array."""
    return {name: np.nan_to_num(fn(bands), nan=0.0, posinf=0.0, neginf=0.0)
            for name, fn in INDEX_FUNCS.items()}


def build_feature_stack(bands: Dict[str, np.ndarray], indices: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack raw bands + indices into a single (H, W, n_features) array."""
    layers = list(bands.values()) + list(indices.values())
    return np.stack(layers, axis=-1)


def feature_names(bands: Dict[str, np.ndarray], indices: Dict[str, np.ndarray]) -> list:
    return list(bands.keys()) + list(indices.keys())


def raster_to_dataframe(feature_stack: np.ndarray, names: list, class_map: np.ndarray = None,
                         sample_frac: float = None, seed: int = 42) -> pd.DataFrame:
    """Flatten an (H, W, F) feature stack (+ optional label raster) into a
    tabular DataFrame, one row per pixel, for classical ML. Optionally
    subsample for tractability on large scenes."""
    h, w, f = feature_stack.shape
    flat = feature_stack.reshape(-1, f)
    df = pd.DataFrame(flat, columns=names)
    df["row"] = np.repeat(np.arange(h), w)
    df["col"] = np.tile(np.arange(w), h)
    if class_map is not None:
        df["label"] = class_map.flatten()
    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
    return df
