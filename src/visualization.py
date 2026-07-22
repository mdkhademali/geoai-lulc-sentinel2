"""
visualization.py
=================
Publication-quality static (Matplotlib) and interactive (Plotly / Folium)
visualization utilities for GeoAI LULC classification outputs.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 11,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
})


def plot_rgb(bands: Dict[str, np.ndarray], title="Sentinel-2 True Color Composite", save_path=None):
    def stretch(a):
        lo, hi = np.nanpercentile(a, 2), np.nanpercentile(a, 98)
        return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = np.dstack([stretch(bands["B04"]), stretch(bands["B03"]), stretch(bands["B02"])])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_index(index_arr: np.ndarray, name: str, cmap="RdYlGn", save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(index_arr, cmap=cmap, vmin=np.nanpercentile(index_arr, 2), vmax=np.nanpercentile(index_arr, 98))
    ax.set_title(f"{name} Spatial Distribution")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=name)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_class_map(class_map: np.ndarray, class_names: dict, class_colors: dict,
                    title="Land Use / Land Cover Classification", save_path=None):
    cmap = mcolors.ListedColormap([class_colors[i] for i in sorted(class_names)])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(class_map, cmap=cmap, vmin=-0.5, vmax=len(class_names) - 0.5)
    ax.set_title(title)
    ax.axis("off")
    handles = [Patch(color=class_colors[i], label=class_names[i]) for i in sorted(class_names)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, frameon=False, fontsize=9)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: list, normalize=True, save_path=None):
    cm_plot = cm.astype("float") / cm.sum(axis=1, keepdims=True) if normalize else cm
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_plot, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.xticks(rotation=40, ha="right")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_feature_importance(model, feature_names, top_n=15, save_path=None):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise ValueError("Model has no feature_importances_ attribute.")
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[order][::-1], importances[order][::-1], color="#2c7fb8")
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_learning_curve(train_sizes, train_scores, val_scores, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training score", color="#d95f02")
    ax.fill_between(train_sizes, train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15, color="#d95f02")
    ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Cross-val score", color="#1b9e77")
    ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15, color="#1b9e77")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve")
    ax.legend()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_roc_curves(fpr, tpr, roc_auc, class_names, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for i, name in enumerate(class_names):
        ax.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f"{name} (AUC={roc_auc[i]:.3f})")
    ax.plot(fpr["macro"], tpr["macro"], color="black", lw=2, linestyle="--",
            label=f"Macro-average (AUC={roc_auc['macro']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-class ROC Curves (One-vs-Rest)")
    ax.legend(fontsize=8, loc="lower right")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def make_folium_map(class_map: np.ndarray, bounds, class_names: dict, class_colors: dict):
    """Overlay the classification raster on an interactive Folium basemap."""
    import folium
    from PIL import Image
    import matplotlib.colors as mcolors
    import io, base64

    cmap = mcolors.ListedColormap([class_colors[i] for i in sorted(class_names)])
    norm = mcolors.Normalize(vmin=-0.5, vmax=len(class_names) - 0.5)
    rgba = (cmap(norm(class_map)) * 255).astype("uint8")
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    min_lon, min_lat, max_lon, max_lat = bounds
    center = [(min_lat + max_lat) / 2, (min_lon + max_lon) / 2]
    m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    folium.raster_layers.ImageOverlay(
        image=data_url, bounds=[[min_lat, min_lon], [max_lat, max_lon]], opacity=0.7,
        name="LULC Classification",
    ).add_to(m)
    legend_html = "".join(
        f'<i style="background:{class_colors[i]};width:10px;height:10px;display:inline-block;margin-right:4px"></i>{class_names[i]}<br>'
        for i in sorted(class_names)
    )
    m.get_root().html.add_child(folium.Element(
        f'<div style="position: fixed; bottom: 20px; left: 20px; z-index:9999; '
        f'background:white; padding:10px; border-radius:6px; font-size:12px;'
        f'box-shadow:0 0 6px rgba(0,0,0,0.3)">{legend_html}</div>'
    ))
    folium.LayerControl().add_to(m)
    return m
