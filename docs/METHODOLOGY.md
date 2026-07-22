# Methodology

## 1. Study Area
Natore District, Rajshahi Division, Bangladesh. Approximate bounding box (WGS84):
`(88.85, 24.30, 89.10, 24.55)`, lon_min, lat_min, lon_max, lat_max.

## 2. Data
Sentinel-2 Level-2A (surface reflectance, atmospherically corrected) imagery, 10 bands
(B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12), sourced via the Microsoft Planetary
Computer STAC API (free, public, no API key). An offline, physically-informed synthetic
scene generator is used as a reproducibility fallback, see `src/preprocessing.py` for
the full rationale and implementation. Both paths produce an identical `Scene` data
structure so every downstream step is agnostic to which was used.

## 3. Preprocessing
1. **Cloud/shadow masking** using a Scene-Classification-Layer-style boolean mask.
2. **Gap filling** of masked pixels via a locally-weighted Gaussian reconstruction.
3. **Band normalization** 1st/99th percentile clip, rescaled to [0, 1].

## 4. Feature Engineering
Ten spectral indices computed from the normalized bands (formulas and rationale in
`src/feature_engineering.py` and the main README). Final feature space: 10 raw bands +
10 indices = 20 features per pixel.

## 5. Sampling
The full scene (320×320 = 102,400 pixels) is reduced to a stratified per-class sample
(≤1,500 pixels/class) for model training, tuning, and SHAP analysis, standard practice
in pixel-based LULC classification to keep kernel-based methods (SVM) and search-based
tuning (RandomizedSearchCV) computationally tractable without sacrificing class
representativeness. The final model is subsequently applied to **every** pixel in the
full scene to produce the wall-to-wall prediction map.

## 6. Models
Random Forest, XGBoost, LightGBM, and an RBF-kernel SVM are trained on a stratified
70/30 train-test split (features standardized via `StandardScaler`). All four are
compared on 5-fold cross-validated training accuracy and held-out test accuracy/macro-F1.
The Random Forest is further tuned via `RandomizedSearchCV` (15 iterations, 3-fold CV)
over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.

## 7. Evaluation
Overall accuracy, Cohen's kappa, per-class precision/recall/F1 (`evaluation.py`),
a normalized confusion matrix, a full `classification_report`, and one-vs-rest
multi-class ROC curves with per-class and macro-average AUC.

## 8. Explainability
SHAP `TreeExplainer` summary plots quantify each feature's marginal contribution to
individual-class predictions, alongside native tree-ensemble feature importances,
a learning curve (bias/variance diagnosis), and a validation curve over
`n_estimators`.

## 9. Prediction & Uncertainty
The tuned model is applied pixel-wise to the entire scene. Prediction confidence
(max class probability) is retained as a per-pixel uncertainty surface, exported
alongside the class map.

## 10. Limitations
See the Discussion section of the notebook and the main README's "Limitations"
subsection for a full treatment of pixel-vs-patch classification, single-date
vs. multi-temporal imagery, and resolution constraints.
