#!/usr/bin/env python
# coding: utf-8

# In[1]:


# src/main.py
"""
Main script for urban typology classification with spatial validation.
Uses a Random Forest model and spatial cross-validation (DBSCAN + GroupKFold).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import clone
from sklearn.cluster import DBSCAN
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local
from scipy.spatial import cKDTree

# =============================================================================
# Configuration
# =============================================================================
# Paths (adjust as needed)
DATA_PATH = Path(r"C:\Users\Abril\Github_projects\urban_system_typology\df_urban_system_typology.gpkg")
OUTPUT_DIR = Path(r"C:\Users\Abril\Github_projects\urban_system_typology")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
DBSCAN_EPS = 500          # meters
DBSCAN_MIN_SAMPLES = 5
N_SPLITS_SPATIAL = 5

# =============================================================================
# Helper functions
# =============================================================================
def extract_coordinates(geometry):
    """Extract x, y coordinates from a geometry (Point or centroid of polygon)."""
    if geometry is None or geometry.is_empty:
        return None, None
    try:
        if geometry.geom_type == 'Point':
            return geometry.x, geometry.y
        else:
            centroid = geometry.centroid
            return centroid.x, centroid.y
    except:
        return None, None

def make_serializable(obj):
    """Convert numpy/pandas objects to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_list()
    else:
        return obj

# =============================================================================
# Main workflow
# =============================================================================
def main():
    print("="*70)
    print("URBAN TYPOLOGY CLASSIFICATION WITH SPATIAL VALIDATION")
    print("="*70)

    # 1. Load data
    print("\n1. Loading GeoPackage...")
    gdf = gpd.read_file(DATA_PATH)
    df = pd.DataFrame(gdf)
    # Replace empty strings with NaN in target column
    df['class'] = df['class'].replace(['', 'NA', 'null', 'NaN', 'None'], np.nan)
    # Extract coordinates
    if 'geometry' in df.columns:
        df['x'], df['y'] = zip(*df['geometry'].apply(extract_coordinates))
    muestras = df[df['class'].notnull()].copy()
    le = LabelEncoder()
    muestras['class_encoded'] = le.fit_transform(muestras['class'])
    print(f"Labeled samples: {len(muestras):,}")

    # 2. Feature selection (13 columns, all numeric except categorical codes)
    features = [
        'code_osm', 'level', 'built_area_block', 'dist_informal', 'area_block',
        'dist_block', 'dist_median_build', 'dist_river', 'dist_min_build',
        'dist_center', 'cluster_block', 'code_dist_via', 'dist_via'
    ]
    # Ensure all features exist
    missing = [f for f in features if f not in muestras.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    X = muestras[features]
    y = muestras['class_encoded']
    print(f"Features used: {len(features)}")
    print(f"Samples: {len(X)}")

    # 3. Gradual class weighting (as in original)
    class_counts = y.value_counts()
    total_samples = len(y)
    n_classes = len(class_counts)
    thresholds = [
        (total_samples * 0.005, 3.0),
        (total_samples * 0.01, 2.0),
        (total_samples * 0.02, 1.5),
        (float('inf'), 1.0)
    ]
    class_weights = {}
    for cls, count in class_counts.items():
        base_weight = total_samples / (n_classes * count)
        multiplier = 1.0
        for thresh, mult in thresholds:
            if count < thresh:
                multiplier = mult
                break
        class_weights[cls] = base_weight * multiplier
    class_weight_dict = class_weights

    # 4. Train/test split (with fallback if any class has <2 samples)
    from collections import Counter

    class_counts = Counter(y)
    min_class_size = min(class_counts.values())
    if min_class_size >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        print("Stratified split used.")
    else:
        print(f"Warning: The least populated class has only {min_class_size} sample(s). Stratification disabled.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=None
        )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 5. Hyperparameter optimization
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'class_weight': [class_weight_dict, 'balanced', 'balanced_subsample', None]
    }
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=param_grid,
        n_iter=100, cv=3, scoring='f1_weighted',
        n_jobs=-1, random_state=RANDOM_STATE
    )
    print("\n2. Hyperparameter optimization...")
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    print(f"Best params: {random_search.best_params_}")
    print(f"Best CV F1: {random_search.best_score_:.4f}")

    # 6. Test evaluation
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_with_age = f1_score(y_test, y_pred, average='weighted')
    # Without age
    y_test_str = le.inverse_transform(y_test)
    y_pred_str = le.inverse_transform(y_pred)
    y_test_sin = [s[:-4] if '_' in s else s for s in y_test_str]
    y_pred_sin = [s[:-4] if '_' in s else s for s in y_pred_str]
    f1_without_age = f1_score(y_test_sin, y_pred_sin, average='weighted')
    print(f"\nTest accuracy: {accuracy:.4f}")
    print(f"Test F1 (with age): {f1_with_age:.4f}")
    print(f"Test F1 (without age): {f1_without_age:.4f}")

    # 7. Spatial cross-validation with DBSCAN (using only training data)
    print("\n3. Spatial cross-validation (DBSCAN)...")
    train_indices = X_train.index
    train_gdf = gdf.loc[train_indices].copy()
    if 'x' not in train_gdf.columns or 'y' not in train_gdf.columns:
        train_gdf['x'], train_gdf['y'] = zip(*train_gdf['geometry'].apply(extract_coordinates))
    train_gdf = train_gdf.dropna(subset=['x', 'y'])
    valid_idx = train_gdf.index
    X_spatial = X_train.loc[valid_idx]
    y_spatial = y_train.loc[valid_idx]
    coords = train_gdf[['x', 'y']].values

    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    cluster_labels = db.fit_predict(coords)
    # Assign noise points to individual groups (unique negative IDs)
    groups = cluster_labels.copy()
    noise_idx = np.where(cluster_labels == -1)[0]
    for idx in noise_idx:
        groups[idx] = -1 - idx
    unique_groups = np.unique(groups)
    n_splits = min(N_SPLITS_SPATIAL, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)

    cv_scores_con = []
    cv_scores_sin = []
    for fold, (train_idx_fold, val_idx_fold) in enumerate(gkf.split(X_spatial, y_spatial, groups=groups), 1):
        X_fold_train = X_spatial.iloc[train_idx_fold]
        X_fold_val = X_spatial.iloc[val_idx_fold]
        y_fold_train = y_spatial.iloc[train_idx_fold]
        y_fold_val = y_spatial.iloc[val_idx_fold]
        model_fold = clone(best_model)
        model_fold.fit(X_fold_train, y_fold_train)
        y_pred_val = model_fold.predict(X_fold_val)
        f1_con = f1_score(y_fold_val, y_pred_val, average='weighted')
        cv_scores_con.append(f1_con)
        # Without age
        y_val_str = le.inverse_transform(y_fold_val)
        y_pred_val_str = le.inverse_transform(y_pred_val)
        y_val_sin = [s[:-4] if '_' in s else s for s in y_val_str]
        y_pred_val_sin = [s[:-4] if '_' in s else s for s in y_pred_val_str]
        f1_sin = f1_score(y_val_sin, y_pred_val_sin, average='weighted')
        cv_scores_sin.append(f1_sin)
        print(f"Fold {fold}: F1 (with age) = {f1_con:.4f}, F1 (without age) = {f1_sin:.4f}")

    mean_f1_con = np.mean(cv_scores_con)
    std_f1_con = np.std(cv_scores_con)
    mean_f1_sin = np.mean(cv_scores_sin)
    std_f1_sin = np.std(cv_scores_sin)
    print(f"\nSpatial CV (DBSCAN): F1 (with age) = {mean_f1_con:.4f} +/- {std_f1_con*2:.4f}")
    print(f"Spatial CV (DBSCAN): F1 (without age) = {mean_f1_sin:.4f} +/- {std_f1_sin*2:.4f}")

    # 8. Residual spatial autocorrelation (on test set)
    print("\n4. Residual spatial autocorrelation (Moran's I)...")
    test_indices = y_test.index
    test_gdf = gdf.loc[test_indices].copy()
    if 'x' not in test_gdf.columns or 'y' not in test_gdf.columns:
        test_gdf['x'], test_gdf['y'] = zip(*test_gdf['geometry'].apply(extract_coordinates))
    test_gdf = test_gdf.dropna(subset=['x', 'y'])
    test_gdf['residual'] = y_test.values - y_pred
    w = KNN.from_dataframe(test_gdf, k=5)
    w.transform = 'r'
    moran = Moran(test_gdf['residual'], w, permutations=999)
    print(f"Global Moran's I: {moran.I:.4f}, expected: {moran.EI:.4f}, p-value: {moran.p_sim:.4f}")
    if moran.p_sim < 0.05:
        print("Residuals show significant spatial autocorrelation.")
    else:
        print("No significant spatial autocorrelation in residuals.")

    # 9. Save model and metrics
    joblib.dump(best_model, OUTPUT_DIR / "modelo_optimizado.pkl")
    metrics = {
        'best_params': random_search.best_params_,
        'test_accuracy': accuracy,
        'test_f1_with_age': f1_with_age,
        'test_f1_without_age': f1_without_age,
        'spatial_cv_f1_without_age': mean_f1_sin,
        'spatial_cv_std': std_f1_sin,
        'spatial_cv_f1_with_age': mean_f1_con,
        'features_used': features,
        'execution_date': datetime.now().isoformat()
    }
    with open(OUTPUT_DIR / 'final_model_metrics.json', 'w') as f:
        json.dump(make_serializable(metrics), f, indent=4)
    print("\nModel and metrics saved.")

    # 10. Generate plots (simple versions)
    print("\n5. Generating plots...")
    # F1 comparison bar plot
    plt.figure(figsize=(10,6))
    categories = ['With Age', 'Without Age']
    values = [f1_with_age, f1_without_age]
    plt.bar(categories, values, color=['lightcoral', 'lightgreen'])
    plt.ylabel('F1-Score')
    plt.title('Test Set F1 Comparison')
    for i, v in enumerate(values):
        plt.text(i, v+0.01, f'{v:.4f}', ha='center')
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'f1_comparison.png', dpi=300)
    plt.close()
    print("Plots saved in results folder.")

    print("\n" + "="*70)
    print("All tasks completed successfully.")
    print("="*70)

if __name__ == "__main__":
    main()


# In[ ]:




