# src/utils.py
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import json

def extract_coordinates(geometry):
    """Extract x, y coordinates from a geometry (Point or centroid)."""
    if geometry is None or geometry.is_empty:
        return None, None
    try:
        if geometry.geom_type == 'Point':
            return geometry.x, geometry.y
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
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    else:
        return obj