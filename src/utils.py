"""
Shared utility functions for the CSE437 NYC Airbnb price prediction project.
Import from notebooks with:
    import sys; sys.path.append('../src')
    from utils import *
"""

import os
import pandas as pd
import numpy as np


DATA_RAW_PATH = "../data/raw/AB_NYC_2019.csv"
DATA_PROCESSED_PATH = "../data/processed/airbnb_clean.csv"

NYC_CENTER_LAT = 40.7580  # Times Square, used as reference point for distance feature
NYC_CENTER_LON = -73.9855


def load_raw_data(path: str = DATA_RAW_PATH) -> pd.DataFrame:
    """Load the raw Kaggle CSV, supporting relative paths from notebooks or root directory."""
    candidate_paths = [
        path,
        path + ".zip",
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", "AB_NYC_2019.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "raw", "AB_NYC_2019.csv.zip"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    return pd.read_csv(path)


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return count and percentage of missing values per column, sorted descending."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return report[report["missing_count"] > 0].sort_values("missing_pct", ascending=False)


def iqr_outlier_bounds(series: pd.Series, k: float = 1.5) -> tuple:
    """Return (lower_bound, upper_bound) for outlier detection via IQR method."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def flag_outliers_iqr(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.Series:
    """Return a boolean mask flagging rows where `column` is an IQR outlier."""
    lower, upper = iqr_outlier_bounds(df[column], k=k)
    return (df[column] < lower) | (df[column] > upper)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points (vectorized)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def distance_from_center(df: pd.DataFrame, lat_col="latitude", lon_col="longitude") -> pd.Series:
    """Distance in km from each listing to NYC_CENTER (Times Square)."""
    return haversine_distance(df[lat_col], df[lon_col], NYC_CENTER_LAT, NYC_CENTER_LON)


def regression_report(y_true, y_pred, label: str = "") -> dict:
    """Compute RMSE, MAE, R2 for a set of predictions. Prints and returns as dict."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}
