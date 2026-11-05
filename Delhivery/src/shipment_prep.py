"""Shared cleaning and feature engineering for shipment delay models."""
import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity", "label_print_batch", "scanner_build_no", "tape_roll_no",
    "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]
MEDIAN_COLUMNS = ["distance_km", "parcel_weight_kg", "declared_value_inr"]


def clean_shipments(raw: pd.DataFrame, medians=None):
    """Return deduplicated, normalized shipments and the three median fill values."""
    df = raw.drop_duplicates().copy()
    df["service_type"] = df["service_type"].astype("string").str.strip().str.title()
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    value = df["declared_value"].astype("string").str.replace("₹", "", regex=False).str.replace(",", "", regex=False).str.strip()
    df["declared_value_inr"] = pd.to_numeric(value, errors="coerce").astype(float)
    df = df.drop(columns=["declared_value"])
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], errors="coerce")
    df["parcel_weight_kg"] = pd.to_numeric(df["parcel_weight_kg"], errors="coerce")
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    for col in ["distance_km", "parcel_weight_kg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if medians is None:
        medians = {col: float(df[col].median()) for col in MEDIAN_COLUMNS}
    else:
        medians = {col: float(medians[col]) for col in MEDIAN_COLUMNS}
    for col in MEDIAN_COLUMNS:
        df[col] = df[col].fillna(medians[col])
    return df, medians


def build_features(clean: pd.DataFrame, hubs: pd.DataFrame) -> pd.DataFrame:
    """Attach hub attributes and create model inputs without using delivery outcomes."""
    df = clean.merge(hubs[["hub_id", "hub_city", "region", "hub_tier", "hub_daily_capacity"]],
                     on="hub_id", how="left", validate="many_to_one")
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)
    df["log_declared_value"] = np.log(df["declared_value_inr"].clip(lower=1))
    return df
