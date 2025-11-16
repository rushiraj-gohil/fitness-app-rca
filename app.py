import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from datetime import timedelta
import os

st.set_page_config(page_title="Fitness RCA Dashboard", layout="wide")

ZIP_PATH = "Dataset_problem.zip"   # <--- ZIP ALREADY IN REPO


# ============================================================
# 1) SAFE CSV LOADER (Fixes __MACOSX junk)
# ============================================================

@st.cache_data
def load_dataset(zip_path):
    users = app_events = app_sessions = activity_data = subscriptions = None

    with zipfile.ZipFile(zip_path, 'r') as z:
        for filename in z.namelist():
            if "__MACOSX" in filename or "._" in filename:
                continue
            if not filename.endswith(".csv"):
                continue

            with z.open(filename) as f:
                try:
                    df = pd.read_csv(f)
                except Exception:
                    continue

            fname = filename.lower()

            if "users" in fname:
                users = df
            elif "app_events" in fname:
                app_events = df
            elif "app_sessions" in fname:
                app_sessions = df
            elif "activity_data" in fname:
                activity_data = df
            elif "subscriptions" in fname:
                subscriptions = df

    return users, app_events, app_sessions, activity_data, subscriptions


# ============================================================
# 2) Activation bucket
# ============================================================

def activation_label(row):
    sd = pd.to_datetime(row["signup_date"], errors="coerce")
    fo = pd.to_datetime(row["first_open_date"], errors="coerce")

    if pd.isna(sd):
        return "Unknown"
    if pd.isna(fo):
        return "Never Opened"

    diff = (fo - sd).days

    if diff == 0:
        return "Day 0"
    elif 1 <= diff <= 7:
        return "1–7 Days"
    elif 8 <= diff <= 30:
        return "8–30 Days"
    elif diff > 30:
        return "30+ Days"
    return "Unknown"


# ============================================================
# 3) Process dataset and compute metrics
# ============================================================

@st.cache_data
def preprocess_and_compute(users, app_events, app_sessions, activity_data, subscriptions):

    # Convert dates
    users["signup_date"] = pd.to_datetime(users["created_at"], errors="coerce")
    app_events["event_time"] = pd.to_datetime(app_events["event_time"], errors="coerce")
    app_sessions["session_start"] = pd.to_datetime(app_sessions["session_start"], errors="coerce")
    subscriptions["subscription_start_date"] = pd.to_datetime(subscriptions["subscription_start_date"], errors="coerce")

    # ---- First App Open ----
    first_open = (
        app_events[app_events.event_name == "app_open"]
        .groupby("user_id")["event_time"]
        .min()
        .reset_index()
        .rename(columns={"event_time": "first_open_date"})
    )

    # ---- Feature Depth ----
    feature_names = ["view_sleep", "view_recovery", "view_strain", "view_coaching"]
    feature_usage = (
        app_events[app_events["event_name"].isin(feature_names)]
        .groupby("user_id")["event_name"]
        .nunique()
        .reset_index()
        .rename(columns={"event_name": "feature_count"})
    )

    # ---- Workout Frequency ----
    workout_freq = (
        activity_data.groupby("user_id")
        .size()
        .reset_index(name="workouts")
    )

    # ---- Retention Windows ----
    merged_sessions = app_sessions.merge(
        users[["user_id", "signup_date"]], on="user_id", how="left"
    )
    merged_sessions["diff_days"] = (merged_sessions["session_start"] - merged_sessions["signup_date"]).dt.days

    retention = merged_sessions.groupby("user_id").agg(
        d7=("diff_days", lambda x: any((x >= 1) & (x <= 7))),
        d14=("diff_days", lambda x: any((x >= 1) & (x <= 14))),
        d30=("diff_days", lambda x: any((x >= 1) & (x <= 30))),
    ).reset_index()

    # ---- Merge base ----
    base = (
        users.merge(first_open, on="user_id", how="left")
        .merge(feature_usage, on="user_id", how="left")
        .merge(workout_freq, on="user_id", how="left")
        .merge(retention, on="user_id", how="left")
        .merge(subscriptions[["user_id", "plan_type"]], on="user_id", how="left")
    )

    base["feature_count"] = base["feature_count"].fillna(0)
    base["workouts"] = base["workouts"].fillna(0)

    # ---- Activation bucket ----
    base["activation_speed"] = base.apply(activation_label, axis=1)

    # ---- Feature bucket ----
    def feat_bucket(x):
        return "3+ features" if x >= 3 else "2 features" if x == 2 else "1 feature" if x == 1 else "0 features"

    base["feature_depth"] = base["feature_count"].apply(feat_bucket)

    # ---- Workout bucket ----
    def workout_bucket(n):
        if n >= 20: return "20+ workouts"
        if n >= 6: return "6–19 workouts"
        if n >= 1: return "1–5 workouts"
        return "0 workouts"

    base["workout_bucket"] = base["workouts"].apply(workout_bucket)

    # Ensure retention columns are numeric
    for col in ["d7", "d14", "d30"]:
        base[col] = base[col].astype(int)

    return base


# ============================================================
# Load dataset directly from repo ZIP
# ============================================================

if not os.path.exists(ZIP_PATH):
    st.error("❌ Dataset_problem.zip not found in repository!")
    st.stop()

users, app_events, app_sessions, activity_data, subscriptions = load_dataset(ZIP_PATH)
base = preprocess_and_compute(users, app_events, app_sessions, activity_data, subscriptions)

st.title("📊 Fitness App – RCA Dashboard")
st.success("Dataset loaded successfully!")


# ============================================================
# SEGMENTATION VIEW
# ============================================================

st.header("📌 Segmentation Explorer")

segment_col = st.selectbox(
    "Select Segment",
    ["age_group", "gender", "plan_type", "activation_speed", "feature_depth", "workout_bucket"]
)

seg = base.groupby(segment_col).agg(
    signed_up=("user_id", "count"),
    opened_app=("first_open_date", lambda x: x.notna().sum()),
    engaged_core=("feature_count", lambda x: (x > 0).sum()),
    d7=("d7", "sum"),
    d14=("d14", "sum"),
    d30=("d30", "sum")
).reset_index()

# ---- Ensure numeric dtype ----
numeric_cols = ["signed_up", "opened_app", "engaged_core", "d7", "d14", "d30"]
seg[numeric_cols] = seg[numeric_cols].astype(float)

# ---- Percentages ----
seg["pct_signup_to_open"] = round(100 * seg["opened_app"] / seg["signed_up"], 2)
seg["pct_open_to_engage"] = round(100 * seg["engaged_core"] / seg["opened_app"].replace(0, np.nan), 2)
seg["pct_signup_to_engage"] = round(100 * seg["engaged_core"] / seg["signed_up"], 2)
seg["pct_signup_to_d7"] = round(100 * seg["d7"] / seg["signed_up"], 2)
seg["pct_signup_to_d14"] = round(100 * seg["d14"] / seg["signed_up"], 2)
seg["pct_signup_to_d30"] = round(100 * seg["d30"] / seg["signed_up"], 2)

st.dataframe(seg)

st.success("Segmentation computed successfully!")
