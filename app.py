import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
from datetime import timedelta

st.set_page_config(page_title="Fitness App RCA", layout="wide")

# -----------------------------------------------------------
# 1️⃣ LOAD DATA (NO UPLOAD REQUIRED)
# -----------------------------------------------------------

@st.cache_data
def load_data():
    zip_path = "Dataset_problem.zip"
    if not os.path.exists(zip_path):
        st.error("❌ Dataset_problem.zip not found in the repository.")
        st.stop()

    valid_dfs = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        for file in z.namelist():
            if file.startswith("__MACOSX") or file.endswith(".DS_Store"):
                continue  # skip mac ghost files

            if file.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(z.open(file))
                    table_name = file.split("/")[-1].replace(".csv", "")
                    valid_dfs[table_name] = df
                except Exception as e:
                    st.warning(f"⚠️ Failed to load {file}: {e}")

    required = ["users", "app_events", "app_sessions", "activity_data", "subscriptions"]
    for r in required:
        if r not in valid_dfs:
            st.error(f"❌ Missing required file: {r}.csv")
            st.stop()

    return valid_dfs


dfs = load_data()
users = dfs["users"]
app_events = dfs["app_events"]
app_sessions = dfs["app_sessions"]
activity_data = dfs["activity_data"]
subscriptions = dfs["subscriptions"]

# -----------------------------------------------------------
# 2️⃣ DATA PREPROCESSING + FEATURE ENGINEERING
# -----------------------------------------------------------

@st.cache_data
def preprocess_and_compute(users, app_events, app_sessions, activity_data, subscriptions):

    # Ensure datetime types
    users["created_at"] = pd.to_datetime(users["created_at"], errors="coerce")
    app_events["event_time"] = pd.to_datetime(app_events["event_time"], errors="coerce")
    app_sessions["session_start"] = pd.to_datetime(app_sessions["session_start"], errors="coerce")
    subscriptions["subscription_start_date"] = pd.to_datetime(
        subscriptions["subscription_start_date"], errors="coerce"
    )

    # ---- 90 DAY WINDOW ----
    max_date = users["created_at"].max()
    min_date = max_date - pd.Timedelta(days=90)

    recent = users[(users["created_at"] >= min_date) & (users["created_at"] <= max_date)].copy()

    # -----------------------------------------------------------
    # FIRST OPEN
    # -----------------------------------------------------------
    first_open = (
        app_events[app_events["event_name"] == "app_open"]
        .groupby("user_id")["event_time"]
        .min()
        .reset_index()
        .rename(columns={"event_time": "first_open_date"})
    )

    # -----------------------------------------------------------
    # FEATURE DEPTH
    # -----------------------------------------------------------
    feature_events = ["view_sleep", "view_recovery", "view_strain", "view_coaching"]

    feature_usage = (
        app_events[app_events["event_name"].isin(feature_events)]
        .groupby("user_id")["event_name"]
        .nunique()
        .reset_index()
        .rename(columns={"event_name": "feature_count"})
    )

    # -----------------------------------------------------------
    # WORKOUT FREQUENCY
    # -----------------------------------------------------------
    workout_freq = (
        activity_data.groupby("user_id")["workout_id"]
        .count()
        .reset_index()
        .rename(columns={"workout_id": "workouts"})
    )

    # -----------------------------------------------------------
    # RETENTION (D7, D14, D30)
    # -----------------------------------------------------------

    app_sessions = app_sessions[~app_sessions["session_start"].isna()].copy()

    def compute_ret(row):
        uid = row["user_id"]
        signup = row["created_at"]

        sessions = app_sessions[app_sessions["user_id"] == uid]["session_start"]

        d7 = any((sessions >= signup + timedelta(days=1)) &
                 (sessions <= signup + timedelta(days=7)))

        d14 = any((sessions >= signup + timedelta(days=1)) &
                  (sessions <= signup + timedelta(days=14)))

        d30 = any((sessions >= signup + timedelta(days=1)) &
                  (sessions <= signup + timedelta(days=30)))

        return pd.Series([d7, d14, d30], index=["d7", "d14", "d30"])

    recent[["d7", "d14", "d30"]] = recent.apply(compute_ret, axis=1)

    # Clean retention columns
    for col in ["d7", "d14", "d30"]:
        recent[col] = recent[col].astype(int)

    # -----------------------------------------------------------
    # MERGE ALL
    # -----------------------------------------------------------
    base = recent.merge(first_open, on="user_id", how="left")
    base = base.merge(feature_usage, on="user_id", how="left")
    base = base.merge(workout_freq, on="user_id", how="left")
    base = base.merge(subscriptions, on="user_id", how="left")

    base["feature_count"] = base["feature_count"].fillna(0)
    base["workouts"] = base["workouts"].fillna(0)

    # -----------------------------------------------------------
    # ACTIVATION SPEED LABEL
    # -----------------------------------------------------------
    def activation_label(row):
        fo = row["first_open_date"]
        sd = row["created_at"]

        if pd.isna(fo):
            return "Never Opened"
        days = (fo.date() - sd.date()).days

        if days == 0:
            return "Day 0"
        if 1 <= days <= 7:
            return "1–7 Days"
        if 8 <= days <= 30:
            return "8–30 Days"
        return "30+ Days"

    base["activation_speed"] = base.apply(activation_label, axis=1)

    # -----------------------------------------------------------
    # FEATURE DEPTH LABEL
    # -----------------------------------------------------------
    def depth_label(x):
        if x >= 3:
            return "3+ features"
        if x == 2:
            return "2 features"
        if x == 1:
            return "1 feature"
        return "0 features"

    base["feature_depth"] = base["feature_count"].apply(depth_label)

    # -----------------------------------------------------------
    # WORKOUT BUCKET
    # -----------------------------------------------------------
    def workout_bucket(x):
        if x >= 20:
            return "20+ workouts"
        if 6 <= x <= 19:
            return "6–19 workouts"
        if 1 <= x <= 5:
            return "1–5 workouts"
        return "0 workouts"

    base["workout_bucket"] = base["workouts"].apply(workout_bucket)

    return base


base = preprocess_and_compute(users, app_events, app_sessions, activity_data, subscriptions)

# -----------------------------------------------------------
# 3️⃣ SEGMENTATION FUNCTION
# -----------------------------------------------------------

def compute_segment_funnel(df, segment_col):
    seg = df.groupby(segment_col).agg(
        signed_up=("user_id", "count"),
        opened_app=("first_open_date", lambda x: x.notna().sum()),
        engaged_core=("feature_count", lambda x: (x > 0).sum()),
        d7=("d7", "sum"),
        d14=("d14", "sum"),
        d30=("d30", "sum"),
    ).reset_index()

    seg["pct_signup_to_open"] = round(100 * seg["opened_app"] / seg["signed_up"], 2)
    seg["pct_open_to_engage"] = round(
        100 * seg["engaged_core"] / seg["opened_app"].replace({0: np.nan}), 2
    )
    seg["pct_signup_to_engage"] = round(100 * seg["engaged_core"] / seg["signed_up"], 2)
    seg["pct_signup_to_d7"] = round(100 * seg["d7"] / seg["signed_up"], 2)
    seg["pct_signup_to_d14"] = round(100 * seg["d14"] / seg["signed_up"], 2)
    seg["pct_signup_to_d30"] = round(100 * seg["d30"] / seg["signed_up"], 2)

    return seg


# -----------------------------------------------------------
# 4️⃣ STREAMLIT UI
# -----------------------------------------------------------

st.title("📊 Fitness App — Root Cause Analysis Dashboard")

st.markdown("""
This dashboard summarizes:

### ✅ Activation  
### ✅ Feature Discovery  
### ✅ Retention (D7 / D14 / D30)  
### ✅ Segmentation (Age, Gender, Plan, Activation, Feature Depth, Workouts)  
---
""")

# -----------------------------------------------------------
# OVERALL SUMMARY
# -----------------------------------------------------------

st.header("🔍 Overall Funnel (Last 90 Days)")

overall = compute_segment_funnel(base, segment_col="activation_speed")
st.dataframe(overall)

# -----------------------------------------------------------
# SEGMENTED FUNNELS
# -----------------------------------------------------------

st.header("📂 Segmented Funnels")

segment_options = {
    "Activation Speed": "activation_speed",
    "Age Group": "age_group",
    "Gender": "gender",
    "Plan Type": "plan_type",
    "Feature Depth": "feature_depth",
    "Workout Bucket": "workout_bucket",
}

choice = st.selectbox("Select a segment:", list(segment_options.keys()))

seg_df = compute_segment_funnel(base, segment_options[choice])

st.subheader(f"Funnel by {choice}")
st.dataframe(seg_df)

st.markdown("---")
st.markdown("Built by Rushiraj — RCA + Cohort Analysis + Streamlit App")
