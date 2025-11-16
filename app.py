import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from datetime import datetime, timedelta
import plotly.express as px


# ============================================================
# 1) LOAD ZIP + CSVs SAFELY
# ============================================================

@st.cache_data
def load_zip(zip_file):
    dfs = {}

    with zipfile.ZipFile(zip_file, 'r') as z:
        for file in z.namelist():
            if file.endswith(".csv") and "__MACOSX" not in file:
                try:
                    df = pd.read_csv(z.open(file))
                    simple_name = file.split("/")[-1].replace(".csv", "")
                    dfs[simple_name] = df
                except:
                    pass  # ignore hidden MACOSX files

    return dfs


# ============================================================
# 2) SAFE HELPERS
# ============================================================

def safe_date(col):
    """Convert to datetime; if fails, return NaT."""
    try:
        return pd.to_datetime(col, errors="coerce")
    except:
        return pd.NaT


def safe_group(df, key, count_col_name):
    """Return df grouped by key if column exists, else empty df."""
    if df is not None and key in df.columns:
        return (
            df.groupby(key)
            .size()
            .reset_index(name=count_col_name)
        )
    return pd.DataFrame({key: [], count_col_name: []})


def safe_merge(left, right, key):
    if len(right) > 0:
        return left.merge(right, on=key, how="left")
    return left


# ============================================================
# 3) COMPUTE BASE TABLE
# ============================================================

@st.cache_data
def compute_base(dfs):

    # -------------------- Load Required Tables --------------------
    users = dfs.get("users")
    app_events = dfs.get("app_events")
    app_sessions = dfs.get("app_sessions")
    activity_data = dfs.get("activity_data")
    subscriptions = dfs.get("subscriptions")

    # Convert dates safely
    if "created_at" in users.columns:
        users["created_at"] = safe_date(users["created_at"])

    if app_events is not None and "event_time" in app_events.columns:
        app_events["event_time"] = safe_date(app_events["event_time"])

    if app_sessions is not None and "session_start" in app_sessions.columns:
        app_sessions["session_start"] = safe_date(app_sessions["session_start"])

    if subscriptions is not None and "subscription_start_date" in subscriptions.columns:
        subscriptions["subscription_start_date"] = safe_date(subscriptions["subscription_start_date"])

    # -------------------------------------------------------------
    # RECENT USERS (LAST 90 DAYS)
    # -------------------------------------------------------------
    max_signup = users["created_at"].max()
    min_signup = max_signup - timedelta(days=90)

    recent = users[(users["created_at"] >= min_signup) & (users["created_at"] <= max_signup)].copy()

    # -------------------------------------------------------------
    # FIRST APP OPEN
    # -------------------------------------------------------------
    if app_events is not None and "event_name" in app_events.columns:
        first_open = (
            app_events[app_events["event_name"] == "app_open"]
            .groupby("user_id")["event_time"]
            .min()
            .reset_index(name="first_open_date")
        )
    else:
        first_open = pd.DataFrame({"user_id": [], "first_open_date": []})

    # -------------------------------------------------------------
    # FEATURE DEPTH (count of core views)
    # -------------------------------------------------------------
    core_features = ["view_sleep", "view_recovery", "view_strain", "view_coaching"]

    if app_events is not None and "event_name" in app_events.columns:
        feature_usage = (
            app_events[app_events["event_name"].isin(core_features)]
            .groupby("user_id")["event_name"]
            .nunique()
            .reset_index(name="feature_count")
        )
    else:
        feature_usage = pd.DataFrame({"user_id": [], "feature_count": []})

    # -------------------------------------------------------------
    # WORKOUT FREQUENCY — BUT ONLY IF user_id EXISTS
    # -------------------------------------------------------------
    if activity_data is not None and "user_id" in activity_data.columns:
        workout_freq = (
            activity_data.groupby("user_id")
            .size()
            .reset_index(name="workouts")
        )
    else:
        workout_freq = pd.DataFrame({"user_id": [], "workouts": []})

    # -------------------------------------------------------------
    # RETENTION (D7 / D14 / D30)
    # -------------------------------------------------------------
    if app_sessions is not None and "session_start" in app_sessions.columns:

        df_ret = recent[["user_id", "created_at"]].copy()
        df_ret = df_ret.merge(app_sessions[["user_id", "session_start"]], on="user_id", how="left")

        def retention_flag(df, days):
            return (
                (df["session_start"] >= df["created_at"] + timedelta(days=1)) &
                (df["session_start"] <= df["created_at"] + timedelta(days=days))
            )

        df_ret["d7"] = retention_flag(df_ret, 7)
        df_ret["d14"] = retention_flag(df_ret, 14)
        df_ret["d30"] = retention_flag(df_ret, 30)

        retention = df_ret.groupby("user_id").agg(
            d7=("d7", "max"),
            d14=("d14", "max"),
            d30=("d30", "max"),
        ).reset_index()

    else:
        retention = pd.DataFrame({
            "user_id": recent["user_id"],
            "d7": 0,
            "d14": 0,
            "d30": 0
        })

    # -------------------------------------------------------------
    # MERGE EVERYTHING
    # -------------------------------------------------------------
    base = recent.copy()
    base = safe_merge(base, first_open, "user_id")
    base = safe_merge(base, feature_usage, "user_id")
    base = safe_merge(base, workout_freq, "user_id")
    base = safe_merge(base, retention, "user_id")

    # Fill numeric NaNs with 0
    for col in ["feature_count", "workouts", "d7", "d14", "d30"]:
        if col in base.columns:
            base[col] = base[col].fillna(0).astype(int)

    # -------------------------------------------------------------
    # SEGMENT LABELS
    # -------------------------------------------------------------
    def label_activation(row):
        sd = row["created_at"]
        fo = row["first_open_date"]
        if pd.isna(fo):
            return "Never Opened"
        if fo.date() == sd.date():
            return "Day 0"
        diff = (fo.date() - sd.date()).days
        if 1 <= diff <= 7:
            return "1–7 Days"
        if 8 <= diff <= 30:
            return "8–30 Days"
        return "30+ Days"

    base["activation_speed"] = base.apply(label_activation, axis=1)

    # Feature depth grouping
    def depth_label(x):
        if x >= 3: return "3+ features"
        if x == 2: return "2 features"
        if x == 1: return "1 feature"
        return "0 features"

    base["feature_depth"] = base["feature_count"].apply(depth_label)

    # Workout bucket
    def workout_label(x):
        if x >= 20: return "20+ workouts"
        if 6 <= x <= 19: return "6–19 workouts"
        if 1 <= x <= 5: return "1–5 workouts"
        return "0 workouts"

    base["workout_bucket"] = base["workouts"].apply(workout_label)

    return base


# ============================================================
# 4) FUNNEL COMPUTATION
# ============================================================

def compute_funnel(base, segment_col):
    df = base.copy()

    # If the column doesn’t exist → skip
    if segment_col not in df.columns:
        return pd.DataFrame()

    df["segment_name"] = df[segment_col]

    grouped = df.groupby("segment_name").agg(
        signed_up=("user_id", "count"),
        opened_app=("first_open_date", lambda x: x.notna().sum()),
        engaged_core=("feature_count", lambda x: (x > 0).sum()),
        d7=("d7", "sum"),
        d14=("d14", "sum"),
        d30=("d30", "sum"),
    ).reset_index()

    # Percentages
    grouped["pct_signup_to_open"] = np.round(100 * grouped["opened_app"] / grouped["signed_up"], 2)
    grouped["pct_open_to_engage"] = np.round(100 * grouped["engaged_core"] / grouped["opened_app"].replace(0, np.nan), 2)
    grouped["pct_signup_to_engage"] = np.round(100 * grouped["engaged_core"] / grouped["signed_up"], 2)
    grouped["pct_signup_to_d7"] = np.round(100 * grouped["d7"] / grouped["signed_up"], 2)
    grouped["pct_signup_to_d14"] = np.round(100 * grouped["d14"] / grouped["signed_up"], 2)
    grouped["pct_signup_to_d30"] = np.round(100 * grouped["d30"] / grouped["signed_up"], 2)

    grouped.insert(0, "segment", segment_col)

    return grouped


# ============================================================
# 5) STREAMLIT UI
# ============================================================

st.set_page_config(page_title="Fitness App RCA Dashboard", layout="wide")

st.title("📊 Fitness App — Root Cause Analysis Dashboard")
st.markdown("Upload **Dataset_problem.zip** to begin.")

uploaded = st.file_uploader("Upload ZIP", type=["zip"])

if uploaded:
    dfs = load_zip(uploaded)
    st.success(f"Loaded {len(dfs)} tables")

    # Show available tables
    st.write("### Available Tables:")
    st.write(list(dfs.keys()))

    # Compute Base
    base = compute_base(dfs)

    st.subheader("📌 Base Dataset (after preprocessing)")
    st.dataframe(base.head())

    # Segments to analyze
    segment_cols = [
        "activation_speed",
        "feature_depth",
        "workout_bucket",
        "gender",
        "age_group",
        "plan_type",
    ]

    all_segments_df = []

    for seg in segment_cols:
        result = compute_funnel(base, seg)
        if len(result) > 0:
            all_segments_df.append(result)

    final_output = pd.concat(all_segments_df, ignore_index=True)

    st.subheader("📈 Funnel Metrics by Segment")
    st.dataframe(final_output)

    # Optional: Visualization
    st.subheader("📊 Retention by Segment")
    fig = px.bar(
        final_output,
        x="segment_name",
        y="pct_signup_to_d30",
        color="segment",
        title="D30 Retention by Segment"
    )
    st.plotly_chart(fig, use_container_width=True)
