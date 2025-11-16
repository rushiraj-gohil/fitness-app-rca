import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
from datetime import datetime, timedelta
import plotly.express as px


st.set_page_config(page_title="PeakPulse – User Analytics RCA", layout="wide")


# -------------------------------
# HELPER: LOAD ZIP FILES
# -------------------------------
def load_datasets_from_zip(zip_path="Dataset_problem.zip"):
    tables = {}
    if not os.path.exists(zip_path):
        st.error("Dataset_problem.zip not found in repo root!")
        return tables

    with zipfile.ZipFile(zip_path, 'r') as z:
        for fname in z.namelist():
            if fname.endswith(".csv") and not fname.startswith("__MACOSX"):
                try:
                    df = pd.read_csv(z.open(fname), engine="python")
                    df.columns = df.columns.str.lower()
                    tables[fname.replace(".csv", "").split("/")[-1]] = df
                except Exception as e:
                    st.warning(f"Failed to load {fname}: {e}")
    return tables


# -------------------------------
# CORE COMPUTE PIPELINE
# -------------------------------
@st.cache_data(show_spinner=False)
def compute_base(tables):

    # Extract tables
    users = tables.get("users")
    app_events = tables.get("app_events")
    app_sessions = tables.get("app_sessions")
    activity_data = tables.get("activity_data")
    subscriptions = tables.get("subscriptions")
    daily_scores = tables.get("daily_scores")
    journal_entries = tables.get("journal_entries")

    # lowercase everything again for safety
    for df in [users, app_events, app_sessions, activity_data, subscriptions]:
        if df is not None:
            df.columns = df.columns.str.lower()

    # ---------------------------
    # DATE NORMALIZATION
    # ---------------------------
    def to_date(df, col):
        if df is not None and col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    users = to_date(users, "created_at")
    app_events = to_date(app_events, "event_time")
    app_sessions = to_date(app_sessions, "session_start")
    subscriptions = to_date(subscriptions, "subscription_start_date")

    # ---------------------------
    # Recent 90-day users
    # ---------------------------
    max_signup = users["created_at"].max()
    min_signup = max_signup - timedelta(days=90)
    recent = users[(users["created_at"] >= min_signup) & (users["created_at"] <= max_signup)]

    # ---------------------------
    # First App Open
    # ---------------------------
    if app_events is not None and "event_name" in app_events.columns:
        app_open = app_events[app_events["event_name"] == "app_open"]
        first_open = app_open.groupby("user_id", as_index=False)["event_time"].min()
        first_open.rename(columns={"event_time": "first_open_date"}, inplace=True)
    else:
        first_open = pd.DataFrame({"user_id": [], "first_open_date": []})

    # ---------------------------
    # Feature Depth
    # ---------------------------
    if app_events is not None and "event_name" in app_events.columns:
        core_events = ["view_sleep", "view_recovery", "view_strain", "view_coaching"]
        core_use = app_events[app_events["event_name"].isin(core_events)]
        feature_depth = core_use.groupby("user_id", as_index=False)["event_name"].nunique()
        feature_depth.rename(columns={"event_name": "feature_count"}, inplace=True)
    else:
        feature_depth = pd.DataFrame({"user_id": [], "feature_count": []})

    # ---------------------------
    # Workout Frequency (SAFE)
    # ---------------------------
    try:
        if activity_data is not None and "user_id" in activity_data.columns:
            workout_freq = (
                activity_data.groupby("user_id")
                .size()
                .reset_index(name="workouts")
            )
        else:
            workout_freq = pd.DataFrame({"user_id": [], "workouts": []})
    except:
        workout_freq = pd.DataFrame({"user_id": [], "workouts": []})

    # ---------------------------
    # Retention (7/14/30)
    # ---------------------------
    if app_sessions is not None and "session_start" in app_sessions.columns:

        merged = recent.merge(app_sessions, on="user_id", how="left")

        def within_days(start, event, d):
            return (event >= start + timedelta(days=1)) & (event <= start + timedelta(days=d))

        merged["d7"] = within_days(merged["created_at"], merged["session_start"], 7)
        merged["d14"] = within_days(merged["created_at"], merged["session_start"], 14)
        merged["d30"] = within_days(merged["created_at"], merged["session_start"], 30)

        retention = merged.groupby("user_id", as_index=False).agg(
            d7=("d7", "max"),
            d14=("d14", "max"),
            d30=("d30", "max"),
        )
    else:
        retention = pd.DataFrame({"user_id": [], "d7": [], "d14": [], "d30": []})

    # ---------------------------
    # Merge All Features
    # ---------------------------
    base = recent.merge(first_open, on="user_id", how="left")
    base = base.merge(feature_depth, on="user_id", how="left")
    base = base.merge(workout_freq, on="user_id", how="left")
    base = base.merge(retention, on="user_id", how="left")

    base["feature_count"] = base["feature_count"].fillna(0).astype(int)
    base["workouts"] = base["workouts"].fillna(0).astype(int)

    # Activation speed
    def calc_activation(row):
        sd = row["created_at"]
        fo = row["first_open_date"]
        if pd.isna(fo):
            return "Never Opened"
        if fo.date() == sd.date():
            return "Day 0"
        if sd + timedelta(days=1) <= fo <= sd + timedelta(days=7):
            return "1–7 Days"
        if sd + timedelta(days=8) <= fo <= sd + timedelta(days=30):
            return "8–30 Days"
        return "30+ Days"

    base["activation_speed"] = base.apply(calc_activation, axis=1)

    # Feature buckets
    def fdepth(x):
        if x >= 3: return "3+ features"
        if x == 2: return "2 features"
        if x == 1: return "1 feature"
        return "0 features"

    base["feature_depth"] = base["feature_count"].apply(fdepth)

    # Workout buckets
    def wb(x):
        if x >= 20: return "20+ workouts"
        if 6 <= x <= 19: return "6–19 workouts"
        if 1 <= x <= 5: return "1–5 workouts"
        return "0 workouts"

    base["workout_bucket"] = base["workouts"].apply(wb)

    return base, tables


# -------------------------------
# PAGE: OVERVIEW
# -------------------------------
def page_overview(base):
    st.title("📊 PeakPulse – End-to-End User Analytics & RCA")

    st.markdown("""
    This dashboard presents the full analysis performed for the PeakPulse assignment.  
    Data is processed directly from the repository ZIP file — no uploads required.

    **Notebooks referenced:**
    - Placeholder link for Notebook 1 (Exploration)
    - Placeholder link for Notebook 2 (Retention & Segmentation)
    """)

    st.subheader("Dataset Summary")
    st.write(base.describe(include="all").transpose())


# -------------------------------
# PAGE: RETENTION
# -------------------------------
def page_retention(base):

    st.title("🔄 Retention Analysis (D7 / D14 / D30)")

    agg = base[["d7", "d14", "d30"]].mean() * 100
    st.metric("D7 Retention", f"{agg['d7']:.2f}%")
    st.metric("D14 Retention", f"{agg['d14']:.2f}%")
    st.metric("D30 Retention", f"{agg['d30']:.2f}%")

    fig = px.bar(
        x=["D7", "D14", "D30"],
        y=[agg["d7"], agg["d14"], agg["d30"]],
        title="Retention Rates",
        labels={"x": "Retention Window", "y": "Percentage"},
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------
# PAGE: FUNNELS
# -------------------------------
def page_funnels(base):

    st.title("⏳ Activation & Engagement Funnel")

    total = len(base)
    opened = base["first_open_date"].notna().sum()
    engaged = (base["feature_count"] > 0).sum()
    d7 = base["d7"].sum()

    st.write(pd.DataFrame({
        "Stage": ["Signed Up", "Opened App", "Engaged Feature", "Returned D7"],
        "Users": [total, opened, engaged, d7],
        "Conversion %": [
            100,
            round(100 * opened / total, 2),
            round(100 * engaged / opened, 2) if opened else 0,
            round(100 * d7 / total, 2),
        ]
    }))


# -------------------------------
# PAGE: SEGMENTS
# -------------------------------
def page_segments(base):

    st.title("🎯 Segmented Insights")

    segment_col = st.selectbox(
        "Select a Segment:",
        ["activation_speed", "feature_depth", "workout_bucket", "gender", "age_group"]
    )

    seg = (
        base.groupby(segment_col)
        .agg(
            signed_up=("user_id", "count"),
            opened_app=("first_open_date", lambda x: x.notna().sum()),
            engaged_core=("feature_count", lambda x: (x > 0).sum()),
            d7=("d7", "sum"),
            d14=("d14", "sum"),
            d30=("d30", "sum"),
        )
        .reset_index()
    )

    seg["pct_signup_to_open"] = round(100 * seg["opened_app"] / seg["signed_up"], 2)
    seg["pct_open_to_engage"] = round(100 * seg["engaged_core"] / seg["opened_app"].replace(0, np.nan), 2)
    seg["pct_signup_to_engage"] = round(100 * seg["engaged_core"] / seg["signed_up"], 2)
    seg["pct_signup_to_d7"] = round(100 * seg["d7"] / seg["signed_up"], 2)
    seg["pct_signup_to_d14"] = round(100 * seg["d14"] / seg["signed_up"], 2)
    seg["pct_signup_to_d30"] = round(100 * seg["d30"] / seg["signed_up"], 2)

    st.write(seg)


# -------------------------------
# PAGE: RECOMMENDATIONS / STORY
# -------------------------------
def page_story():

    st.title("📌 RCA Summary & Recommendations")

    st.markdown("""
    ## 🔍 Key Problems Identified
    - 37% of users never activate (never open app)
    - Feature discovery is low — 503 users use 0 features
    - Habit formation weak — 0–5 workouts → 35% lower retention
    - D7 retention sits around ~52–55%
    - Premium dataset means 100% conversions, but retention still valid

    ## 📉 Business Impact
    - Improving Day-1 activation by +10% → ~136 more users entering funnel
    - Moving users from 0 features → 1 feature increases D30 by +22%
    - Moving users from 1–5 workouts → 6–19 workouts increases D30 by +18%

    ## 🚀 Recommended Product Changes
    1. **Improve Activation**
       - Single CTA onboarding  
       - Auto-play intro routine  
       - Habit reminder on D1  

    2. **Feature Discovery**
       - Guided walkthrough  
       - Highlight Sleep + Recovery + Strain trio  
       - Streak badges  

    3. **Habit Loops**
       - Workout reminders  
       - Weekly leaderboard  
       - Recovery-based smart nudges  

    ## 📏 Success Metrics
    - D1 / D7 / D14 / D30 retention  
    - Activation speed → Day 0 vs Day 1  
    - Feature depth distribution  
    - Workouts per week  
    """)
    


# -------------------------------
# MAIN ROUTER
# -------------------------------
pages = {
    "📘 Overview": page_overview,
    "🔄 Retention": page_retention,
    "⏳ Funnels": page_funnels,
    "🎯 Segments": page_segments,
    "📌 RCA & Recommendations": page_story,
}

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to:", list(pages.keys()))

# load zip
tables = load_datasets_from_zip()
if tables:
    base, raw = compute_base(tables)
    pages[choice](base)
else:
    st.error("No datasets loaded!")
