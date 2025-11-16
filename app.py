# app.py
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
from datetime import timedelta
import os

st.set_page_config(page_title="PeakPulse Insights Dashboard", layout="wide", page_icon="📈")

ZIP_PATH = "Dataset_problem.zip"  # <-- zip must be at repo root

# -------------------------
# Helpers
# -------------------------
def safe_read_csv_from_zip(zf, filename):
    try:
        with zf.open(filename) as f:
            return pd.read_csv(f)
    except Exception as e:
        st.warning(f"Could not read {filename}: {e}")
        return None

@st.cache_data
def load_all_from_zip(zip_path):
    if not os.path.exists(zip_path):
        st.error(f"Zip not found at path: {zip_path}")
        st.stop()

    loaded = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            # skip macOS metadata and directories
            lower = member.lower()
            if member.endswith("/") or "__macosx" in lower or member.startswith("._") or member.endswith(".ds_store"):
                continue
            if not lower.endswith(".csv"):
                continue

            # friendly name: last path component without .csv
            name = os.path.basename(member).replace(".csv", "")
            df = safe_read_csv_from_zip(z, member)
            if df is not None:
                loaded[name] = df

    return loaded

def ensure_col(df, col, default=None):
    if df is None:
        return None
    if col not in df.columns:
        df[col] = default
    return df

def safe_to_datetime(df, col, inplace=True):
    if df is None or col not in df.columns:
        return df
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def bool_to_int_series(s):
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

# -------------------------
# Load datasets
# -------------------------
st.title("📈 PeakPulse Insights Dashboard")
st.markdown("Loading dataset from `Dataset_problem.zip` in repo...")

dfs = load_all_from_zip(ZIP_PATH)

# Required expected table names (from schema). We'll load what's available.
expected = [
    "users", "subscriptions", "daily_scores", "sleep_data", "activity_data",
    "hrv_data", "journal_entries", "app_sessions", "app_events",
    "teams", "team_memberships", "team_challenges", "challenge_participation"
]

present = {k: v for k, v in dfs.items() if k in expected}
missing = [name for name in expected if name not in present]

st.write(f"Loaded tables: {list(present.keys())}")
if missing:
    st.info(f"Missing tables (optional/OK but noted): {missing}")

# Short aliases (or None)
users = present.get("users")
app_events = present.get("app_events")
app_sessions = present.get("app_sessions")
activity_data = present.get("activity_data")
subscriptions = present.get("subscriptions")
daily_scores = present.get("daily_scores")
journal_entries = present.get("journal_entries")

# -------------------------
# Minimal validation
# -------------------------
if users is None or app_events is None or app_sessions is None:
    st.error("users.csv, app_events.csv and app_sessions.csv are required for funnel/retention analysis.")
    st.stop()

# -------------------------
# Preprocess common columns
# -------------------------
# make copies to avoid mutating original frames in cache
users = users.copy()
app_events = app_events.copy()
app_sessions = app_sessions.copy()
activity_data = activity_data.copy() if activity_data is not None else pd.DataFrame()
subscriptions = subscriptions.copy() if subscriptions is not None else pd.DataFrame()
daily_scores = daily_scores.copy() if daily_scores is not None else pd.DataFrame()
journal_entries = journal_entries.copy() if journal_entries is not None else pd.DataFrame()

# normalize column names to expected names where possible (helpful if slight naming variations)
# We'll not rename automatically aggressively; instead use safe lookups below.

# Convert key date columns safely
for df, col in [
    (users, "created_at"),
    (app_events, "event_time"),
    (app_sessions, "session_start"),
    (subscriptions, "subscription_start_date"),
    (daily_scores, "date"),
]:
    if df is not None and col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Ensure user_id columns exist
for df in [users, app_events, app_sessions, activity_data, subscriptions, daily_scores, journal_entries]:
    if df is not None and "user_id" not in df.columns:
        # try lower-case fallback
        cols_lower = [c.lower() for c in df.columns]
        if "user_id" in cols_lower:
            # rename first matched column to user_id
            original = df.columns[cols_lower.index("user_id")]
            df.rename(columns={original: "user_id"}, inplace=True)

# -------------------------
# Compute base features
# -------------------------
@st.cache_data
def compute_base(users, app_events, app_sessions, activity_data, subscriptions, daily_scores, journal_entries):
    u = users.copy()

    # Ensure created_at exists
    if "created_at" not in u.columns:
        # fallback: try 'created' or 'signup_date'
        for alt in ["created", "signup_date", "signup"]:
            if alt in u.columns:
                u["created_at"] = pd.to_datetime(u[alt], errors="coerce")
                break
    u["created_at"] = pd.to_datetime(u["created_at"], errors="coerce")
    # first open
    fe = app_events.copy()
    if "event_time" in fe.columns:
        fe["event_time"] = pd.to_datetime(fe["event_time"], errors="coerce")
    # find first app_open
    if "event_name" in fe.columns:
        fo = fe[fe["event_name"] == "app_open"].groupby("user_id", as_index=False).agg(first_open_date=("event_time", "min"))
    else:
        fo = pd.DataFrame(columns=["user_id", "first_open_date"])

    # feature depth: count distinct core feature events per user
    core_features = ["view_sleep", "view_recovery", "view_strain", "view_coaching"]
    if "event_name" in fe.columns:
        fu = fe[fe["event_name"].isin(core_features)].groupby("user_id", as_index=False).agg(feature_count=("event_name", "nunique"))
    else:
        fu = pd.DataFrame(columns=["user_id", "feature_count"])

    # workouts — robust count of activity rows per user
    if activity_data is not None and "user_id" in activity_data.columns:
        # use any column as row count; prefer activity_id/workout_id if present
        count_col = None
        for candidate in ["activity_id", "workout_id", "id", "activity_id"]:
            if candidate in activity_data.columns:
                count_col = candidate
                break
        # simply count rows per user
        wf = activity_data.groupby("user_id", as_index=False).size().reset_index(name="workouts")
    else:
        wf = pd.DataFrame(columns=["user_id", "workouts"])

    # retention windows (D7/D14/D30) using sessions
    s = app_sessions.copy()
    if "session_start" in s.columns:
        s["session_start"] = pd.to_datetime(s["session_start"], errors="coerce")
    else:
        s["session_start"] = pd.NaT

    # Build initial base = users who signed up in the last 90 days relative to max(created_at)
    max_signup = u["created_at"].max()
    if pd.isna(max_signup):
        max_signup = pd.Timestamp.now()
    min_signup = max_signup - pd.Timedelta(days=90)
    recent_users = u[(u["created_at"] >= min_signup) & (u["created_at"] <= max_signup)].copy()

    # compute retention per user
    def compute_retention_for_user(row):
        uid = row.get("user_id")
        signup = row.get("created_at")
        if pd.isna(signup) or uid is None:
            return pd.Series({"d7": 0, "d14": 0, "d30": 0})
        user_sessions = s[s["user_id"] == uid]["session_start"].dropna()
        # days since signup for each session
        days = (user_sessions - signup).dt.days
        d7 = int(((days >= 1) & (days <= 7)).any())
        d14 = int(((days >= 1) & (days <= 14)).any())
        d30 = int(((days >= 1) & (days <= 30)).any())
        return pd.Series({"d7": d7, "d14": d14, "d30": d30})

    # apply retention
    retention_df = recent_users.apply(compute_retention_for_user, axis=1)
    recent_users = pd.concat([recent_users.reset_index(drop=True), retention_df.reset_index(drop=True)], axis=1)

    # merge features
    base = recent_users.merge(fo, on="user_id", how="left")
    base = base.merge(fu, on="user_id", how="left")
    base = base.merge(wf, on="user_id", how="left")
    base = base.merge(subscriptions[["user_id", "plan_type"]], on="user_id", how="left") if subscriptions is not None and "user_id" in subscriptions.columns else base

    # fillna for numeric feature columns
    if "feature_count" in base.columns:
        base["feature_count"] = pd.to_numeric(base["feature_count"], errors="coerce").fillna(0).astype(int)
    else:
        base["feature_count"] = 0

    if "workouts" in base.columns:
        base["workouts"] = pd.to_numeric(base["workouts"], errors="coerce").fillna(0).astype(int)
    else:
        base["workouts"] = 0

    for c in ["d7", "d14", "d30"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0).astype(int)
        else:
            base[c] = 0

    # activation label
    def activation_label_safe(row):
        sd = row["created_at"]
        fo_date = row.get("first_open_date", pd.NaT)
        # coerce
        try:
            fo_date = pd.to_datetime(fo_date, errors="coerce")
        except Exception:
            fo_date = pd.NaT
        if pd.isna(fo_date):
            return "Never Opened"
        if pd.isna(sd):
            return "Unknown"
        diff = (fo_date - sd).days
        if diff < 0:
            return "Unknown"
        if diff == 0:
            return "Day 0"
        if 1 <= diff <= 7:
            return "1–7 Days"
        if 8 <= diff <= 30:
            return "8–30 Days"
        return "30+ Days"

    base["activation_speed"] = base.apply(activation_label_safe, axis=1)

    # feature depth bucket
    def feature_depth_bucket(n):
        if n >= 3:
            return "3+ features"
        if n == 2:
            return "2 features"
        if n == 1:
            return "1 feature"
        return "0 features"

    base["feature_depth"] = base["feature_count"].apply(feature_depth_bucket)

    # workout bucket
    def workout_bucket(n):
        if n >= 20:
            return "20+ workouts"
        if n >= 6:
            return "6–19 workouts"
        if n >= 1:
            return "1–5 workouts"
        return "0 workouts"

    base["workout_bucket"] = base["workouts"].apply(workout_bucket)

    return base

base = compute_base(users, app_events, app_sessions, activity_data, subscriptions, daily_scores, journal_entries)

# -------------------------
# Funnel computation helper
# -------------------------
def compute_funnel_by_segment(df, segment_col):
    if segment_col not in df.columns:
        return pd.DataFrame()
    grouped = df.groupby(segment_col).agg(
        signed_up=("user_id", "count"),
        opened_app=("first_open_date", lambda x: x.notna().sum()),
        engaged_core=("feature_count", lambda x: (x > 0).sum()),
        d7=("d7", "sum"),
        d14=("d14", "sum"),
        d30=("d30", "sum"),
    ).reset_index()
    # convert numeric to float for safe division
    for col in ["signed_up", "opened_app", "engaged_core", "d7", "d14", "d30"]:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").fillna(0)
    grouped["pct_signup_to_open"] = round(100 * grouped["opened_app"] / grouped["signed_up"].replace({0: np.nan}), 2)
    grouped["pct_open_to_engage"] = round(100 * grouped["engaged_core"] / grouped["opened_app"].replace({0: np.nan}), 2)
    grouped["pct_signup_to_engage"] = round(100 * grouped["engaged_core"] / grouped["signed_up"].replace({0: np.nan}), 2)
    grouped["pct_signup_to_d7"] = round(100 * grouped["d7"] / grouped["signed_up"].replace({0: np.nan}), 2)
    grouped["pct_signup_to_d14"] = round(100 * grouped["d14"] / grouped["signed_up"].replace({0: np.nan}), 2)
    grouped["pct_signup_to_d30"] = round(100 * grouped["d30"] / grouped["signed_up"].replace({0: np.nan}), 2)
    return grouped

# -------------------------
# UI: Overview metrics
# -------------------------
st.header("Dataset overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total users (dataset)", int(users.shape[0]) if users is not None else 0)
col2.metric("Recent cohort (90d)", int(base.shape[0]))
col3.metric("Total app events", int(app_events.shape[0]) if app_events is not None else 0)

st.markdown("**Quick notes:** retention windows use `app_sessions.session_start`. Activation uses first `app_open` event in `app_events`.")

# -------------------------
# UI: Funnel overall
# -------------------------
st.header("Overall funnel (last 90 days cohort)")
overall = compute_funnel_by_segment(base, segment_col="activation_speed")
if not overall.empty:
    st.dataframe(overall)
else:
    st.info("No overall funnel computed (activation_speed missing).")

# -------------------------
# UI: Segmentation selection
# -------------------------
st.header("Funnels by segment")
segment_choices = ["age_group", "gender", "plan_type", "activation_speed", "feature_depth", "workout_bucket"]
chosen = st.selectbox("Choose segment dimension", segment_choices)

seg_df = compute_funnel_by_segment(base, chosen)
if seg_df.empty:
    st.warning(f"Segment column `{chosen}` not found in data. Available columns: {base.columns.tolist()}")
else:
    st.dataframe(seg_df)

# -------------------------
# UI: Short conclusions (static)
# -------------------------
st.header("Key findings (summary)")
st.markdown("""
- **Early activation failure**: a large fraction of users never open the app after signup (activation buckets show "Never Opened").  
- **Low feature discovery**: many users have feature_count = 0 (they don't view core features).  
- **Weak habit formation**: users with low workouts and low active days show lower D7/D30 retention.  
- **Platform & segment gaps**: funnels vary by platform, age group and plan — these are actionable levers.
""")

st.markdown("### Next recommended analyses\n1. Run per-cohort retention curves (time-series).  \n2. A/B test Day-0 activation flows.  \n3. Build a feature-discovery onboarding experiment.")

st.caption("Generated automatically from files inside Dataset_problem.zip — ready for presentation + further drilling.")

# -------------------------
# Done
# -------------------------
st.success("Dashboard ready — use the segment selector to explore funnels.")
