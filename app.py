# app.py
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
from datetime import timedelta
import plotly.express as px

st.set_page_config(page_title="PeakPulse RCA", layout="wide")

ZIP_PATH = "Dataset_problem.zip"  # <-- must be in repo root

# -----------------------
# Helpers
# -----------------------
@st.cache_data
def load_from_zip(zip_path):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"{zip_path} not found in repo root.")
    tables = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            lower = member.lower()
            if member.endswith("/") or "__macosx" in lower or member.startswith("._") or member.endswith(".ds_store"):
                continue
            if not lower.endswith(".csv"):
                continue
            name = os.path.basename(member).replace(".csv", "")
            try:
                df = pd.read_csv(z.open(member))
                tables[name] = df
            except Exception as e:
                # skip unreadable file but log
                st.warning(f"Skipped {member}: {e}")
    return tables

def safe_to_datetime(df, col):
    if df is None or col not in df.columns:
        return df
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def safe_group_count(df, by_col, new_col):
    if df is None or by_col not in df.columns:
        return pd.DataFrame({by_col: [], new_col: []})
    return df.groupby(by_col).size().reset_index(name=new_col)

# -----------------------
# Compute base features
# -----------------------
@st.cache_data
def compute_base(tables):
    users = tables.get("users", pd.DataFrame()).copy()
    app_events = tables.get("app_events", pd.DataFrame()).copy()
    app_sessions = tables.get("app_sessions", pd.DataFrame()).copy()
    activity_data = tables.get("activity_data", pd.DataFrame()).copy()
    subscriptions = tables.get("subscriptions", pd.DataFrame()).copy()
    daily_scores = tables.get("daily_scores", pd.DataFrame()).copy()
    journal_entries = tables.get("journal_entries", pd.DataFrame()).copy()

    # safe datetime conversions
    users = safe_to_datetime(users, "created_at")
    app_events = safe_to_datetime(app_events, "event_time")
    app_sessions = safe_to_datetime(app_sessions, "session_start")
    subscriptions = safe_to_datetime(subscriptions, "subscription_start_date")
    daily_scores = safe_to_datetime(daily_scores, "date")

    # identify recent cohort window (last 90 days relative to max signup)
    max_signup = users["created_at"].max() if "created_at" in users.columns else pd.Timestamp.now()
    min_signup = max_signup - timedelta(days=90)
    recent = users[(users.get("created_at") >= min_signup) & (users.get("created_at") <= max_signup)].copy()

    # FIRST OPEN
    first_open = pd.DataFrame(columns=["user_id", "first_open_date"])
    if (app_events is not None) and ("event_name" in app_events.columns):
        fo = app_events[app_events["event_name"] == "app_open"]
        if not fo.empty:
            first_open = fo.groupby("user_id", as_index=False).agg(first_open_date=("event_time", "min"))

    # FEATURE DEPTH
    core_features = {"view_sleep", "view_recovery", "view_strain", "view_coaching"}
    feature_usage = pd.DataFrame(columns=["user_id", "feature_count"])
    if (app_events is not None) and ("event_name" in app_events.columns):
        fe = app_events[app_events["event_name"].isin(core_features)]
        if not fe.empty:
            feature_usage = fe.groupby("user_id", as_index=False).agg(feature_count=("event_name", "nunique"))

    # WORKOUTS — only if user_id exists in activity_data
    workout_freq = pd.DataFrame(columns=["user_id", "workouts"])
    if (activity_data is not None) and ("user_id" in activity_data.columns):
        workout_freq = activity_data.groupby("user_id", as_index=False).size().reset_index(name="workouts")

    # RETENTION D7/D14/D30 using app_sessions
    retention = pd.DataFrame(columns=["user_id", "d7", "d14", "d30"])
    if (app_sessions is not None) and ("session_start" in app_sessions.columns) and ("user_id" in app_sessions.columns):
        # join recent users with their sessions and compute flags
        sessions = app_sessions[["user_id", "session_start"]].copy()
        rec = recent[["user_id", "created_at"]].copy()
        if not sessions.empty and not rec.empty:
            merged = rec.merge(sessions, on="user_id", how="left")
            merged["days_since_signup"] = (merged["session_start"] - merged["created_at"]).dt.days
            # flags per row
            merged["d7_flag"] = ((merged["days_since_signup"] >= 1) & (merged["days_since_signup"] <= 7)).fillna(False)
            merged["d14_flag"] = ((merged["days_since_signup"] >= 1) & (merged["days_since_signup"] <= 14)).fillna(False)
            merged["d30_flag"] = ((merged["days_since_signup"] >= 1) & (merged["days_since_signup"] <= 30)).fillna(False)
            retention = merged.groupby("user_id", as_index=False).agg(
                d7=("d7_flag", "max"),
                d14=("d14_flag", "max"),
                d30=("d30_flag", "max"),
            )
            # cast bool->int
            for c in ["d7","d14","d30"]:
                retention[c] = retention[c].astype(int)
    else:
        # fallback: set zeros for recent users
        if "user_id" in recent.columns:
            retention = pd.DataFrame({"user_id": recent["user_id"].tolist(), "d7": 0, "d14": 0, "d30": 0})

    # MERGE everything into base
    base = recent.copy()
    if not first_open.empty:
        base = base.merge(first_open, on="user_id", how="left")
    if not feature_usage.empty:
        base = base.merge(feature_usage, on="user_id", how="left")
    if not workout_freq.empty:
        base = base.merge(workout_freq, on="user_id", how="left")
    if not retention.empty:
        base = base.merge(retention, on="user_id", how="left")
    if (subscriptions is not None) and ("user_id" in subscriptions.columns) and ("plan_type" in subscriptions.columns):
        # take latest subscription per user if multiple
        subs_latest = subscriptions.sort_values("subscription_start_date").drop_duplicates("user_id", keep="last")[["user_id", "plan_type"]]
        base = base.merge(subs_latest, on="user_id", how="left")

    # clean missing numeric columns
    base["feature_count"] = pd.to_numeric(base.get("feature_count", 0), errors="coerce").fillna(0).astype(int)
    base["workouts"] = pd.to_numeric(base.get("workouts", 0), errors="coerce").fillna(0).astype(int)
    for c in ["d7", "d14", "d30"]:
        base[c] = pd.to_numeric(base.get(c, 0), errors="coerce").fillna(0).astype(int)

    # LABELS: activation_speed, feature_depth, workout_bucket
    def activation_label(row):
        fo = row.get("first_open_date")
        sd = row.get("created_at")
        try:
            if pd.isna(fo):
                return "Never Opened"
            if pd.isna(sd):
                return "Unknown"
            diff = (pd.to_datetime(fo).date() - pd.to_datetime(sd).date()).days
            if diff < 0: return "Unknown"
            if diff == 0: return "Day 0"
            if 1 <= diff <= 7: return "1–7 Days"
            if 8 <= diff <= 30: return "8–30 Days"
            return "30+ Days"
        except:
            return "Unknown"

    base["activation_speed"] = base.apply(activation_label, axis=1)

    def depth_label(x):
        if x >= 3: return "3+ features"
        if x == 2: return "2 features"
        if x == 1: return "1 feature"
        return "0 features"
    base["feature_depth"] = base["feature_count"].apply(depth_label)

    def workout_bucket(x):
        if x >= 20: return "20+ workouts"
        if 6 <= x <= 19: return "6–19 workouts"
        if 1 <= x <= 5: return "1–5 workouts"
        return "0 workouts"
    base["workout_bucket"] = base["workouts"].apply(workout_bucket)

    return base, {"users": users, "app_events": app_events, "app_sessions": app_sessions,
                  "activity_data": activity_data, "subscriptions": subscriptions,
                  "daily_scores": daily_scores, "journal_entries": journal_entries}

# -----------------------
# Funnel helper
# -----------------------
def funnel_by_segment(base, segment_col):
    if segment_col not in base.columns:
        return pd.DataFrame()
    g = base.groupby(segment_col).agg(
        signed_up=("user_id", "count"),
        opened_app=("first_open_date", lambda x: x.notna().sum()),
        engaged_core=("feature_count", lambda x: (x > 0).sum()),
        d7=("d7", "sum"),
        d14=("d14", "sum"),
        d30=("d30", "sum"),
    ).reset_index().rename(columns={segment_col: "segment_name"})
    # percentages
    g["pct_signup_to_open"] = np.round(100 * g["opened_app"] / g["signed_up"].replace(0, np.nan), 2)
    g["pct_open_to_engage"] = np.round(100 * g["engaged_core"] / g["opened_app"].replace(0, np.nan), 2)
    g["pct_signup_to_engage"] = np.round(100 * g["engaged_core"] / g["signed_up"].replace(0, np.nan), 2)
    g["pct_signup_to_d7"] = np.round(100 * g["d7"] / g["signed_up"].replace(0, np.nan), 2)
    g["pct_signup_to_d14"] = np.round(100 * g["d14"] / g["signed_up"].replace(0, np.nan), 2)
    g["pct_signup_to_d30"] = np.round(100 * g["d30"] / g["signed_up"].replace(0, np.nan), 2)
    g.insert(0, "segment", segment_col)
    return g

# -----------------------
# UI: multi-page (sidebar)
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "1. Exploratory Analysis",
    "2. Problem Identification",
    "3. Business Impact",
    "4. Recommendations",
    "5. Success Metrics",
    "6. Notebooks / Links"
])

# load data once
try:
    tables = load_from_zip(ZIP_PATH)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

base, raw_tables = compute_base(tables)

# Page 1: Exploratory Analysis
if page == "1. Exploratory Analysis":
    st.title("Exploratory Analysis")
    st.markdown("Cohort & funnel overview for recent 90-day signups.")
    st.write("Cohort size (recent 90d):", int(base.shape[0]))

    st.subheader("Overall funnel (activation_speed)")
    funnel_overall = funnel_by_segment(base, "activation_speed")
    st.dataframe(funnel_overall)

    st.subheader("Platform comparison (if available)")
    # platform might be 'app_platform' or 'platform' or not present
    plat_col = None
    for c in ["app_platform", "platform"]:
        if c in base.columns:
            plat_col = c
            break
    if plat_col:
        st.dataframe(funnel_by_segment(base, plat_col))
    else:
        st.info("Platform column not found in users/datasets.")

    st.subheader("Feature depth distribution")
    fig = px.histogram(base, x="feature_depth", title="Feature depth (count of core features used)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Retention rates by activation bucket")
    if not funnel_overall.empty:
        fig2 = px.bar(funnel_overall, x="segment_name", y=["pct_signup_to_d7", "pct_signup_to_d30"],
                      title="Retention (D7 & D30) by activation bucket")
        st.plotly_chart(fig2, use_container_width=True)

# Page 2: Problem Identification
elif page == "2. Problem Identification":
    st.title("Problem Identification")
    st.markdown("From exploratory work, top issues identified:")
    st.markdown("""
    1. **Early activation failure** — many users never open the app after signup (activation bucket 'Never Opened').
    2. **Low feature discovery** — substantial users have `feature_count == 0`.
    3. **Weak habit formation** — low workout counts and low early session counts correlate with lower D7/D30.
    4. **Segment gaps** — differences across age/plan/gender/platform.
    5. **Dataset quirk** — dataset contains paid users only; free→premium funnel cannot be validated.
    """)
    st.subheader("Evidence: Activation buckets")
    st.dataframe(funnel_by_segment(base, "activation_speed"))

    st.subheader("Evidence: Feature depth vs D30 retention")
    if "feature_depth" in base.columns:
        feat = funnel_by_segment(base, "feature_depth")
        st.dataframe(feat)
    else:
        st.info("feature_depth not present.")

# Page 3: Business Impact
elif page == "3. Business Impact":
    st.title("Business Impact")
    st.markdown("Quantify impact of improving early activation & feature discovery.")
    total_users = int(raw_tables.get("users", pd.DataFrame()).shape[0]) if raw_tables.get("users") is not None else base.shape[0]
    st.write("Total users in dataset:", total_users)
    # simple revenue calc: assume $30/mo average
    arpu = 30
    st.write("Assumed ARPU (monthly):", f"${arpu}")

    # compute current D30 retention overall
    current_d30 = base["d30"].mean()  # fraction 0..1
    st.write("Current D30 retention (approx):", f"{current_d30:.2%}")

    # hypothetical improvement: +10% points in Day1 open -> estimate users retained
    improvement_points = st.slider("Hypothetical D1 open lift (percentage points)", 0, 30, 10)
    # assume D1 lift translates to D30 lift proportionally by 50% (conservative)
    d30_lift = improvement_points * 0.5 / 100
    new_d30 = current_d30 + d30_lift
    additional_revenue_monthly = (new_d30 - current_d30) * total_users * arpu
    st.metric("Estimated additional MRR (conservative)", f"${additional_revenue_monthly:,.0f}")

# Page 4: Recommendations
elif page == "4. Recommendations":
    st.title("Recommendations")
    st.markdown("""
    **Top recommendations (prioritised):**

    1. Improve Day-0 activation: tutorials, push/email reminders, one-click device pairing flow.  
    2. Increase feature discovery: guided walkthroughs, contextual banners to highlight sleep/strain/coaching.  
    3. Habit formation nudges: streaks, micro-goals (first 7-day routine), and weekly summaries.  
    4. Re-engagement flows for 'Never Opened' and low-usage cohorts with personalized messaging.
    """)
    st.subheader("Suggested experiments")
    st.markdown("""
    - A/B test a Day-0 CTA vs. control — measure D1 open and D7 retention.  
    - Run onboarding variant that forces first core feature view — measure feature_depth and D30.  
    - Push campaign for users who signed up but never opened — measure open rate lift.
    """)

# Page 5: Success Metrics
elif page == "5. Success Metrics":
    st.title("Success Metrics")
    st.markdown("""
    Track these KPIs:
    - D1, D7, D30 retention (cohort curves)  
    - Signup → Open % (activation)  
    - Feature depth (share using 1/2/3+ core features)  
    - Weekly active workouts and session frequency  
    - Renewal / churn rate (monthly)
    """)
    st.subheader("Current baseline (examples)")
    st.write("Signup -> Open (overall):", f"{(base['first_open_date'].notna().mean()*100):.2f}%")
    st.write("Feature_count > 0 (share):", f"{(base['feature_count']>0).mean()*100:.2f}%")
    st.write("Avg workouts (recent cohort):", f"{base['workouts'].mean():.2f}")

# Page 6: Notebooks / Links
elif page == "6. Notebooks / Links":
    st.title("Notebooks / Links")
    st.markdown("Placeholders — add links to your notebooks or Notion pages here.")
    st.markdown("- [fam_v2.ipynb]()  \n- [fam_funnel_v1.ipynb]()")
