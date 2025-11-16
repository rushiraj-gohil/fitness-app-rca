import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from datetime import timedelta

st.set_page_config(page_title="Fitness App RCA Dashboard", layout="wide")

# -----------------------------------------------------------
# 1) SAFE FILE LOADING (Fixes macOS __MACOSX issues)
# -----------------------------------------------------------

def load_dataset(zip_file):
    users = app_events = app_sessions = activity_data = subscriptions = None

    with zipfile.ZipFile(zip_file, 'r') as z:
        for filename in z.namelist():

            # Skip macOS junk files
            if filename.startswith("__MACOSX") or filename.startswith("._") or "/._" in filename:
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


# -----------------------------------------------------------
# 2) SAFE ACTIVATION SPEED FUNCTION
# -----------------------------------------------------------

def activation_label(row):
    sd = pd.to_datetime(row["signup_date"], errors="coerce")
    fo = pd.to_datetime(row["first_open_date"], errors="coerce")

    if pd.isna(sd):
        return "Unknown"

    if pd.isna(fo):
        return "Never Opened"

    diff = (fo - sd).days

    if diff < 0:
        return "Unknown"
    elif diff == 0:
        return "Day 0"
    elif 1 <= diff <= 7:
        return "1–7 Days"
    elif 8 <= diff <= 30:
        return "8–30 Days"
    else:
        return "30+ Days"


# -----------------------------------------------------------
# 3) MAIN PREPROCESSING + METRICS COMPUTATION
# -----------------------------------------------------------

@st.cache_data
def preprocess_and_compute(users, app_events, app_sessions, activity_data, subscriptions):

    # Standardize datetime fields
    users["signup_date"] = pd.to_datetime(users["created_at"], errors="coerce")
    app_events["event_time"] = pd.to_datetime(app_events["event_time"], errors="coerce")
    app_sessions["session_start"] = pd.to_datetime(app_sessions["session_start"], errors="coerce")
    subscriptions["subscription_start_date"] = pd.to_datetime(subscriptions["subscription_start_date"], errors="coerce")

    # -----------------------
    # FIRST APP OPEN
    # -----------------------
    first_open = (
        app_events[app_events["event_name"] == "app_open"]
        .groupby("user_id")["event_time"]
        .min()
        .reset_index()
        .rename(columns={"event_time": "first_open_date"})
    )

    # -----------------------
    # FEATURE DEPTH
    # -----------------------
    feature_events = [
        "view_sleep", "view_recovery", "view_strain", "view_coaching"
    ]

    feature_usage = (
        app_events[app_events["event_name"].isin(feature_events)]
        .groupby("user_id")["event_name"]
        .nunique()
        .reset_index()
        .rename(columns={"event_name": "feature_count"})
    )

    # -----------------------
    # WORKOUT ACTIVITY
    # -----------------------
    workout_freq = (
        activity_data.groupby("user_id")
        .size()
        .reset_index(name="workouts")
    )

    # -----------------------
    # D7 / D14 / D30 RETENTION
    # -----------------------
    merged_sessions = app_sessions.merge(
        users[["user_id", "signup_date"]], on="user_id", how="left"
    )

    merged_sessions["diff_days"] = (
        merged_sessions["session_start"] - merged_sessions["signup_date"]
    ).dt.days

    retention = merged_sessions.groupby("user_id").agg(
        d7_retained=("diff_days", lambda x: any((x >= 1) & (x <= 7))),
        d14_retained=("diff_days", lambda x: any((x >= 1) & (x <= 14))),
        d30_retained=("diff_days", lambda x: any((x >= 1) & (x <= 30))),
    ).reset_index()

    # -----------------------
    # BUILD FINAL BASE TABLE
    # -----------------------
    base = users.merge(first_open, on="user_id", how="left") \
                .merge(feature_usage, on="user_id", how="left") \
                .merge(workout_freq, on="user_id", how="left") \
                .merge(retention, on="user_id", how="left") \
                .merge(subscriptions[["user_id", "plan_type"]], on="user_id", how="left")

    base["feature_count"] = base["feature_count"].fillna(0)
    base["workouts"] = base["workouts"].fillna(0)

    # Activation bucket
    base["activation_speed"] = base.apply(activation_label, axis=1)

    # Feature depth bucket
    def feature_bucket(n):
        if n >= 3:
            return "3+ features"
        elif n == 2:
            return "2 features"
        elif n == 1:
            return "1 feature"
        else:
            return "0 features"

    base["feature_depth"] = base["feature_count"].apply(feature_bucket)

    # Workout bucket
    def workout_bucket(n):
        if n >= 20:
            return "20+ workouts"
        elif n >= 6:
            return "6–19 workouts"
        elif n >= 1:
            return "1–5 workouts"
        else:
            return "0 workouts"

    base["workout_bucket"] = base["workouts"].apply(workout_bucket)

    return base


# -----------------------------------------------------------
# 4) UI — Upload Zip File
# -----------------------------------------------------------

st.title("📊 Fitness App — RCA Dashboard")
st.markdown("Upload **Dataset_problem.zip** to begin analysis.")

uploaded = st.file_uploader("Upload Dataset ZIP", type=["zip"])

if not uploaded:
    st.stop()

users, app_events, app_sessions, activity_data, subscriptions = load_dataset(uploaded)

computed = preprocess_and_compute(users, app_events, app_sessions, activity_data, subscriptions)

st.success("Dataset loaded successfully!")

# -----------------------------------------------------------
# 5) SHOW SUMMARY STATISTICS
# -----------------------------------------------------------

st.header("📌 Dataset Overview")
col1, col2, col3 = st.columns(3)

col1.metric("Users", len(users))
col2.metric("App Events", len(app_events))
col3.metric("Sessions", len(app_sessions))

# -----------------------------------------------------------
# 6) FUNNEL + SEGMENTATION
# -----------------------------------------------------------

st.header("📊 Segmentation Explorer")

segment_col = st.selectbox(
    "Choose Segment",
    ["age_group", "gender", "plan_type", "activation_speed", "feature_depth", "workout_bucket"]
)

seg = computed.groupby(segment_col).agg(
    signed_up=("user_id", "count"),
    opened_app=("first_open_date", lambda x: x.notna().sum()),
    engaged_core=("feature_count", lambda x: (x > 0).sum()),
    d7=("d7_retained", "sum"),
    d14=("d14_retained", "sum"),
    d30=("d30_retained", "sum"),
).reset_index()

seg["pct_signup_to_open"] = round(100 * seg["opened_app"] / seg["signed_up"], 2)
seg["pct_open_to_engage"] = round(100 * seg["engaged_core"] / seg["opened_app"].replace(0, np.nan), 2)
seg["pct_signup_to_engage"] = round(100 * seg["engaged_core"] / seg["signed_up"], 2)
seg["pct_signup_to_d7"] = round(100 * seg["d7"] / seg["signed_up"], 2)
seg["pct_signup_to_d14"] = round(100 * seg["d14"] / seg["signed_up"], 2)
seg["pct_signup_to_d30"] = round(100 * seg["d30"] / seg["signed_up"], 2)

st.dataframe(seg)

# -----------------------------------------------------------
# 7) CONCLUSION SECTION (Static)
# -----------------------------------------------------------

st.header("📌 RCA Summary — Key Problems Identified")

st.markdown("""
### 🔥 Your Biggest Business Problems
1. **Early activation failure** → 37% users never open the app.
2. **Low feature discovery** → 503 users never explore core features.
3. **Weak habit formation** → low workout frequency = low retention.
4. **Segment gaps** → Older users & “Never Opened” groups churn fastest.
5. **Retention leakage** → Only ~52–55% retained by D7.
6. **Premium dataset quirk** → Everyone is subscription-active; retention still valid.

---

### 🎯 Recommendations (Data-Backed)
1. **Improve activation (Day 0–7)**  
   → personalized nudges, single CTA, guided setup.

2. **Increase feature depth**  
   → tooltips, walkthroughs, in-app banners.

3. **Improve habit formation**  
   → workout reminders, streaks, weekly score summaries.

---

### 📏 Success Metrics to Track
- D1/D7/D30 retention  
- Feature depth % (1→2→3 features)  
- Activation within Day 0/Day 1  
- Weekly workout frequency  
- Session frequency / DAU  
""")

st.success("App Ready — Your Interview Dashboard is Complete!")
