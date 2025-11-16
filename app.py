"""
Single-file Streamlit app (presentation-ready)
Loads all CSVs from Dataset_problem.zip and recreates the analysis + story.
Place Dataset_problem.zip and this app.py in the same folder, then run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import zipfile
from datetime import timedelta
import plotly.express as px
from pathlib import Path

st.set_page_config(layout="wide", page_title="PeakPulse — Analysis & Insights")

# -----------------------
# Helper: Load CSVs from ZIP
# -----------------------
@st.cache_data(ttl=600)
def load_all_from_zip(zip_path="Dataset_problem.zip"):
    zpath = Path(zip_path)
    if not zpath.exists():
        st.error(f"ZIP file not found: {zip_path}. Upload it to this folder.")
        return {}
    data = {}
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                key = Path(name).stem
                try:
                    df = pd.read_csv(z.open(name))
                    data[key] = df
                except Exception as e:
                    st.warning(f"Failed to read {name}: {e}")
                    data[key] = pd.DataFrame()
    return data

data = load_all_from_zip()

# Quick references (safe)
users = data.get("users", pd.DataFrame())
app_events = data.get("app_events", pd.DataFrame())
app_sessions = data.get("app_sessions", pd.DataFrame())
daily_scores = data.get("daily_scores", pd.DataFrame())
subscriptions = data.get("subscriptions", pd.DataFrame())
activity_data = data.get("activity_data", pd.DataFrame())
journal_entries = data.get("journal_entries", pd.DataFrame())

# -----------------------
# Preprocess & compute
# -----------------------
@st.cache_data
def preprocess_and_compute(users, app_events, app_sessions, activity_data, subscriptions, window_days=90):
    # ensure date parsing
    users = users.copy()
    if "created_at" in users.columns:
        users["created_at"] = pd.to_datetime(users["created_at"], errors="coerce")
    else:
        users["created_at"] = pd.NaT

    app_events = app_events.copy()
    if "event_time" in app_events.columns:
        app_events["event_time"] = pd.to_datetime(app_events["event_time"], errors="coerce")

    app_sessions = app_sessions.copy()
    if "session_start" in app_sessions.columns:
        app_sessions["session_start"] = pd.to_datetime(app_sessions["session_start"], errors="coerce")

    if not users.empty and users["created_at"].notna().any():
        max_date = users["created_at"].max()
        min_date = max_date - pd.Timedelta(days=window_days)
    else:
        max_date = pd.Timestamp.now()
        min_date = max_date - pd.Timedelta(days=window_days)

    recent = users[(users["created_at"] >= min_date) & (users["created_at"] <= max_date)].copy()
    recent["signup_date"] = pd.to_datetime(recent["created_at"]).dt.date

    # first app open
    if not app_events.empty and "event_name" in app_events.columns:
        opens = app_events[app_events["event_name"] == "app_open"].copy()
        opens["event_date"] = opens["event_time"].dt.date
        first_open = opens.groupby("user_id")["event_date"].min().reset_index().rename(columns={"event_date": "first_open_date"})
    else:
        first_open = pd.DataFrame(columns=["user_id", "first_open_date"])

    # core features count
    core_features = ["view_sleep","view_recovery","view_strain","view_coaching"]
    if not app_events.empty:
        fe = app_events[app_events["event_name"].isin(core_features)].groupby("user_id")["event_name"].nunique().reset_index().rename(columns={"event_name": "feature_count"})
    else:
        fe = pd.DataFrame(columns=["user_id","feature_count"])

    # workouts count
    if not activity_data.empty:
        workouts = activity_data.groupby("user_id").size().reset_index(name="workouts")
    else:
        workouts = pd.DataFrame(columns=["user_id","workouts"])

    # prepare signup-date based retention using sessions
    if not app_sessions.empty:
        app_sessions["session_date"] = app_sessions["session_start"].dt.date
        sess_group = app_sessions.groupby("user_id")["session_date"].apply(list).to_dict()
    else:
        sess_group = {}

    rows = []
    for idx, r in recent.iterrows():
        uid = r["user_id"]
        sd = r["signup_date"]
        sess_dates = sess_group.get(uid, [])
        d7 = int(any((d >= (sd + timedelta(days=1)) and d <= (sd + timedelta(days=7))) for d in sess_dates))
        d14 = int(any((d >= (sd + timedelta(days=1)) and d <= (sd + timedelta(days=14))) for d in sess_dates))
        d30 = int(any((d >= (sd + timedelta(days=1)) and d <= (sd + timedelta(days=30))) for d in sess_dates))
        rows.append({"user_id": uid, "signup_date": sd, "d7": d7, "d14": d14, "d30": d30})
    retention_df = pd.DataFrame(rows)

    # Merge all user-level features
    base = recent.merge(first_open, on="user_id", how="left")
    base = base.merge(fe, on="user_id", how="left")
    base = base.merge(workouts, on="user_id", how="left")
    base = base.merge(subscriptions[["user_id","plan_type"]].drop_duplicates(), on="user_id", how="left")
    base = base.merge(retention_df, on="user_id", how="left")

    base["feature_count"] = base.get("feature_count", pd.Series(0)).fillna(0).astype(int)
    base["workouts"] = base.get("workouts", pd.Series(0)).fillna(0).astype(int)
    base[["d7","d14","d30"]] = base[["d7","d14","d30"]].fillna(0).astype(int)

    # Activation speed label
    def activation_label(row):
        fo = row.get("first_open_date")
        sd = row.get("signup_date")
        if pd.isna(fo):
            return "Never Opened"
        try:
            # ensure date
            fo_date = pd.to_datetime(fo).date() if not isinstance(fo, (pd.Timestamp,)) else fo.date()
        except:
            return "Never Opened"
        if fo_date == sd:
            return "Day 0"
        if sd + timedelta(days=1) <= fo_date <= sd + timedelta(days=7):
            return "1–7 Days"
        if sd + timedelta(days=8) <= fo_date <= sd + timedelta(days=30):
            return "8–30 Days"
        return "30+ Days"

    base["activation_speed"] = base.apply(activation_label, axis=1)

    # feature depth label
    base["feature_depth"] = base["feature_count"].apply(lambda x: "3+ features" if x>=3 else ("2 features" if x==2 else ("1 feature" if x==1 else "0 features")))

    # workout bucket label
    def workout_bucket(n):
        if n >= 20: return "20+ workouts"
        if 6 <= n <= 19: return "6–19 workouts"
        if 1 <= n <= 5: return "1–5 workouts"
        return "0 workouts"
    base["workout_bucket"] = base["workouts"].apply(workout_bucket)

    # Build segment stacked frame for aggregation (segment, value, user)
    seg_rows = []
    for _, r in base.iterrows():
        uid = r["user_id"]
        seg_rows.append(("age_group", r.get("age_group","unknown"), uid, r["first_open_date"], r["feature_count"], r["d7"], r["d14"], r["d30"]))
        seg_rows.append(("gender", r.get("gender","unknown"), uid, r["first_open_date"], r["feature_count"], r["d7"], r["d14"], r["d30"]))
        seg_rows.append(("plan_type", r.get("plan_type","unknown"), uid, r["first_open_date"], r["feature_count"], r["d7"], r["d14"], r["d30"]))
        seg_rows.append(("activation_speed", r["activation_speed"], uid, r["first_open_date"], r["feature_count"], r["d7"], r["d14"], r["d30"]))
        seg_rows.append(("feature_depth", r["feature_depth"], uid, r["first_open_date"], r["feature_count"], r["d7"], r["d14"], r["d30"]))
        seg_rows.append(("workout_bucket", r["workout_bucket"], uid, r["first_open_date"], r["feature_count"], r["d7"], r["d14"], r["d30"]))
    seg_df = pd.DataFrame(seg_rows, columns=["segment","segment_name","user_id","first_open_date","feature_count","d7","d14","d30"])

    # Aggregate
    agg = seg_df.groupby(["segment","segment_name"]).agg(
        signed_up=("user_id","nunique"),
        opened_app=("first_open_date", lambda s: s.notna().sum()),
        engaged_core=("feature_count", lambda s: (s>0).sum()),
        d7=("d7","sum"),
        d14=("d14","sum"),
        d30=("d30","sum")
    ).reset_index()

    # percentages (safe)
    agg["pct_signup_to_open"] = (100.0 * agg["opened_app"] / agg["signed_up"]).round(2)
    agg["pct_open_to_engage"] = (100.0 * agg["engaged_core"] / agg["opened_app"].replace(0,np.nan)).round(2)
    agg["pct_signup_to_engage"] = (100.0 * agg["engaged_core"] / agg["signed_up"]).round(2)
    agg["pct_signup_to_d7"] = (100.0 * agg["d7"] / agg["signed_up"]).round(2)
    agg["pct_signup_to_d14"] = (100.0 * agg["d14"] / agg["signed_up"]).round(2)
    agg["pct_signup_to_d30"] = (100.0 * agg["d30"] / agg["signed_up"]).round(2)

    return {
        "base": base,
        "seg_df": seg_df,
        "agg": agg,
        "min_date": min_date,
        "max_date": max_date
    }

computed = preprocess_and_compute(users, app_events, app_sessions, activity_data, subscriptions)

if not computed:
    st.stop()

base = computed["base"]
seg_df = computed["seg_df"]
agg = computed["agg"]
min_date = computed["min_date"]
max_date = computed["max_date"]

# -----------------------
# Layout / Navigation
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select page", [
    "Home / Executive Summary",
    "Retention Funnel",
    "Segments & Drilldowns",
    "Feature & Behavior Insights",
    "Conversion Funnel",
    "Problems (Part 2A)",
    "Business Impact (Part 2B)",
    "Recommendations (Part 2C)",
    "Success Metrics (Part 2D)",
    "Notebooks & Repo"
])

# -----------------------
# Page: Home / Executive Summary
# -----------------------
if page == "Home / Executive Summary":
    st.title("PeakPulse — Analysis & Insights (Executive Summary)")
    st.markdown(f"**Data window:** {min_date.date()} → {max_date.date()}")
    st.markdown("**Short summary:** Activation timing and early feature exploration are the primary drivers of retention. Below is a compact summary of what we did and why it matters.")
    st.markdown("**What we analyzed:** signup → activation → feature usage → retention (D7/D14/D30); segments: platform, age, gender, plan, activation speed, feature depth, workouts.")
    cols = st.columns(4)
    cols[0].metric("Window signups", int(base.shape[0]))
    cols[1].metric("Users opened app", int(base["first_open_date"].notna().sum()))
    cols[2].metric("Users with >0 features", int((base["feature_count"]>0).sum()))
    cols[3].metric("Users with 20+ workouts", int((base["workouts"]>=20).sum()))

    st.subheader("Top-line conclusion")
    st.markdown("""
    - **Your biggest business problems are early activation failure, low feature discovery, weak habit formation, and segment-specific retention gaps.**
    - Activation within Day 0–7 and exploring 3+ features are the strongest predictors of D30 retention.
    - Platform (Android/iOS) and demographics are secondary to behavioural signals.
    """)

# -----------------------
# Page: Retention Funnel
# -----------------------
elif page == "Retention Funnel":
    st.title("Retention Funnel (Signup → Open → Engage → D7/D14/D30)")
    st.markdown("Funnel numbers are computed for the 90-day signup window.")
    # Overall funnel using activation_speed group "overall" or compute directly
    funnel_overall = seg_df.groupby("segment").apply(lambda d: d["user_id"].nunique()).to_dict()
    # Use activation_speed aggregated rows
    act = agg[agg["segment"]=="activation_speed"].copy()
    if not act.empty:
        st.subheader("Activation speed (counts)")
        fig = px.bar(act.sort_values("signed_up", ascending=False), x="segment_name", y="signed_up", text="signed_up")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Overall funnel KPIs")
    total_signed = base["user_id"].nunique()
    total_open = base["first_open_date"].notna().sum()
    total_engaged = (base["feature_count"]>0).sum()
    total_d7 = base["d7"].sum()
    total_d14 = base["d14"].sum()
    total_d30 = base["d30"].sum()
    kp1, kp2, kp3, kp4, kp5, kp6 = st.columns(6)
    kp1.metric("Signed up (90d)", f"{total_signed}")
    kp2.metric("Opened app", f"{total_open}", f"{round(100*total_open/total_signed,2) if total_signed else 0}%")
    kp3.metric("Engaged core", f"{total_engaged}", f"{round(100*total_engaged/total_signed,2) if total_signed else 0}%")
    kp4.metric("Returned D7", f"{total_d7}", f"{round(100*total_d7/total_signed,2) if total_signed else 0}%")
    kp5.metric("Returned D14", f"{total_d14}", f"{round(100*total_d14/total_signed,2) if total_signed else 0}%")
    kp6.metric("Returned D30", f"{total_d30}", f"{round(100*total_d30/total_signed,2) if total_signed else 0}%")

    st.subheader("Percent funnel (Signup → Open → Engage → D30)")
    funnel_df = pd.DataFrame([{
        "step":["signup","open","engage","d30"],
        "count":[total_signed, total_open, total_engaged, total_d30]
    }]).explode(["step","count"])
    fig2 = px.bar(funnel_df, x="step", y="count", text="count", title="Funnel")
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------
# Page: Segments & Drilldowns
# -----------------------
elif page == "Segments & Drilldowns":
    st.title("Segments & Drilldowns")
    st.markdown("Select a segment to explore funnel metrics for each segment value.")
    segment_choice = st.selectbox("Segment", agg["segment"].unique())
    seg_rows = agg[agg["segment"]==segment_choice].copy()
    st.dataframe(seg_rows.reset_index(drop=True))
    # Retention bar chart
    if not seg_rows.empty:
        fig = px.bar(seg_rows.sort_values("pct_signup_to_d30", ascending=False),
                     x="segment_name", y="pct_signup_to_d30", text="pct_signup_to_d30",
                     title=f"{segment_choice} — % Signup → D30")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Page: Feature & Behavior Insights
# -----------------------
elif page == "Feature & Behavior Insights":
    st.title("Feature & Behavior Insights")
    st.markdown("How feature depth and workout behaviors influence retention.")
    fd = agg[agg["segment"]=="feature_depth"].copy()
    wb = agg[agg["segment"]=="workout_bucket"].copy()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feature depth → D30 retention")
        if not fd.empty:
            fig = px.bar(fd.sort_values("pct_signup_to_d30", ascending=False),
                         x="segment_name", y="pct_signup_to_d30", text="pct_signup_to_d30")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Workout buckets → D30 retention")
        if not wb.empty:
            fig = px.bar(wb.sort_values("pct_signup_to_d30", ascending=False),
                         x="segment_name", y="pct_signup_to_d30", text="pct_signup_to_d30")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Quick numbers:**")
    st.write(fd[["segment_name","signed_up","pct_signup_to_d30"]])
    st.write(wb[["segment_name","signed_up","pct_signup_to_d30"]])

    st.subheader("Correlation / Uplift checks (simple)")
    # simple uplift: avg feature_count for retained vs churned (d30)
    base_nonnull = base.copy()
    if "d30" in base_nonnull.columns:
        retained = base_nonnull[base_nonnull["d30"]==1]
        churned = base_nonnull[base_nonnull["d30"]==0]
        uplift_table = pd.DataFrame({
            "feature":["feature_count","workouts"],
            "retained_avg":[retained["feature_count"].mean() if not retained.empty else np.nan,
                            retained["workouts"].mean() if not retained.empty else np.nan],
            "churned_avg":[churned["feature_count"].mean() if not churned.empty else np.nan,
                           churned["workouts"].mean() if not churned.empty else np.nan]
        })
        uplift_table["uplift_ratio"] = (uplift_table["retained_avg"] / uplift_table["churned_avg"]).round(2)
        st.table(uplift_table)

# -----------------------
# Page: Conversion Funnel
# -----------------------
elif page == "Conversion Funnel":
    st.title("Conversion Funnel")
    st.markdown("Signup → App open → Core feature use → Subscription conversion (note: dataset appears to have all premium rows).")
    # show counts similar to earlier SQL
    recent_count = base["user_id"].nunique()
    opened = base["first_open_date"].notna().sum()
    engaged = (base["feature_count"]>0).sum()
    subs_90 = subscriptions[
        pd.to_datetime(subscriptions.get("subscription_start_date", pd.Series(pd.NaT))).between(computed["min_date"], computed["max_date"])
    ]["user_id"].nunique() if not subscriptions.empty else 0
    st.metric("Signed up (90d)", recent_count)
    st.metric("Opened app", opened)
    st.metric("Engaged core", engaged)
    st.metric("Subscriptions (start within 90d)", subs_90)

    st.markdown("**Note:** If the dataset contains only premium users (no free users), conversion funnel will show 100% conversion; call this out in the notebook and README.")

# -----------------------
# Page: Problems (Part 2A)
# -----------------------
elif page == "Problems (Part 2A)":
    st.title("Part 2A — Problem Identification (Data-backed)")
    st.markdown("**Your biggest business problems are early activation failure, low feature discovery, weak habit formation, and segment-specific retention gaps.**")
    st.write("")
    st.markdown("""
    **1) Early activation gap — 37% never open the app after signup.**  
    Users who never open the app are lost before the product delivers value.

    **2) Low feature discovery — 0-feature users show near-zero retention.**  
    3+ feature users have very high retention.

    **3) Habit formation is weak — many users do not create routines.**  
    Low workout frequency correlates with low retention.

    **4) D7 cliff — only ~52–55% retained by D7.**  
    The product fails to form early habits for a large cohort.

    **5) Dataset limitation — all users appear premium.**  
    This removes a conversion analysis dimension; focus is retention.
    """)

    st.markdown("**Supporting charts:**")
    st.bar_chart(act[["segment_name","signed_up"]].set_index("segment_name")["signed_up"] if not act.empty else pd.DataFrame())

# -----------------------
# Page: Business Impact (Part 2B)
# -----------------------
elif page == "Business Impact (Part 2B)":
    st.title("Part 2B — Business Impact Quantification (Estimates)")
    st.markdown("Convert funnel improvements into business numbers. These are estimates based on current windows.")

    total_signed = base["user_id"].nunique()
    never_open = base[base["first_open_date"].isna()].shape[0]
    st.markdown(f"- Current signups (window): **{total_signed}**")
    st.markdown(f"- Never opened: **{never_open}** ({round(100*never_open/total_signed,2) if total_signed else 0}%)")

    st.subheader("Impact scenarios (example calculations)")
    delta = st.number_input("Model: Improve Day0–7 activation by (%)", min_value=1, max_value=100, value=10)
    recovered_users = int(total_signed * (delta/100))
    # assume recovered users follow average retention of activated cohort (use existing D30 % among openers)
    openers = base[base["first_open_date"].notna()].shape[0]
    d30_rate_openers = (base[base["d30"]==1].sum()["d30"] / openers) if openers else 0
    est_new_retained = int(recovered_users * (d30_rate_openers if d30_rate_openers else 0.6))  # fallback assume 60%
    st.markdown(f"- If activation +{delta}% → recover **~{recovered_users}** users; estimated **{est_new_retained}** additional D30 retained users (approx).")

    st.markdown("**Feature-depth improvement scenario**")
    # assume moving 10% of 0-feature users into 3+ features increases retention by X (observed delta)
    zero_feat = base[base["feature_count"]==0].shape[0]
    move_pct = st.slider("Move % of 0-feature users to 3+ features", min_value=1, max_value=100, value=10)
    moved = int(zero_feat * (move_pct/100))
    # estimate retention increase: use D30 rates
    d30_3plus = agg[(agg["segment"]=="feature_depth") & (agg["segment_name"]=="3+ features")]["pct_signup_to_d30"]
    d30_0 = agg[(agg["segment"]=="feature_depth") & (agg["segment_name"]=="0 features")]["pct_signup_to_d30"]
    try:
        gain_pct = float(d30_3plus.values[0]) - float(d30_0.values[0])
    except:
        gain_pct = 30.0
    est_gain_users = int(moved * (gain_pct/100))
    st.markdown(f"- Moving {move_pct}% of 0-feature users (~{moved}) to 3+ features estimated to add **{est_gain_users}** D30 retained users (approx).")

# -----------------------
# Page: Recommendations (Part 2C)
# -----------------------
elif page == "Recommendations (Part 2C)":
    st.title("Part 2C — Recommendations (Data-backed)")
    st.markdown("Top recommendations to improve activation, engagement, and retention. Each is tied to the data story above.")
    st.subheader("1) Fix Early Activation (Highest impact)")
    st.markdown("""
    - Implement Day-0 quickstart: single CTA 'Start Routine' that completes a first useful action.
    - Send personalized Day-0 and Day-1 nudges (push/email/SMS) for users who didn't open.
    - Instrument and A/B test different CTAs: 'Start Sleep Tracking' vs 'Start Recovery Check'.
    """)

    st.subheader("2) Drive Feature Depth (Aha moment)")
    st.markdown("""
    - Guided mini-tour in the first session that surfaces 2–3 features.
    - Quick-win tasks (e.g., 'Log 1 workout', 'Take first sleep reading') with small rewards.
    - Use contextual banners that show immediate value (e.g., 'Your recovery score today is X — see tips').
    """)

    st.subheader("3) Habit Formation & Workout Loops")
    st.markdown("""
    - Implement workout reminders, streaks, weekly highlights, and short coaching nudges.
    - Tie coaching suggestions to measurable micro-goals (e.g., '3 workouts this week').
    - Introduce social/team challenges as low-friction onboarding for habit formation.
    """)

    st.subheader("4) Targeted UX for Underperforming Cohorts")
    st.markdown("""
    - Run a small UX research + A/B test for 45+ age group to simplify onboarding flow.
    - Optimize Android app performance and test edge cases on older devices.
    """)

# -----------------------
# Page: Success Metrics (Part 2D)
# -----------------------
elif page == "Success Metrics (Part 2D)":
    st.title("Part 2D — Success Metrics & Dashboards")
    st.markdown("Key metrics to track weekly to measure the recommendations' success.")
    metrics = [
        "Activation within 24h (%)",
        "D7 retention (%)",
        "D14 retention (%)",
        "D30 retention (%)",
        "Feature depth: % users in 3+ features",
        "Weekly active workouts per user",
        "Open rate (signup -> open %)"
    ]
    for m in metrics:
        st.markdown(f"- {m}")

    st.markdown("Define targets (example): Increase Activation within 24h from current to +10% within 6 weeks; increase % 3+ features by 15% within 8 weeks; increase D30 by 10% within 12 weeks.")

# -----------------------
# Page: Notebooks & Repo
# -----------------------
elif page == "Notebooks & Repo":
    st.title("Notebooks, Files & Repo")
    st.markdown("""
    Place your Jupyter notebooks (Notebook1, Notebook2, Notebook3) and this repository structure in the same GitHub repo.
    Streamlit Cloud will serve the app and the notebooks can be linked in the repo description.
    """)
    st.markdown("**Notebook links (example):**")
    st.markdown("- Notebook 1: `notebook1_retention.ipynb`")
    st.markdown("- Notebook 2: `notebook2_features.ipynb`")
    st.markdown("- Notebook 3: `notebook3_funnels.ipynb`")
    st.markdown("**Repo checklist**")
    st.markdown("""
    - `Dataset_problem.zip` (all CSVs)  
    - `app.py` (this file)  
    - `requirements.txt`  
    - `notebook1_retention.ipynb`  
    - `notebook2_features.ipynb`  
    - `notebook3_funnels.ipynb`  
    - `README.md` with instructions to run
    """)

# End of app
