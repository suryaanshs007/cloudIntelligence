import streamlit as st
import boto3
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="CloudSentinel", page_icon="⚡", layout="wide")

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #080c14; color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; }
.sentinel-title { font-family: 'Space Mono', monospace; font-size: 2.2rem; font-weight: 700; color: #ffffff; letter-spacing: -1px; }
.sentinel-subtitle { font-size: 0.95rem; color: #64748b; margin-bottom: 2rem; font-weight: 300; }
.badge         { display: inline-block; background: #0f1f0f; border: 1px solid #16a34a; color: #4ade80; font-family: 'Space Mono', monospace; font-size: 0.65rem; padding: 3px 10px; border-radius: 20px; margin-left: 12px; vertical-align: middle; letter-spacing: 1px; }
.badge-if      { display: inline-block; background: #1a0f2e; border: 1px solid #7c3aed; color: #c4b5fd; font-family: 'Space Mono', monospace; font-size: 0.65rem; padding: 3px 10px; border-radius: 20px; margin-left: 8px; vertical-align: middle; letter-spacing: 1px; }
.badge-lr      { display: inline-block; background: #0f1a2e; border: 1px solid #3b82f6; color: #93c5fd; font-family: 'Space Mono', monospace; font-size: 0.65rem; padding: 3px 10px; border-radius: 20px; margin-left: 8px; vertical-align: middle; letter-spacing: 1px; }
.metric-card { background: #0d1117; border: 1px solid #1e293b; border-radius: 12px; padding: 1.4rem 1.6rem; position: relative; overflow: hidden; }
.metric-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #3b82f6, #06b6d4); }
.metric-card.danger::before  { background: linear-gradient(90deg, #ef4444, #f97316); }
.metric-card.success::before { background: linear-gradient(90deg, #22c55e, #16a34a); }
.metric-card.warning::before { background: linear-gradient(90deg, #f59e0b, #eab308); }
.metric-card.purple::before  { background: linear-gradient(90deg, #7c3aed, #a855f7); }
.metric-card.violet::before  { background: linear-gradient(90deg, #a855f7, #ec4899); }
.metric-card.cyan::before    { background: linear-gradient(90deg, #06b6d4, #0ea5e9); }
.metric-card.teal::before    { background: linear-gradient(90deg, #14b8a6, #06b6d4); }
.metric-card.rose::before    { background: linear-gradient(90deg, #f43f5e, #e11d48); }
.metric-card.amber::before   { background: linear-gradient(90deg, #f59e0b, #f97316); }
.metric-label { font-size: 0.72rem; color: #475569; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600; margin-bottom: 0.5rem; font-family: 'Space Mono', monospace; }
.metric-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: #f1f5f9; line-height: 1; }
.metric-sub   { font-size: 0.78rem; color: #475569; margin-top: 0.4rem; }
.section-header         { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #3b82f6; text-transform: uppercase; letter-spacing: 2px; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #1e293b; }
.section-header-if      { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #a855f7; text-transform: uppercase; letter-spacing: 2px; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #2d1f3d; }
.section-header-compare { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #06b6d4; text-transform: uppercase; letter-spacing: 2px; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #1e293b; }
.section-header-cat     { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #f59e0b; text-transform: uppercase; letter-spacing: 2px; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #2a1f0a; }
.section-header-perf    { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #14b8a6; text-transform: uppercase; letter-spacing: 2px; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #0a2a27; }
.instance-card    { background: #0d1117; border: 1px solid #1e293b; border-radius: 12px; padding: 1.4rem 1.6rem; }
.instance-running { color: #4ade80; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.1rem; }
.instance-stopped { color: #f87171; font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.1rem; }
.remediation-low    { background: #0f1a0f; border-left: 3px solid #22c55e; padding: 10px 14px; margin: 5px 0; border-radius: 0 6px 6px 0; font-size: 0.82rem; color: #86efac; }
.remediation-medium { background: #1a140a; border-left: 3px solid #f59e0b; padding: 10px 14px; margin: 5px 0; border-radius: 0 6px 6px 0; font-size: 0.82rem; color: #fcd34d; }
.remediation-high   { background: #1a0a0a; border-left: 3px solid #ef4444; padding: 10px 14px; margin: 5px 0; border-radius: 0 6px 6px 0; font-size: 0.82rem; color: #fca5a5; }
.info-box    { background: #0d1117; border: 1px solid #1e293b; border-radius: 10px; padding: 1.2rem 1.4rem; font-size: 0.85rem; color: #94a3b8; line-height: 1.7; }
.info-box-if { background: #0d0d17; border: 1px solid #2d1f3d; border-radius: 10px; padding: 1.2rem 1.4rem; font-size: 0.85rem; color: #c4b5fd; line-height: 1.7; }
.model-chip-lr { display: inline-block; background: #0f1a2e; border: 1px solid #3b82f6; color: #93c5fd; font-family: 'Space Mono', monospace; font-size: 0.7rem; padding: 4px 12px; border-radius: 20px; margin-right: 8px; }
.model-chip-if { display: inline-block; background: #1a0f2e; border: 1px solid #7c3aed; color: #c4b5fd; font-family: 'Space Mono', monospace; font-size: 0.7rem; padding: 4px 12px; border-radius: 20px; margin-right: 8px; }
.divider       { border: none; border-top: 1px solid #1e293b; margin: 2rem 0; }
.agree-badge    { display: inline-block; background: #0f1a0f; border: 1px solid #22c55e; color: #4ade80; font-family: 'Space Mono', monospace; font-size: 0.65rem; padding: 2px 8px; border-radius: 10px; }
.disagree-badge { display: inline-block; background: #1a0a0a; border: 1px solid #ef4444; color: #f87171; font-family: 'Space Mono', monospace; font-size: 0.65rem; padding: 2px 8px; border-radius: 10px; }
.cat-cpu      { display:inline-block; background:#1a0a0a; border:1px solid #ef4444; color:#fca5a5; font-family:'Space Mono',monospace; font-size:0.65rem; padding:3px 10px; border-radius:20px; margin:2px 4px; }
.cat-network  { display:inline-block; background:#0a1020; border:1px solid #3b82f6; color:#93c5fd; font-family:'Space Mono',monospace; font-size:0.65rem; padding:3px 10px; border-radius:20px; margin:2px 4px; }
.cat-combined { display:inline-block; background:#1a0f00; border:1px solid #f97316; color:#fdba74; font-family:'Space Mono',monospace; font-size:0.65rem; padding:3px 10px; border-radius:20px; margin:2px 4px; }
.cat-idle     { display:inline-block; background:#0f1a14; border:1px solid #22c55e; color:#86efac; font-family:'Space Mono',monospace; font-size:0.65rem; padding:3px 10px; border-radius:20px; margin:2px 4px; }
.cat-asym     { display:inline-block; background:#1a0f2e; border:1px solid #a855f7; color:#d8b4fe; font-family:'Space Mono',monospace; font-size:0.65rem; padding:3px 10px; border-radius:20px; margin:2px 4px; }
.cat-row { background:#0d1117; border:1px solid #1e293b; border-radius:10px; padding:10px 14px; margin:5px 0; font-size:0.82rem; line-height:1.8; }
.timing-card { background:#0a1a17; border:1px solid #14b8a6; border-radius:12px; padding:1.2rem 1.6rem; position:relative; overflow:hidden; }
.timing-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,#14b8a6,#06b6d4); }
.timing-fast   { color:#4ade80; font-family:'Space Mono',monospace; font-weight:700; font-size:1.6rem; }
.timing-medium { color:#fcd34d; font-family:'Space Mono',monospace; font-weight:700; font-size:1.6rem; }
.timing-slow   { color:#f87171; font-family:'Space Mono',monospace; font-weight:700; font-size:1.6rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
REGION      = "us-east-2"
INSTANCE_ID = "i-0372640781068e64b"
ENABLE_STOP = False  # set True only for live demo

ec2_client        = boto3.client("ec2",        region_name=REGION)
cloudwatch_client = boto3.client("cloudwatch", region_name=REGION)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="sentinel-title">⚡ CloudSentinel <span class="badge">LIVE</span></div>
<div class="sentinel-subtitle">
    Autonomous Cloud Cost Intelligence · AWS EC2 ·
    <span class="model-chip-lr">Logistic Regression</span>
    <span class="model-chip-if">Isolation Forest</span>
    · Auto-Remediation
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# ANOMALY CATEGORY DEFINITIONS
# ─────────────────────────────────────────
CAT_CPU_SPIKE  = "CPU Spike"
CAT_NET_SPIKE  = "Network Spike"
CAT_COMBINED   = "CPU+Network Combined"
CAT_IDLE_WASTE = "Idle Waste"
CAT_NET_ASYM   = "Network Asymmetry"

CAT_META = {
    CAT_CPU_SPIKE : {"cls": "cat-cpu",      "icon": "🔴", "desc": "Sudden CPU surge above 2.5σ from baseline"},
    CAT_NET_SPIKE : {"cls": "cat-network",  "icon": "🔵", "desc": "Sudden surge in NetworkIn or NetworkOut"},
    CAT_COMBINED  : {"cls": "cat-combined", "icon": "🟠", "desc": "CPU and Network both elevated — possible crypto-mining or exfiltration"},
    CAT_IDLE_WASTE: {"cls": "cat-idle",     "icon": "🟢", "desc": "CPU near 0% for sustained periods — instance idle, cost wasted"},
    CAT_NET_ASYM  : {"cls": "cat-asym",     "icon": "🟣", "desc": "High NetworkOut vs low NetworkIn — potential data leak pattern"},
}

def categorise_anomalies(df):
    df["net_in_z"]  = np.abs(stats.zscore(df["network_in_mb"].fillna(0)  + 1e-9))
    df["net_out_z"] = np.abs(stats.zscore(df["network_out_mb"].fillna(0) + 1e-9))

    df["is_idle_point"] = df["cpu_usage_pct"] < 2.0
    df["idle_streak"]   = (
        df["is_idle_point"]
          .groupby((df["is_idle_point"] != df["is_idle_point"].shift()).cumsum())
          .transform("cumsum")
    )
    df["idle_anomaly"] = (df["idle_streak"] >= 3) & df["is_idle_point"]

    cats = []
    for _, row in df.iterrows():
        row_cats = []
        cpu_z       = row["z_score"]
        net_in_z    = row["net_in_z"]
        net_out_z   = row["net_out_z"]
        net_in_raw  = max(row["network_in_mb"],  1e-9)
        net_out_raw = max(row["network_out_mb"], 1e-9)

        if cpu_z >= 2.5:
            row_cats.append(CAT_CPU_SPIKE)
        if net_in_z >= 2.5 or net_out_z >= 2.5:
            row_cats.append(CAT_NET_SPIKE)
        if cpu_z >= 2.5 and (net_in_z >= 2.5 or net_out_z >= 2.5):
            row_cats.append(CAT_COMBINED)
        if row.get("idle_anomaly", False):
            row_cats.append(CAT_IDLE_WASTE)
        if (net_out_raw / net_in_raw) >= 5.0 and net_out_z >= 1.5:
            row_cats.append(CAT_NET_ASYM)

        cats.append(row_cats)

    df["anomaly_categories"] = cats
    df["has_any_anomaly"]    = df["anomaly_categories"].apply(lambda x: len(x) > 0)
    return df

# ─────────────────────────────────────────
# REMEDIATION ENGINE
# ─────────────────────────────────────────
def classify_severity(z):
    if z >= 6.0:   return "HIGH"
    elif z >= 4.0: return "MEDIUM"
    else:          return "LOW"

def tag_instance(z, severity, ts):
    try:
        ec2_client.create_tags(Resources=[INSTANCE_ID], Tags=[
            {"Key": "AnomalyDetected",  "Value": "true"},
            {"Key": "AnomalySeverity",  "Value": severity},
            {"Key": "AnomalyZScore",    "Value": str(round(z, 4))},
            {"Key": "AnomalyTimestamp", "Value": ts},
            {"Key": "RemediatedBy",     "Value": "CloudSentinel"},
        ])
        return True, "Tagged instance in AWS console"
    except Exception as e:
        return False, str(e)

def create_alarm(z):
    try:
        threshold = 30.0 if z >= 6.0 else 50.0
        cloudwatch_client.put_metric_alarm(
            AlarmName="CloudSentinel-CPU-Anomaly",
            AlarmDescription=f"Auto-created by CloudSentinel. Z={z:.3f}",
            ActionsEnabled=False,
            MetricName="CPUUtilization", Namespace="AWS/EC2", Statistic="Average",
            Dimensions=[{"Name": "InstanceId", "Value": INSTANCE_ID}],
            Period=300, EvaluationPeriods=1, Threshold=threshold,
            ComparisonOperator="GreaterThanThreshold", TreatMissingData="notBreaching"
        )
        return True, f"CloudWatch alarm set — threshold {threshold}%"
    except Exception as e:
        return False, str(e)

def stop_instance():
    if not ENABLE_STOP:
        return False, "Stop skipped (safe mode)"
    try:
        ec2_client.stop_instances(InstanceIds=[INSTANCE_ID])
        return True, "Stop command issued"
    except Exception as e:
        return False, str(e)

def remediate(z, cpu, ts):
    severity = classify_severity(z)
    actions  = []
    if severity in ["LOW", "MEDIUM", "HIGH"]:
        ok, msg = tag_instance(z, severity, ts)
        actions.append({"action": "Tag Instance",    "ok": ok, "msg": msg})
    if severity in ["MEDIUM", "HIGH"]:
        ok, msg = create_alarm(z)
        actions.append({"action": "Create CW Alarm", "ok": ok, "msg": msg})
    if severity == "HIGH":
        ok, msg = stop_instance()
        actions.append({"action": "Stop Instance",   "ok": ok, "msg": msg})
    return severity, actions

# ─────────────────────────────────────────
# INSTANCE STATUS
# ─────────────────────────────────────────
def get_instance_status():
    try:
        r    = ec2_client.describe_instances(InstanceIds=[INSTANCE_ID])
        inst = r["Reservations"][0]["Instances"][0]
        tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
        return {
            "state"        : inst["State"]["Name"],
            "type"         : inst["InstanceType"],
            "az"           : inst["Placement"]["AvailabilityZone"],
            "anomaly"      : tags.get("AnomalyDetected", "false"),
            "severity"     : tags.get("AnomalySeverity", "—"),
            "z_score"      : tags.get("AnomalyZScore",   "—"),
            "remediated_by": tags.get("RemediatedBy",     "—"),
            "remediated_at": tags.get("AnomalyTimestamp", "—"),
        }
    except Exception as e:
        return {"state": "unknown", "error": str(e)}

# ─────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────
@st.cache_data(ttl=300)
def run_pipeline():
    def get_metric(name):
        r = cloudwatch_client.get_metric_statistics(
            Namespace="AWS/EC2", MetricName=name,
            Dimensions=[{"Name": "InstanceId", "Value": INSTANCE_ID}],
            StartTime=datetime.now(timezone.utc) - timedelta(hours=24),
            EndTime=datetime.now(timezone.utc),
            Period=300, Statistics=["Average"]
        )
        return r.get("Datapoints", [])

    def to_df(data, col):
        if not data: return pd.DataFrame(columns=["Timestamp", col])
        d = pd.DataFrame(data)
        if "Timestamp" not in d.columns: return pd.DataFrame(columns=["Timestamp", col])
        return d.sort_values("Timestamp")[["Timestamp","Average"]].rename(columns={"Average": col})

    df = pd.merge(to_df(get_metric("CPUUtilization"), "cpu_usage_pct"),
                  to_df(get_metric("NetworkIn"),      "network_in_mb"),  on="Timestamp", how="outer")
    df = pd.merge(df, to_df(get_metric("NetworkOut"), "network_out_mb"), on="Timestamp", how="outer")
    df = df.sort_values("Timestamp").reset_index(drop=True).ffill().bfill()

    for col in ["network_in_mb", "network_out_mb"]:
        if col in df.columns:
            df[col] /= (1024 * 1024)

    if len(df) < 10:
        return (None,) * 13

    df["z_score"]    = np.abs(stats.zscore(df["cpu_usage_pct"]))
    df["is_anomaly"] = (df["z_score"] > 2.5).astype(int)

    features = ["cpu_usage_pct", "network_in_mb", "network_out_mb"]
    X = df[features].values
    y = df["is_anomaly"].values

    df = categorise_anomalies(df)

    # ── Logistic Regression (optimised) ──
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )
    sc     = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    t0_lr = time.perf_counter()
    lr_model = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="lbfgs",
        tol=1e-3,
    )
    lr_model.fit(X_tr_s, y_tr)
    y_pred_lr = lr_model.predict(X_te_s)
    lr_time   = (time.perf_counter() - t0_lr) * 1000

    lr_acc   = accuracy_score(y_te, y_pred_lr)
    lr_cm    = confusion_matrix(y_te, y_pred_lr, labels=[0, 1])
    lr_coefs = lr_model.coef_[0]

    X_full_s         = sc.transform(X)
    df["lr_anomaly"] = lr_model.predict(X_full_s)
    df["lr_proba"]   = lr_model.predict_proba(X_full_s)[:, 1]

    # ── Isolation Forest (optimised) ──
    sc_if         = StandardScaler()
    X_scaled      = sc_if.fit_transform(X)
    contamination = max(0.01, min(0.5, float(y.mean()) if y.mean() > 0 else 0.05))

    t0_if = time.perf_counter()
    if_model = IsolationForest(
        n_estimators=50,
        contamination=contamination,
        max_samples="auto",
        bootstrap=False,
        n_jobs=-1,
        random_state=42,
    )
    if_model.fit(X_scaled)
    if_preds  = if_model.predict(X_scaled)
    if_scores = if_model.decision_function(X_scaled)
    if_time   = (time.perf_counter() - t0_if) * 1000

    df["if_anomaly"] = (if_preds == -1).astype(int)
    df["if_score"]   = if_scores

    if_acc = accuracy_score(y, df["if_anomaly"].values)
    if_cm  = confusion_matrix(y, df["if_anomaly"].values, labels=[0, 1])

    df["model_agree"] = (df["lr_anomaly"] == df["if_anomaly"]).astype(int)

    return (
        df,
        lr_acc, lr_cm, lr_coefs, features, len(X_tr), len(X_te),
        if_acc, if_cm, contamination,
        lr_time, if_time
    )

# ─────────────────────────────────────────
# REFRESH
# ─────────────────────────────────────────
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Fetching live AWS telemetry..."):
    result = run_pipeline()

if result[0] is None:
    st.error("Not enough data. Wait a few minutes then refresh.")
    st.stop()

(df,
 lr_acc, lr_cm, lr_coefs, features, n_train, n_test,
 if_acc, if_cm, contamination,
 lr_time, if_time) = result

# ─────────────────────────────────────────
# SYSTEM OVERVIEW
# ─────────────────────────────────────────
st.markdown('<div class="section-header">System Overview</div>', unsafe_allow_html=True)

total      = len(df)
n_anom_lr  = int(df["lr_anomaly"].sum())
n_anom_if  = int(df["if_anomaly"].sum())
latest_cpu = df["cpu_usage_pct"].iloc[-1]
max_z      = df["z_score"].max()

cat_counts = {cat: 0 for cat in CAT_META}
for cats in df["anomaly_categories"]:
    for c in cats:
        if c in cat_counts:
            cat_counts[c] += 1

c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    (c1, "",        "Total Datapoints", str(total),             "Last 24h · 5min intervals"),
    (c2, "warning", "Peak Z-Score",     f"{max_z:.2f}σ",        "Threshold 2.5σ"),
    (c3, "danger" if latest_cpu > 20 else "success", "Current CPU", f"{latest_cpu:.2f}%",
         df["Timestamp"].max().strftime("%H:%M UTC")),
    (c4, "danger" if n_anom_lr > 0 else "success", "LR Anomalies", str(n_anom_lr),
         f"Accuracy {lr_acc*100:.1f}%"),
    (c5, "purple" if n_anom_if > 0 else "success",  "IF Anomalies", str(n_anom_if),
         f"Accuracy {if_acc*100:.1f}%"),
]
for col, cls, label, val, sub in cards:
    with col:
        st.markdown(f"""<div class="metric-card {cls}">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# MODEL EXECUTION TIMING
# ─────────────────────────────────────────
st.markdown('<div class="section-header-perf">Model Execution Performance · Per Batch (300s interval)</div>', unsafe_allow_html=True)

def timing_cls(ms):
    return "timing-fast" if ms < 50 else ("timing-medium" if ms < 200 else "timing-slow")

def timing_label(ms):
    return "FAST" if ms < 50 else ("MODERATE" if ms < 200 else "SLOW")

total_time  = lr_time + if_time
budget_pct  = (total_time / (300 * 1000)) * 100

t1, t2, t3, t4 = st.columns(4)
with t1:
    st.markdown(f"""<div class="timing-card">
        <div class="metric-label">LR Training + Inference</div>
        <div class="{timing_cls(lr_time)}">{lr_time:.2f} ms</div>
        <div class="metric-sub">{timing_label(lr_time)} · max_iter=300 · lbfgs</div>
    </div>""", unsafe_allow_html=True)
with t2:
    st.markdown(f"""<div class="timing-card">
        <div class="metric-label">IF Training + Inference</div>
        <div class="{timing_cls(if_time)}">{if_time:.2f} ms</div>
        <div class="metric-sub">{timing_label(if_time)} · n_estimators=50 · n_jobs=-1</div>
    </div>""", unsafe_allow_html=True)
with t3:
    st.markdown(f"""<div class="timing-card">
        <div class="metric-label">Total Pipeline Time</div>
        <div class="{timing_cls(total_time)}">{total_time:.2f} ms</div>
        <div class="metric-sub">LR + IF combined</div>
    </div>""", unsafe_allow_html=True)
with t4:
    st.markdown(f"""<div class="timing-card">
        <div class="metric-label">% of 300s Budget Used</div>
        <div class="timing-fast">{budget_pct:.4f}%</div>
        <div class="metric-sub">{total_time:.2f}ms of 300,000ms window</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(8, 2))
fig.patch.set_facecolor("#0a1a17"); ax.set_facecolor("#0a1a17")
bars = ax.barh(["Isolation Forest", "Logistic Regression"], [if_time, lr_time],
               color=["#a855f7", "#3b82f6"], alpha=0.85, height=0.4)
for bar, val in zip(bars, [if_time, lr_time]):
    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} ms", va="center", color="#94a3b8", fontsize=9)
ax.set_xlabel("Execution Time (ms)", color="#14b8a6", fontsize=8)
ax.set_title("Model Execution Time Comparison", color="#14b8a6", fontsize=10, loc="left", pad=8)
ax.tick_params(colors="#475569", labelsize=9); ax.spines[:].set_color("#0a2a27")
ax.grid(color="#0a2a27", linewidth=0.5, alpha=0.5, axis="x")
fig.tight_layout(); st.pyplot(fig); plt.close()

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# INSTANCE STATUS
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Instance Status</div>', unsafe_allow_html=True)
status = get_instance_status()
is1, is2, is3 = st.columns(3)
state_cls  = "instance-running" if status.get("state") == "running" else "instance-stopped"
state_icon = "🟢" if status.get("state") == "running" else "🔴"

with is1:
    st.markdown(f"""<div class="instance-card">
        <div class="metric-label">Instance State</div>
        <div class="{state_cls}">{state_icon} {status.get('state','—').upper()}</div>
        <div class="metric-sub">{INSTANCE_ID}</div>
        <div class="metric-sub">Type: {status.get('type','—')} · AZ: {status.get('az','—')}</div>
    </div>""", unsafe_allow_html=True)
with is2:
    flagged   = status.get("anomaly") == "true"
    sev_color = {"HIGH":"#f87171","MEDIUM":"#fcd34d","LOW":"#86efac"}.get(status.get("severity",""),"#64748b")
    st.markdown(f"""<div class="instance-card">
        <div class="metric-label">Anomaly Tag</div>
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;color:{sev_color}">
            {'🚨 FLAGGED' if flagged else '✅ CLEAN'}
        </div>
        <div class="metric-sub">Severity: {status.get('severity','—')}</div>
        <div class="metric-sub">Z-Score recorded: {status.get('z_score','—')}σ</div>
    </div>""", unsafe_allow_html=True)
with is3:
    st.markdown(f"""<div class="instance-card">
        <div class="metric-label">Last Remediation</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.85rem;color:#94a3b8;font-weight:600">
            {status.get('remediated_by','—')}
        </div>
        <div class="metric-sub">{status.get('remediated_at','—')}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# ANOMALY CATEGORIES
# ═══════════════════════════════════════════
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown('<div class="section-header-cat">Anomaly Categories · Multi-Metric Detection</div>', unsafe_allow_html=True)

ac1, ac2, ac3, ac4, ac5 = st.columns(5)
for col, cat, cls in [
    (ac1, CAT_CPU_SPIKE,  "danger"),
    (ac2, CAT_NET_SPIKE,  "cyan"),
    (ac3, CAT_COMBINED,   "amber"),
    (ac4, CAT_IDLE_WASTE, "success"),
    (ac5, CAT_NET_ASYM,   "purple"),
]:
    with col:
        st.markdown(f"""<div class="metric-card {cls}">
            <div class="metric-label">{CAT_META[cat]['icon']} {cat}</div>
            <div class="metric-value">{cat_counts.get(cat, 0)}</div>
            <div class="metric-sub">{CAT_META[cat]['desc'][:42]}...</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Multi-metric timeline
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
fig.patch.set_facecolor("#0d1117")

for ax, col, color, title, cat1, cat2 in [
    (axes[0], "cpu_usage_pct",  "#ef4444", "CPU Utilization (%)", CAT_CPU_SPIKE,  CAT_COMBINED),
    (axes[1], "network_in_mb",  "#3b82f6", "Network In (MB)",     CAT_NET_SPIKE,  CAT_NET_ASYM),
    (axes[2], "network_out_mb", "#a855f7", "Network Out (MB)",    CAT_NET_SPIKE,  CAT_IDLE_WASTE),
]:
    ax.set_facecolor("#0d1117")
    ax.plot(df.index, df[col], color=color, linewidth=1.1, alpha=0.7)
    ax.fill_between(df.index, df[col], alpha=0.08, color=color)

    mask1 = df["anomaly_categories"].apply(lambda x: cat1 in x)
    if mask1.any():
        ax.scatter(df.index[mask1], df[col][mask1],
                   color="#f97316", s=50, zorder=5, marker="^", label=cat1, alpha=0.9)
    mask2 = df["anomaly_categories"].apply(lambda x: cat2 in x)
    if mask2.any():
        ax.scatter(df.index[mask2], df[col][mask2],
                   color="#a855f7", s=50, zorder=5, marker="D", label=cat2, alpha=0.9)

    ax.set_title(title, color="#94a3b8", fontsize=9, loc="left", pad=4)
    ax.tick_params(colors="#475569", labelsize=7)
    ax.spines[:].set_color("#2a1f0a")
    ax.grid(color="#2a1f0a", linewidth=0.4, alpha=0.5)
    ax.legend(facecolor="#0d1117", edgecolor="#2a1f0a", labelcolor="#94a3b8", fontsize=7, loc="upper right")

fig.suptitle("Multi-Metric Anomaly Category Timeline", color="#f59e0b", fontsize=11, y=1.01)
fig.tight_layout(); st.pyplot(fig); plt.close()

st.markdown("<br>", unsafe_allow_html=True)

# Category legend
legend_html = '<div class="info-box"><b style="color:#e2e8f0">Anomaly Category Definitions</b><br><br>'
for cat, meta in CAT_META.items():
    legend_html += f'{meta["icon"]} <span class="{meta["cls"]}">{cat}</span> &nbsp; {meta["desc"]}<br>'
legend_html += "</div>"
st.markdown(legend_html, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Per-row breakdown
flagged_rows = df[df["anomaly_categories"].apply(lambda x: len(x) > 0)].copy()
if len(flagged_rows) > 0:
    st.markdown(f"**{len(flagged_rows)} datapoints carry at least one anomaly category:**")
    for _, row in flagged_rows.iterrows():
        ts = row["Timestamp"].strftime("%Y-%m-%d %H:%M UTC") \
             if hasattr(row["Timestamp"], "strftime") else str(row["Timestamp"])
        chips = " ".join([
            f'<span class="{CAT_META[c]["cls"]}">{CAT_META[c]["icon"]} {c}</span>'
            for c in row["anomaly_categories"]
        ])
        st.markdown(f"""<div class="cat-row">
            <b style="color:#e2e8f0">{ts}</b> &nbsp;|&nbsp;
            CPU: <b>{row['cpu_usage_pct']:.2f}%</b> &nbsp;|&nbsp;
            Net↓ <b>{row['network_in_mb']:.4f} MB</b> &nbsp;|&nbsp;
            Net↑ <b>{row['network_out_mb']:.4f} MB</b><br>{chips}
        </div>""", unsafe_allow_html=True)
else:
    st.markdown('<div class="info-box">✅ No multi-metric anomaly categories detected in current window.</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────
# LR TELEMETRY
# ─────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown('<div class="section-header">Live Telemetry <span class="badge-lr">Logistic Regression</span></div>', unsafe_allow_html=True)

lr_mask = df["lr_anomaly"] == 1
ch1, ch2 = st.columns(2)
for col, y_data, title, color, show_thresh in [
    (ch1, df["cpu_usage_pct"], "CPU Utilization (%)",    "#3b82f6", False),
    (ch2, df["lr_proba"],      "LR Anomaly Probability", "#3b82f6", True),
]:
    with col:
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
        ax.plot(df.index, y_data, color=color, linewidth=1.2, alpha=0.8)
        ax.fill_between(df.index, y_data, alpha=0.1, color=color)
        ax.scatter(df.index[lr_mask], y_data[lr_mask], color="#ef4444", s=60, zorder=5, label="LR Anomaly")
        if show_thresh:
            ax.axhline(y=0.5, color="#ef4444", linewidth=1, linestyle="--", label="Decision Boundary (0.5)")
        ax.set_title(title, color="#94a3b8", fontsize=10, pad=10, loc="left")
        ax.tick_params(colors="#475569", labelsize=8); ax.spines[:].set_color("#1e293b")
        ax.legend(facecolor="#0d1117", edgecolor="#1e293b", labelcolor="#94a3b8", fontsize=8)
        ax.grid(color="#1e293b", linewidth=0.5, alpha=0.5)
        fig.tight_layout(); st.pyplot(fig); plt.close()

# LR Performance
st.markdown('<div class="section-header">Model Performance <span class="badge-lr">Logistic Regression</span></div>', unsafe_allow_html=True)
ch3, ch4 = st.columns(2)
with ch3:
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    ax.imshow(lr_cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Anomaly"], color="#94a3b8", fontsize=9)
    ax.set_yticklabels(["Normal","Anomaly"], color="#94a3b8", fontsize=9)
    ax.set_xlabel("Predicted", color="#64748b", fontsize=9)
    ax.set_ylabel("Actual",    color="#64748b", fontsize=9)
    ax.set_title("LR Confusion Matrix", color="#94a3b8", fontsize=10, pad=10, loc="left")
    ax.spines[:].set_color("#1e293b")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(lr_cm[i,j]), ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    fig.tight_layout(); st.pyplot(fig); plt.close()
with ch4:
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    c_bars = ["#3b82f6" if c >= 0 else "#ef4444" for c in lr_coefs]
    bars = ax.barh(features, lr_coefs, color=c_bars, alpha=0.85)
    ax.set_title("LR Feature Importance (Coefficients)", color="#94a3b8", fontsize=10, pad=10, loc="left")
    ax.tick_params(colors="#64748b", labelsize=9); ax.spines[:].set_color("#1e293b")
    ax.set_xlabel("Coefficient Value", color="#64748b", fontsize=8)
    ax.grid(color="#1e293b", linewidth=0.5, alpha=0.5, axis="x")
    ax.axvline(x=0, color="#475569", linewidth=0.8)
    for bar, val in zip(bars, lr_coefs):
        ax.text(val+(0.01 if val>=0 else -0.01), bar.get_y()+bar.get_height()/2,
                f"{val:.4f}", va="center", ha="left" if val>=0 else "right", color="#94a3b8", fontsize=8)
    fig.tight_layout(); st.pyplot(fig); plt.close()

# ═══════════════════════════════════════════
# ISOLATION FOREST
# ═══════════════════════════════════════════
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown('<div class="section-header-if">Isolation Forest Model <span class="badge-if">UNSUPERVISED</span></div>', unsafe_allow_html=True)

n_agree_total = int(df["model_agree"].sum())
agree_pct     = (n_agree_total / total * 100) if total > 0 else 0

ifc1, ifc2, ifc3, ifc4 = st.columns(4)
with ifc1:
    st.markdown(f"""<div class="metric-card purple">
        <div class="metric-label">IF Accuracy</div>
        <div class="metric-value">{if_acc*100:.1f}%</div>
        <div class="metric-sub">vs Z-score ground truth</div>
    </div>""", unsafe_allow_html=True)
with ifc2:
    st.markdown(f"""<div class="metric-card purple">
        <div class="metric-label">IF Anomalies Found</div>
        <div class="metric-value">{n_anom_if}</div>
        <div class="metric-sub">Contamination: {contamination:.2f}</div>
    </div>""", unsafe_allow_html=True)
with ifc3:
    st.markdown(f"""<div class="metric-card violet">
        <div class="metric-label">Min Anomaly Score</div>
        <div class="metric-value">{df['if_score'].min():.3f}</div>
        <div class="metric-sub">Avg: {df['if_score'].mean():.3f}</div>
    </div>""", unsafe_allow_html=True)
with ifc4:
    st.markdown(f"""<div class="metric-card {'success' if agree_pct >= 80 else 'warning'}">
        <div class="metric-label">Model Agreement</div>
        <div class="metric-value">{agree_pct:.1f}%</div>
        <div class="metric-sub">LR vs IF consensus</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header-if">IF Telemetry</div>', unsafe_allow_html=True)

if_mask = df["if_anomaly"] == 1
ich1, ich2 = st.columns(2)
with ich1:
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    ax.plot(df.index, df["cpu_usage_pct"], color="#a855f7", linewidth=1.2, alpha=0.8)
    ax.fill_between(df.index, df["cpu_usage_pct"], alpha=0.1, color="#a855f7")
    ax.scatter(df.index[if_mask], df["cpu_usage_pct"][if_mask],
               color="#f97316", s=60, zorder=5, label="IF Anomaly", marker="D")
    ax.set_title("CPU Utilization — IF Detections", color="#94a3b8", fontsize=10, pad=10, loc="left")
    ax.tick_params(colors="#475569", labelsize=8); ax.spines[:].set_color("#2d1f3d")
    ax.legend(facecolor="#0d1117", edgecolor="#2d1f3d", labelcolor="#94a3b8", fontsize=8)
    ax.grid(color="#2d1f3d", linewidth=0.5, alpha=0.5)
    fig.tight_layout(); st.pyplot(fig); plt.close()
with ich2:
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    ax.plot(df.index, df["if_score"], color="#a855f7", linewidth=1.2, alpha=0.8)
    ax.fill_between(df.index, df["if_score"], alpha=0.1, color="#a855f7")
    ax.axhline(y=0, color="#f97316", linewidth=1, linestyle="--", label="Decision Boundary (0)")
    ax.scatter(df.index[if_mask], df["if_score"][if_mask], color="#f97316", s=60, zorder=5, marker="D")
    ax.set_title("IF Anomaly Score (lower = more anomalous)", color="#94a3b8", fontsize=10, pad=10, loc="left")
    ax.tick_params(colors="#475569", labelsize=8); ax.spines[:].set_color("#2d1f3d")
    ax.legend(facecolor="#0d1117", edgecolor="#2d1f3d", labelcolor="#94a3b8", fontsize=8)
    ax.grid(color="#2d1f3d", linewidth=0.5, alpha=0.5)
    fig.tight_layout(); st.pyplot(fig); plt.close()

st.markdown('<div class="section-header-if">IF Model Performance</div>', unsafe_allow_html=True)
ich3, ich4 = st.columns(2)
with ich3:
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    ax.imshow(if_cm, interpolation="nearest", cmap=plt.cm.get_cmap("Purples"))
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Anomaly"], color="#c4b5fd", fontsize=9)
    ax.set_yticklabels(["Normal","Anomaly"], color="#c4b5fd", fontsize=9)
    ax.set_xlabel("Predicted", color="#7c3aed", fontsize=9)
    ax.set_ylabel("Actual",    color="#7c3aed", fontsize=9)
    ax.set_title("IF Confusion Matrix", color="#c4b5fd", fontsize=10, pad=10, loc="left")
    ax.spines[:].set_color("#2d1f3d")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(if_cm[i,j]), ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    fig.tight_layout(); st.pyplot(fig); plt.close()
with ich4:
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    normal_mask = df["if_anomaly"] == 0
    ax.scatter(df.loc[normal_mask, "cpu_usage_pct"], df.loc[normal_mask, "network_in_mb"],
               c="#a855f7", alpha=0.5, s=20, label="Normal")
    ax.scatter(df.loc[if_mask, "cpu_usage_pct"], df.loc[if_mask, "network_in_mb"],
               c="#f97316", alpha=0.9, s=60, marker="D", label="IF Anomaly", zorder=5)
    ax.set_title("IF Decision Space: CPU vs Network In", color="#c4b5fd", fontsize=10, pad=10, loc="left")
    ax.set_xlabel("CPU Utilization (%)", color="#7c3aed", fontsize=8)
    ax.set_ylabel("Network In (MB)", color="#7c3aed", fontsize=8)
    ax.tick_params(colors="#475569", labelsize=8); ax.spines[:].set_color("#2d1f3d")
    ax.legend(facecolor="#0d1117", edgecolor="#2d1f3d", labelcolor="#c4b5fd", fontsize=8)
    ax.grid(color="#2d1f3d", linewidth=0.5, alpha=0.5)
    fig.tight_layout(); st.pyplot(fig); plt.close()

# ─────────────────────────────────────────
# MODEL COMPARISON
# ─────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown('<div class="section-header-compare">Model Comparison · LR vs Isolation Forest</div>', unsafe_allow_html=True)

comp1, comp2 = st.columns(2)
with comp1:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
    fig.patch.set_facecolor("#0d1117")
    for ax, mask, color, label in [
        (ax1, lr_mask, "#3b82f6", "Logistic Regression"),
        (ax2, if_mask, "#a855f7", "Isolation Forest"),
    ]:
        ax.set_facecolor("#0d1117")
        ax.plot(df.index, df["cpu_usage_pct"], color=color, linewidth=1, alpha=0.6)
        ax.scatter(df.index[mask], df["cpu_usage_pct"][mask], color="#ef4444", s=40, zorder=5)
        ax.set_title(label, color="#94a3b8", fontsize=9, loc="left", pad=4)
        ax.tick_params(colors="#475569", labelsize=7)
        ax.spines[:].set_color("#1e293b")
        ax.grid(color="#1e293b", linewidth=0.4, alpha=0.4)
    fig.suptitle("Anomaly Detections: CPU Over Time", color="#94a3b8", fontsize=10, y=1.01)
    fig.tight_layout(); st.pyplot(fig); plt.close()

with comp2:
    agree_mask   = (df["lr_anomaly"] == 1) & (df["if_anomaly"] == 1)
    lr_only_mask = (df["lr_anomaly"] == 1) & (df["if_anomaly"] == 0)
    if_only_mask = (df["lr_anomaly"] == 0) & (df["if_anomaly"] == 1)
    n_both    = int(agree_mask.sum())
    n_lr_only = int(lr_only_mask.sum())
    n_if_only = int(if_only_mask.sum())
    n_neither = total - n_both - n_lr_only - n_if_only

    labels_pie, sizes_pie, colors_pie = [], [], []
    for lbl, sz, clr in [
        ("Both Agree", n_both, "#22c55e"), ("LR Only", n_lr_only, "#3b82f6"),
        ("IF Only", n_if_only, "#a855f7"), ("Neither", n_neither, "#1e293b"),
    ]:
        if sz > 0:
            labels_pie.append(f"{lbl}\n({sz})")
            sizes_pie.append(sz)
            colors_pie.append(clr)

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    _, texts, autotexts = ax.pie(
        sizes_pie, labels=labels_pie, colors=colors_pie,
        autopct="%1.0f%%", startangle=90,
        textprops={"color": "#94a3b8", "fontsize": 8},
        wedgeprops={"edgecolor": "#0d1117", "linewidth": 2}
    )
    for at in autotexts: at.set_color("white"); at.set_fontsize(8)
    ax.set_title("Detection Agreement", color="#94a3b8", fontsize=10, pad=10, loc="left")
    fig.tight_layout(); st.pyplot(fig); plt.close()

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""<div class="info-box">
    <b style="color:#e2e8f0">Model Consensus Summary</b><br><br>
    <span class="agree-badge">✅ BOTH AGREE</span>&nbsp; {n_both} datapoints flagged by both — highest confidence anomalies<br><br>
    <span class="model-chip-lr">LR ONLY</span>&nbsp; {n_lr_only} datapoints — near the linear decision boundary<br><br>
    <span class="model-chip-if">IF ONLY</span>&nbsp; {n_if_only} datapoints — multivariate anomalies LR missed<br><br>
    <span class="disagree-badge">NEITHER</span>&nbsp; {n_neither} datapoints classified normal by both models
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# REMEDIATION ENGINE
# ─────────────────────────────────────────
st.markdown('<div class="section-header">Autonomous Remediation Engine <span class="badge-lr">LR-Driven</span></div>', unsafe_allow_html=True)

anomalies = df[df["is_anomaly"] == 1].copy()

if len(anomalies) == 0:
    st.markdown('<div class="info-box">✅ No anomalies in current window. No remediation triggered.</div>',
                unsafe_allow_html=True)
else:
    st.markdown(f"""<div class="info-box">
        🤖 <b style="color:#e2e8f0">Auto-Remediation Active</b> —
        {len(anomalies)} anomal{'y' if len(anomalies)==1 else 'ies'} detected.
        Executing corrective actions scaled to z-score severity.
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    for _, row in anomalies.iterrows():
        ts  = row["Timestamp"].strftime("%Y-%m-%d %H:%M UTC") \
              if hasattr(row["Timestamp"], "strftime") else str(row["Timestamp"])
        sev, actions = remediate(row["z_score"], row["cpu_usage_pct"], ts)
        cls  = {"HIGH":"remediation-high","MEDIUM":"remediation-medium","LOW":"remediation-low"}.get(sev,"remediation-low")
        icon = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(sev,"⚪")
        acts = " &nbsp;·&nbsp; ".join([f"{'✅' if a['ok'] else '⚠️'} {a['action']}: {a['msg']}" for a in actions])
        if_agreed = row.get("if_anomaly", 0) == 1
        if_badge  = '<span class="agree-badge">IF ✓</span>' if if_agreed else '<span class="disagree-badge">IF ✗</span>'
        cat_chips = " ".join([
            f'<span class="{CAT_META[c]["cls"]}">{CAT_META[c]["icon"]} {c}</span>'
            for c in row.get("anomaly_categories", [])
        ]) or '<span style="color:#475569;font-size:0.75rem;">no category</span>'

        st.markdown(f"""<div class="{cls}">
            {icon} <b>{ts}</b> &nbsp;|&nbsp; Severity: <b>{sev}</b> &nbsp;|&nbsp;
            CPU: <b>{row['cpu_usage_pct']:.2f}%</b> &nbsp;|&nbsp;
            Z-Score: <b>{row['z_score']:.3f}σ</b> &nbsp;|&nbsp; {if_badge}<br>
            <span style="font-size:0.75rem;">{cat_chips}</span><br>
            <span style="opacity:0.8;font-size:0.75rem;">{acts}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
        ℹ️ Instance status reflects <b style="color:#e2e8f0">post-remediation state</b> pulled live from AWS.
        Anomaly category chips show which detection rules triggered per event.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""<div style="text-align:center;color:#1e293b;font-family:'Space Mono',monospace;font-size:0.7rem;padding:1rem 0;">
    CLOUDSENTINEL · MIT BENGALURU HACKATHON 2026 · AUTONOMOUS CLOUD COST INTELLIGENCE
</div>""", unsafe_allow_html=True)
