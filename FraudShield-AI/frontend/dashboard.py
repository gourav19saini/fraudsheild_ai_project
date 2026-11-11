"""
FraudShield AI - Streamlit Dashboard
========================================
Run:
    streamlit run frontend/dashboard.py
"""

import sys
import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend import api_client
from simulator.stream import get_stream

# ----------------------------------------------------------------------
# PAGE CONFIG & THEME
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main { background-color: #0e1117; }
    .stApp { background: linear-gradient(180deg, #0b0f19 0%, #0e1117 100%); }

    .fs-header-title { font-size: 2.4rem; font-weight: 800; color: #f1f5f9; margin-bottom: 0; }
    .fs-header-sub { font-size: 1.05rem; color: #8b95a7; margin-top: 0.2rem; }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #161b28, #1b2130);
        border: 1px solid #262d3d;
        border-radius: 14px;
        padding: 14px 16px 10px 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetricLabel"] { color: #93a0b8 !important; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: #f1f5f9 !important; }

    .fs-badge-safe { background:#0f3d2e; color:#4ade80; padding:3px 10px; border-radius:999px; font-weight:700; font-size:0.78rem; }
    .fs-badge-review { background:#3d3410; color:#facc15; padding:3px 10px; border-radius:999px; font-weight:700; font-size:0.78rem; }
    .fs-badge-high { background:#3d1414; color:#f87171; padding:3px 10px; border-radius:999px; font-weight:700; font-size:0.78rem; }

    .fs-alert-card {
        background: linear-gradient(145deg, #241414, #1a1010);
        border: 1px solid #4c1d1d;
        border-left: 4px solid #ef4444;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .fs-alert-title { color: #f87171; font-weight: 800; font-size: 1rem; margin-bottom: 6px; }
    .fs-alert-meta { color: #cbd5e1; font-size: 0.85rem; }
    .fs-reason { color: #e2e8f0; font-size: 0.85rem; margin: 2px 0; }

    .fs-section-title { color: #f1f5f9; font-size: 1.25rem; font-weight: 700; margin: 6px 0 10px 0; border-left: 4px solid #6366f1; padding-left: 10px; }

    .fs-panel {
        background: #131826;
        border: 1px solid #232a3b;
        border-radius: 12px;
        padding: 16px;
    }

    .fs-status-pill { padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; display:inline-block; }
    .fs-status-normal { background:#0f3d2e; color:#4ade80; }
    .fs-status-warning { background:#3d3410; color:#facc15; }
    .fs-status-detected { background:#3d1414; color:#f87171; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

RISK_COLORS = {"SAFE": "#4ade80", "REVIEW": "#facc15", "HIGH RISK": "#f87171"}


def risk_badge(level: str) -> str:
    cls = {"SAFE": "fs-badge-safe", "REVIEW": "fs-badge-review", "HIGH RISK": "fs-badge-high"}.get(level, "fs-badge-review")
    return f'<span class="{cls}">{level}</span>'


# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.markdown('<p class="fs-header-title">🛡️ FRAUDSHIELD AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="fs-header-sub">Real-Time Fraud &amp; Anomaly Detection System &nbsp;|&nbsp; PS-02</p>', unsafe_allow_html=True)
with header_col2:
    health, health_err = api_client.check_health()
    if health:
        st.success(f"Backend ONLINE — model {health.get('model_version') or 'n/a'}")
    else:
        st.error("Backend OFFLINE")

st.markdown("---")

# ----------------------------------------------------------------------
# SIDEBAR - STREAM & DEMO CONTROLS
# ----------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Controls")

stream = get_stream(api_client.API_BASE_URL)

st.sidebar.markdown("### 📡 Live Transaction Stream")
stream_col1, stream_col2 = st.sidebar.columns(2)
if stream_col1.button("▶️ START", use_container_width=True):
    stream.start()
if stream_col2.button("⏹️ STOP", use_container_width=True):
    stream.stop()

if stream.is_running:
    st.sidebar.success(f"Streaming... ({stream.sent_count} sent)")
else:
    st.sidebar.info("Stream stopped")
if stream.last_error:
    st.sidebar.warning(f"Last error: {stream.last_error}")

st.sidebar.markdown("### 🧪 Manual Transaction Injection")
if st.sidebar.button("Generate NORMAL transaction", use_container_width=True):
    from simulator.generator import generate_transaction
    txn = generate_transaction("normal")
    result, err = api_client.submit_transaction(txn)
    if err:
        st.sidebar.error(err)
    else:
        st.sidebar.success(f"{result['transaction_id']}: {result['risk_level']} ({result['risk_score']})")

if st.sidebar.button("🚨 Generate SUSPICIOUS transaction", use_container_width=True):
    from simulator.generator import generate_transaction
    txn = generate_transaction("suspicious")
    result, err = api_client.submit_transaction(txn)
    if err:
        st.sidebar.error(err)
    else:
        st.sidebar.warning(f"{result['transaction_id']}: {result['risk_level']} ({result['risk_score']})")

if st.sidebar.button("💀 Generate EXTREME FRAUD transaction", use_container_width=True):
    from simulator.generator import generate_transaction
    txn = generate_transaction("extreme")
    result, err = api_client.submit_transaction(txn)
    if err:
        st.sidebar.error(err)
    else:
        st.sidebar.error(f"{result['transaction_id']}: {result['risk_level']} ({result['risk_score']})")

st.sidebar.markdown("### ⚠️ Concept Drift")
if st.sidebar.button("SIMULATE CONCEPT DRIFT", use_container_width=True):
    result, err = api_client.simulate_drift()
    if err:
        st.sidebar.error(err)
    else:
        st.sidebar.success("Drift scenario injected")
        st.session_state["last_drift_result"] = result

st.sidebar.markdown("### 🔄 Model")
if st.sidebar.button("RETRAIN MODEL", use_container_width=True):
    with st.spinner("Retraining Isolation Forest..."):
        result, err = api_client.retrain_model()
    if err:
        st.sidebar.error(err)
    else:
        st.sidebar.success(f"Retrained -> {result['version']} ({result['training_samples']} samples, {result['training_time_seconds']}s)")

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("🔁 Auto-refresh every 5s", value=stream.is_running)
if st.sidebar.button("Refresh now", use_container_width=True):
    st.rerun()

# ----------------------------------------------------------------------
# FETCH DATA
# ----------------------------------------------------------------------
stats, stats_err = api_client.get_statistics()
txns, txns_err = api_client.get_transactions(limit=200)
alerts, alerts_err = api_client.get_alerts(limit=10)
model_status, model_err = api_client.get_model_status()
drift, drift_err = api_client.get_drift_status()

if stats_err:
    st.warning(stats_err)

# ----------------------------------------------------------------------
# A. KPI CARDS
# ----------------------------------------------------------------------
st.markdown('<p class="fs-section-title">📊 Key Metrics</p>', unsafe_allow_html=True)
if stats:
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total Transactions", f"{stats['total_transactions']:,}")
    k2.metric("Safe", f"{stats['safe_count']:,}")
    k3.metric("Review", f"{stats['review_count']:,}")
    k4.metric("High Risk", f"{stats['high_risk_count']:,}")
    k5.metric("Avg Risk Score", f"{stats['average_risk_score']:.1f}")
    k6.metric("Anomaly Rate", f"{stats['anomaly_rate_pct']:.2f}%")
else:
    st.info("No statistics available yet — start the stream or generate a transaction.")

st.markdown("")

# ----------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------
tab_monitor, tab_alerts, tab_analyzer, tab_charts, tab_model = st.tabs(
    ["🔴 Live Monitor", "🚨 Alerts", "🔍 Transaction Analyzer", "📈 Charts", "🧠 Model & Drift"]
)

# ----------------------------------------------------------------------
# B. LIVE MONITOR
# ----------------------------------------------------------------------
with tab_monitor:
    st.markdown('<p class="fs-section-title">Latest Transactions</p>', unsafe_allow_html=True)
    if txns_err:
        st.warning(txns_err)
    elif not txns:
        st.info("No transactions yet. Start the live stream or generate one from the sidebar.")
    else:
        df = pd.DataFrame(txns)
        display_df = df[[
            "transaction_id", "user_id", "amount", "merchant", "location",
            "risk_score", "risk_level", "decision", "timestamp",
        ]].copy()
        display_df.columns = ["Transaction ID", "User", "Amount (₹)", "Merchant", "Location", "Risk Score", "Risk Level", "Decision", "Timestamp"]

        def style_risk(val):
            color = RISK_COLORS.get(val, "#cbd5e1")
            return f"color: {color}; font-weight: 700;"

        styled = display_df.head(50).style.map(style_risk, subset=["Risk Level"])
        st.dataframe(styled, use_container_width=True, height=520)

# ----------------------------------------------------------------------
# C. LIVE ALERT PANEL
# ----------------------------------------------------------------------
with tab_alerts:
    st.markdown('<p class="fs-section-title">High-Risk & Review Alerts</p>', unsafe_allow_html=True)
    if alerts_err:
        st.warning(alerts_err)
    elif not alerts:
        st.info("No alerts yet. Suspicious or high-risk transactions will appear here in real time.")
    else:
        for a in alerts:
            icon = "🚨" if a["risk_level"] == "HIGH RISK" else "⚠️"
            reasons_html = "".join(f'<div class="fs-reason">✓ {r}</div>' for r in a["reasons"])
            st.markdown(
                f"""
                <div class="fs-alert-card">
                    <div class="fs-alert-title">{icon} {a['risk_level']} TRANSACTION — {a['transaction_id']}</div>
                    <div class="fs-alert-meta">Risk Score: <b>{a['risk_score']}/100</b> &nbsp;|&nbsp; Status: {a['status']}</div>
                    <div style="margin-top:8px;">{reasons_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------
# D. TRANSACTION ANALYZER
# ----------------------------------------------------------------------
with tab_analyzer:
    st.markdown('<p class="fs-section-title">Analyze a Transaction</p>', unsafe_allow_html=True)
    with st.form("analyzer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            user_id = st.text_input("User ID", value="U2001")
            amount = st.number_input("Amount (₹)", min_value=1.0, value=1500.0, step=100.0)
            merchant = st.text_input("Merchant", value="Croma")
        with c2:
            merchant_category = st.text_input("Merchant Category", value="Electronics")
            location = st.text_input("Location", value="Jaipur")
            device_id = st.text_input("Device ID", value="DEV12345")
        with c3:
            latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=26.9124, format="%.4f")
            longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=75.7873, format="%.4f")
        submitted = st.form_submit_button("🔍 ANALYZE TRANSACTION", use_container_width=True)

    if submitted:
        payload = {
            "user_id": user_id, "amount": amount, "merchant": merchant,
            "merchant_category": merchant_category, "location": location,
            "latitude": latitude, "longitude": longitude, "device_id": device_id,
        }
        with st.spinner("Scoring transaction through the ML pipeline..."):
            result, err = api_client.analyze_transaction(payload)
        if err:
            st.error(err)
        else:
            level = result["risk_level"]
            color = RISK_COLORS.get(level, "#cbd5e1")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Risk Score", f"{result['risk_score']}/100")
            r2.markdown(f"**Risk Level**<br><span style='color:{color}; font-size:1.3rem; font-weight:800'>{level}</span>", unsafe_allow_html=True)
            r3.metric("Decision", result["decision"])
            r4.metric("Latency", f"{result['prediction_latency_ms']:.2f} ms")

            st.markdown("**Anomaly detected:** " + ("✅ Yes" if result["anomaly"] else "❌ No"))
            st.markdown("**Reasons:**")
            for reason in result["reasons"]:
                st.markdown(f"- {reason}")

# ----------------------------------------------------------------------
# E. CHARTS
# ----------------------------------------------------------------------
with tab_charts:
    if txns_err or not txns:
        st.info("No transaction data yet to chart.")
    else:
        df = pd.DataFrame(txns)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Transactions Over Time**")
            ts_df = df.set_index("timestamp").resample("1min").size().reset_index(name="count")
            fig = px.line(ts_df, x="timestamp", y="count", markers=True)
            fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Risk Level Distribution**")
            level_counts = df["risk_level"].value_counts().reset_index()
            level_counts.columns = ["risk_level", "count"]
            fig = px.pie(level_counts, names="risk_level", values="count",
                         color="risk_level", color_discrete_map=RISK_COLORS, hole=0.45)
            fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Risk Score Distribution**")
            fig = px.histogram(df, x="risk_score", nbins=20, color_discrete_sequence=["#6366f1"])
            fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.markdown("**Transaction Amount Distribution**")
            fig = px.histogram(df, x="amount", nbins=25, color_discrete_sequence=["#22d3ee"])
            fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            st.markdown("**Safe vs Suspicious Over Time**")
            df["bucket"] = df["risk_level"].apply(lambda x: "Suspicious" if x != "SAFE" else "Safe")
            trend = df.groupby([pd.Grouper(key="timestamp", freq="1min"), "bucket"]).size().reset_index(name="count")
            fig = px.bar(trend, x="timestamp", y="count", color="bucket",
                         color_discrete_map={"Safe": "#4ade80", "Suspicious": "#f87171"}, barmode="stack")
            fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col6:
            st.markdown("**Merchant Category Analysis**")
            if "merchant_category" in df.columns:
                cat_counts = df["merchant_category"].fillna("Other").value_counts().reset_index()
                cat_counts.columns = ["category", "count"]
                fig = px.bar(cat_counts, x="category", y="count", color_discrete_sequence=["#a78bfa"])
                fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# F. MODEL MONITORING + DRIFT + EVALUATION
# ----------------------------------------------------------------------
with tab_model:
    st.markdown('<p class="fs-section-title">Model Monitoring</p>', unsafe_allow_html=True)
    if model_err:
        st.warning(model_err)
    elif model_status:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model", model_status["model_name"])
        m2.metric("Status", model_status["status"])
        m3.metric("Version", model_status["version"])
        m4.metric("Transactions Processed", f"{model_status['transactions_processed']:,}")

        m5, m6, m7 = st.columns(3)
        m5.metric("Anomalies Detected", f"{model_status['anomalies_detected']:,}")
        m6.metric("Anomaly Rate", f"{model_status['anomaly_rate_pct']:.2f}%")
        m7.metric("Last Retrained", (model_status["last_retrained"] or "—")[:19])

    st.markdown('<p class="fs-section-title">⚠️ Concept Drift Status</p>', unsafe_allow_html=True)
    if drift_err:
        st.warning(drift_err)
    elif drift:
        pill_class = {"NORMAL": "fs-status-normal", "WARNING": "fs-status-warning", "DETECTED": "fs-status-detected"}.get(drift["status"], "fs-status-normal")
        icon = {"NORMAL": "✅", "WARNING": "⚠️", "DETECTED": "🚨"}.get(drift["status"], "✅")
        st.markdown(f'<span class="fs-status-pill {pill_class}">{icon} {drift["status"]}</span>', unsafe_allow_html=True)
        st.caption(drift["message"])
        if drift.get("reference_mean") is not None:
            d1, d2, d3 = st.columns(3)
            d1.metric("Reference Mean Amount", f"₹{drift['reference_mean']:,.0f}")
            d2.metric("Current Mean Amount", f"₹{drift['current_mean']:,.0f}")
            d3.metric("Z-Score", f"{drift['z_score']:.2f}")

    st.markdown('<p class="fs-section-title">📐 Evaluation Metrics (on labeled synthetic data)</p>', unsafe_allow_html=True)
    eval_metrics, eval_err = api_client.get_model_evaluation()
    if eval_err:
        st.info("Run `python data/generate_dataset.py` and train the model to see evaluation metrics.")
    elif eval_metrics:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Precision", f"{eval_metrics['precision']:.3f}")
        e2.metric("Recall", f"{eval_metrics['recall']:.3f}")
        e3.metric("F1 Score", f"{eval_metrics['f1_score']:.3f}")
        e4.metric("ROC-AUC", f"{eval_metrics['roc_auc']:.3f}" if eval_metrics["roc_auc"] else "n/a")
        st.caption(eval_metrics["note"])

        cm = eval_metrics["confusion_matrix"]
        cm_fig = go.Figure(data=go.Heatmap(
            z=[[cm["true_negative"], cm["false_positive"]], [cm["false_negative"], cm["true_positive"]]],
            x=["Predicted Normal", "Predicted Anomaly"],
            y=["Actual Normal", "Actual Fraud"],
            colorscale="Blues",
            text=[[cm["true_negative"], cm["false_positive"]], [cm["false_negative"], cm["true_positive"]]],
            texttemplate="%{text}",
        ))
        cm_fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=30, b=10), title="Confusion Matrix")
        st.plotly_chart(cm_fig, use_container_width=True)

st.markdown("---")
st.caption("FraudShield AI · Built for PS-02: Real-Time Algorithmic Fraud & Anomaly Detection in Streaming Data")

if auto_refresh:
    time.sleep(5)
    st.rerun()
