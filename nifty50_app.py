
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Configuration and visual system
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="ASTRAQUANT X | NIFTY 50 AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES = {0: "BEARISH", 1: "SIDEWAYS", 2: "BULLISH"}
CLASS_COLORS = {"BEARISH": "#ff5d6c", "SIDEWAYS": "#f5c451", "BULLISH": "#35d49a"}
FEATURES = [
    "return_1d", "return_3d", "return_5d", "volatility_5d",
    "volatility_10d", "ema_gap_9_21", "ema_gap_21_50",
    "price_vs_sma20", "rsi_14", "macd_pct", "macd_signal_pct",
    "macd_hist_pct", "range_pct", "volume_change", "momentum_10d",
]
LABELS = [0, 1, 2]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --bg:#080d18; --panel:#101a2b; --line:#22324a; --muted:#91a2ba; --text:#edf4ff; }
.stApp { background: radial-gradient(ellipse at 15% 0%, #14243a 0%, #080d18 48%); color:var(--text); }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
[data-testid="stSidebar"] { background:#0b1423; border-right:1px solid #22324a; }
.block-container { padding-top:1.6rem; padding-bottom:3rem; max-width:1500px; }
h1,h2,h3 { color:#f2f7ff !important; letter-spacing:-.035em; }
p, label, .stMarkdown { color:#d6e1f1; }
small, .muted { color:var(--muted); }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#142238,#0d1727); border:1px solid #263952; border-radius:14px; padding:15px 17px; }
div[data-testid="stMetricLabel"] { color:#9fb0c8; }
div[data-testid="stMetricValue"] { color:#f2f7ff; font-family:'DM Mono',monospace; }
div[data-testid="stMetricDelta"] { font-family:'DM Mono',monospace; }
div[data-testid="stTabs"] button { color:#a9bad1; }
div[data-testid="stTabs"] button[aria-selected="true"] { color:#35d49a; border-bottom-color:#35d49a; }
.stButton button[kind="primary"] { background:#20b985; border:0; color:#06130f; font-weight:800; border-radius:9px; }
.stButton button { border-radius:9px; }
div[data-testid="stAlert"] { border-radius:12px; }
.hero { padding:22px 26px; border:1px solid #2a405e; border-radius:18px; background:linear-gradient(115deg,rgba(23,43,68,.96),rgba(13,24,41,.92)); margin-bottom:18px; }
.eyebrow { color:#45d9ab; font-size:11px; font-weight:800; letter-spacing:.18em; text-transform:uppercase; }
.hero-title { font-size:30px; line-height:1.15; font-weight:800; color:#f4f8ff; margin-top:7px; }
.hero-sub { color:#9db0c9; font-size:13px; margin-top:8px; }
.section-kicker { color:#48d9ad; font-size:11px; font-weight:800; letter-spacing:.16em; text-transform:uppercase; margin:10px 0 4px; }
.signal { display:inline-block; padding:7px 13px; border-radius:9px; font-size:12px; font-weight:800; letter-spacing:.08em; }
.signal-buy { color:#45e0ad; background:rgba(53,212,154,.12); border:1px solid rgba(53,212,154,.35); }
.signal-sell { color:#ff7b86; background:rgba(255,93,108,.12); border:1px solid rgba(255,93,108,.35); }
.signal-hold { color:#f5c451; background:rgba(245,196,81,.12); border:1px solid rgba(245,196,81,.35); }
.note { background:#101d30; border:1px solid #263a55; border-radius:12px; padding:13px 16px; color:#aebed2; font-size:12px; }
hr { border-color:#22324a !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def load_data(period: str) -> pd.DataFrame:
    data = yf.download("^NSEI", period=period, interval="1d",
                       auto_adjust=True, progress=False, threads=False)
    if data is None or data.empty:
        raise ValueError("Yahoo Finance returned no rows. Check connectivity or try another period.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Required market columns are missing: {missing}")
    data = data[required].copy()
    data.index = pd.to_datetime(data.index)
    data = data[~data.index.duplicated(keep="last")].sort_index()
    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    data = data[(data[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    # A broad range consistency check; rows failing OHLC ordering are discarded.
    valid = ((data["High"] >= data[["Open", "Close", "Low"]].max(axis=1)) &
             (data["Low"] <= data[["Open", "Close", "High"]].min(axis=1)))
    data = data.loc[valid].copy()
    data["Volume"] = data["Volume"].fillna(0).clip(lower=0)
    return data


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.mask((loss == 0) & (gain > 0), 100)
    out = out.mask((gain == 0) & (loss > 0), 0)
    return out


def engineer_features(raw: pd.DataFrame, threshold: float):
    df = raw.copy()
    c = df["Close"]
    df["return_1d"] = c.pct_change()
    df["return_3d"] = c.pct_change(3)
    df["return_5d"] = c.pct_change(5)
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["ema_9"] = c.ewm(span=9, adjust=False).mean()
    df["ema_21"] = c.ewm(span=21, adjust=False).mean()
    df["ema_50"] = c.ewm(span=50, adjust=False).mean()
    df["sma_20"] = c.rolling(20).mean()
    df["ema_gap_9_21"] = df["ema_9"] / df["ema_21"] - 1
    df["ema_gap_21_50"] = df["ema_21"] / df["ema_50"] - 1
    df["price_vs_sma20"] = c / df["sma_20"] - 1
    df["rsi_14"] = calculate_rsi(c, 14)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    # Scale MACD by close so models are less dependent on index level.
    df["macd_pct"] = df["macd"] / c
    df["macd_signal_pct"] = df["macd_signal"] / c
    df["macd_hist_pct"] = (df["macd"] - df["macd_signal"]) / c
    df["range_pct"] = (df["High"] - df["Low"]) / c
    df["volume_change"] = df["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    df["momentum_10d"] = c / c.shift(10) - 1
    df["daily_return"] = c.pct_change()
    df["future_return"] = c.shift(-1) / c - 1
    # Future return is used only for the label, never as a model feature.
    df["target"] = np.select(
        [df["future_return"] < -threshold, df["future_return"] > threshold],
        [0, 2], default=1
    )
    usable = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["future_return"]).copy()
    usable["target"] = usable["target"].astype(int)
    latest = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES).copy()
    return df, usable, latest


# ---------------------------------------------------------------------
# Model comparison: chronological train / validation / final test
# ---------------------------------------------------------------------
def candidate_models(trees: int):
    return {
        "Random Forest": Pipeline([
            ("scale", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=trees, max_depth=8, min_samples_leaf=4,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1
            ))
        ]),
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000, class_weight="balanced", C=0.5
            ))
        ]),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42
        ),
    }


def fit_and_evaluate(dataset: pd.DataFrame, trees: int, train_fraction: float):
    n = len(dataset)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + (1 - train_fraction) / 2))
    train = dataset.iloc[:train_end]
    valid = dataset.iloc[train_end:val_end]
    test = dataset.iloc[val_end:]
    if min(len(train), len(valid), len(test)) < 20:
        raise ValueError("Not enough observations in chronological train/validation/test segments.")
    if train.target.nunique() < 2 or valid.target.nunique() < 2 or test.target.nunique() < 2:
        raise ValueError("At least one time segment has fewer than two classes. Adjust threshold or history.")
    scores, fitted = [], {}
    for name, estimator in candidate_models(trees).items():
        try:
            estimator.fit(train[FEATURES], train.target)
            pred = estimator.predict(valid[FEATURES])
            score = f1_score(valid.target, pred, labels=LABELS, average="macro", zero_division=0)
            scores.append({"Model": name, "Validation macro F1": score})
            fitted[name] = estimator
        except Exception as exc:
            scores.append({"Model": name, "Validation macro F1": np.nan, "Error": str(exc)})
    valid_scores = [x for x in scores if np.isfinite(x["Validation macro F1"])]
    if not valid_scores:
        raise ValueError("All candidate models failed during validation.")
    winner = max(valid_scores, key=lambda x: x["Validation macro F1"])["Model"]
    # Refit selected model on train + validation. The final test remains untouched.
    final_eval = candidate_models(trees)[winner]
    train_valid = dataset.iloc[:val_end]
    final_eval.fit(train_valid[FEATURES], train_valid.target)
    test_pred = final_eval.predict(test[FEATURES])
    report = {
        "accuracy": accuracy_score(test.target, test_pred),
        "precision": precision_score(test.target, test_pred, labels=LABELS, average="macro", zero_division=0),
        "recall": recall_score(test.target, test_pred, labels=LABELS, average="macro", zero_division=0),
        "f1": f1_score(test.target, test_pred, labels=LABELS, average="macro", zero_division=0),
    }
    # Baseline predicts the most common class in the train+validation period.
    baseline_class = int(train_valid.target.mode().iloc[0])
    baseline_pred = np.full(len(test), baseline_class)
    baseline_f1 = f1_score(test.target, baseline_pred, labels=LABELS, average="macro", zero_division=0)
    return winner, scores, final_eval, test, test_pred, report, baseline_f1, train_valid


def train_latest(dataset: pd.DataFrame, winner: str, trees: int):
    model = candidate_models(trees)[winner]
    model.fit(dataset[FEATURES], dataset.target)
    return model


def model_probabilities(model, row):
    raw = model.predict_proba(row[FEATURES])[0]
    classes = model.classes_ if hasattr(model, "classes_") else model.named_steps["model"].classes_
    return {int(k): float(v) for k, v in zip(classes, raw)}


# ---------------------------------------------------------------------
# Backtest and metrics
# ---------------------------------------------------------------------
def run_backtest(test: pd.DataFrame, predictions: np.ndarray,
                 cost_bps: float, slippage_bps: float):
    bt = test.copy()
    bt["predicted"] = predictions
    bt["predicted_trend"] = bt.predicted.map(CLASS_NAMES)
    bt["actual_trend"] = bt.target.map(CLASS_NAMES)
    bt["next_return"] = bt.future_return
    bt["position"] = bt.predicted.map({0: -1, 1: 0, 2: 1}).astype(float)
    bt["turnover"] = bt.position.diff().abs()
    if len(bt):
        bt.iloc[0, bt.columns.get_loc("turnover")] = abs(bt.position.iloc[0])
    friction = (cost_bps + slippage_bps) / 10000
    bt["strategy_return"] = bt.position * bt.next_return - bt.turnover * friction
    bt["buy_hold_return"] = bt.next_return
    bt = bt.replace([np.inf, -np.inf], np.nan).dropna(subset=["strategy_return", "buy_hold_return"])
    bt["strategy_equity"] = (1 + bt.strategy_return).cumprod()
    bt["buy_hold_equity"] = (1 + bt.buy_hold_return).cumprod()
    bt["drawdown"] = bt.strategy_equity / bt.strategy_equity.cummax() - 1
    wins = (bt.loc[bt.position != 0, "strategy_return"] > 0).sum()
    active = int((bt.position != 0).sum())
    gross_profit = bt.loc[bt.strategy_return > 0, "strategy_return"].sum()
    gross_loss = abs(bt.loc[bt.strategy_return < 0, "strategy_return"].sum())
    std = bt.strategy_return.std()
    metrics = {
        "net_return": (bt.strategy_equity.iloc[-1] - 1) * 100 if len(bt) else 0,
        "buy_hold": (bt.buy_hold_equity.iloc[-1] - 1) * 100 if len(bt) else 0,
        "win_rate": wins / active * 100 if active else 0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.nan,
        "sharpe": (bt.strategy_return.mean() / std * np.sqrt(252)) if std and np.isfinite(std) else 0,
        "max_drawdown": bt.drawdown.min() * 100 if len(bt) else 0,
        "trades_days": active,
    }
    return bt, metrics


def style_chart(fig, height=380):
    fig.update_layout(
        height=height, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,13,24,0.25)", font=dict(family="Manrope", color="#cbd7e8"),
        margin=dict(l=12, r=12, t=42, b=12), legend=dict(orientation="h", y=1.08),
        xaxis=dict(gridcolor="#1c2a3e", zerolinecolor="#1c2a3e"),
        yaxis=dict(gridcolor="#1c2a3e", zerolinecolor="#1c2a3e"),
    )
    return fig


# ---------------------------------------------------------------------
# Sidebar controls and execution
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ◈ ASTRAQUANT X")
    st.caption("NIFTY 50 · AI RESEARCH TERMINAL")
    st.divider()
    page = st.radio("WORKSPACE", [
        "Overview", "AI Prediction", "Technical Analysis",
        "Model Performance", "Strategy & Backtest", "Methodology"
    ], label_visibility="visible")
    st.divider()
    st.markdown("**DATA & MODEL CONFIGURATION**")
    period = st.selectbox("Historical window", ["1y", "2y", "5y", "10y"], index=2)
    threshold_pct = st.slider("Class boundary (%)", 0.05, 1.00, 0.20, 0.05)
    confidence_threshold = st.slider("Signal confidence floor", 0.35, 0.90, 0.55, 0.05)
    trees = st.slider("Random Forest trees", 100, 600, 300, 50)
    train_fraction = st.slider("Training share (pre-test)", 0.60, 0.85, 0.70, 0.05)
    cost_bps = st.number_input("Transaction cost (bps)", 0.0, 50.0, 10.0, 1.0)
    slippage_bps = st.number_input("Slippage (bps)", 0.0, 50.0, 5.0, 1.0)
    refresh = st.button("↻  Refresh data & retrain", type="primary", use_container_width=True)
    st.caption("Data: Yahoo Finance · ^NSEI · Daily adjusted OHLCV")
    st.caption("Research prototype only. No order execution.")


if refresh:
    load_data.clear()

try:
    with st.spinner("Loading NIFTY 50 history and training chronological model candidates…"):
        raw = load_data(period)
        engineered, dataset, latest_features = engineer_features(raw, threshold_pct / 100)
        if len(dataset) < 180:
            raise ValueError("Fewer than 180 labelled observations. Select a longer historical window.")
        winner, model_scores, eval_model, test, test_pred, metrics, baseline_f1, train_valid = fit_and_evaluate(
            dataset, trees, train_fraction
        )
        latest_model = train_latest(dataset, winner, trees)
        latest_row = latest_features.iloc[[-1]]
        latest_probs_map = model_probabilities(latest_model, latest_row)
        latest_class = max(latest_probs_map, key=latest_probs_map.get)
        confidence = latest_probs_map[latest_class]
        signal = "HOLD"
        if confidence >= confidence_threshold:
            if latest_class == 2:
                signal = "BUY"
            elif latest_class == 0:
                signal = "SELL"
        bt, bt_metrics = run_backtest(test, test_pred, cost_bps, slippage_bps)
except Exception as exc:
    st.error(f"Unable to build the analytics workspace: {exc}")
    st.info("Try a longer historical window, a slightly wider/narrower class boundary, or refresh the Yahoo Finance data.")
    st.stop()

last_close = float(raw.Close.iloc[-1])
prev_close = float(raw.Close.iloc[-2]) if len(raw) > 1 else last_close
daily_change = (last_close / prev_close - 1) * 100 if prev_close else 0
last_date = raw.index[-1].strftime("%d %b %Y")
probabilities = {CLASS_NAMES[k]: latest_probs_map.get(k, 0.0) for k in LABELS}
signal_class = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}[signal]

st.markdown(f"""
<div class="hero">
  <div class="eyebrow">ASTRAQUANT X 2026 · QUANTITATIVE RESEARCH</div>
  <div class="hero-title">NIFTY 50 <span style="color:#45d9ab">/</span> AI TREND INTELLIGENCE</div>
  <div class="hero-sub">Short-horizon classification · Technical feature engine · Chronological model validation</div>
</div>
""", unsafe_allow_html=True)

# Top-level data provenance and current status
st.caption(f"Latest available completed daily candle: {last_date}  ·  Source: Yahoo Finance (^NSEI)  ·  Model selected on validation: {winner}")
if page == "Overview":
    st.markdown('<div class="section-kicker">MARKET SNAPSHOT</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("NIFTY 50 CLOSE", f"{last_close:,.2f}", f"{daily_change:+.2f}%")
    b.metric("AI TREND", CLASS_NAMES[latest_class], f"{confidence:.1%} probability")
    c.metric("MODEL SIGNAL", signal, "Confidence-gated")
    d.metric("MODEL", winner, f"{len(dataset):,} labelled rows")
    st.markdown('<div class="section-kicker">MARKET OVERVIEW</div>', unsafe_allow_html=True)
    left, right = st.columns([1.65, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=raw.index, open=raw.Open, high=raw.High, low=raw.Low, close=raw.Close, name="NIFTY 50"))
        fig.add_trace(go.Scatter(x=engineered.index, y=engineered.ema_21, name="EMA 21", line=dict(color="#4ba3ff", width=1.4)))
        fig.add_trace(go.Scatter(x=engineered.index, y=engineered.ema_50, name="EMA 50", line=dict(color="#e7b957", width=1.4)))
        fig.update_layout(title="Index price action & trend averages", xaxis_rangeslider_visible=False)
        st.plotly_chart(style_chart(fig, 430), use_container_width=True)
    with right:
        pfig = go.Figure(go.Bar(
            x=list(probabilities.values()), y=list(probabilities.keys()), orientation="h",
            marker_color=[CLASS_COLORS[k] for k in probabilities]
        ))
        pfig.update_layout(title="Next-session class probabilities", xaxis=dict(range=[0, 1], tickformat=".0%"))
        st.plotly_chart(style_chart(pfig, 280), use_container_width=True)
        st.markdown(f'<span class="signal {signal_class}">{signal} · {confidence:.1%} CONFIDENCE</span>', unsafe_allow_html=True)
        st.markdown('<div class="note">Signals are generated from the model class and a configurable confidence floor. HOLD is used for sideways predictions or insufficient confidence.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">MODEL HEALTH</div>', unsafe_allow_html=True)
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Final test accuracy", f"{metrics['accuracy']:.1%}")
    x2.metric("Macro F1", f"{metrics['f1']:.1%}")
    x3.metric("Baseline macro F1", f"{baseline_f1:.1%}")
    x4.metric("Test observations", f"{len(test):,}")
    st.info("The test metrics are measured on a later chronological segment that was not used to select the model. They are historical estimates, not a forecast of future performance.")

elif page == "AI Prediction":
    st.markdown('<div class="section-kicker">NEXT-SESSION CLASSIFICATION</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    for col, name in zip([p1, p2, p3], ["BULLISH", "BEARISH", "SIDEWAYS"]):
        col.metric(f"{name} probability", f"{probabilities[name]:.1%}")
    st.markdown(f"### Predicted trend: **{CLASS_NAMES[latest_class]}**")
    st.markdown(f'<span class="signal {signal_class}">{signal} · CONFIDENCE {confidence:.1%}</span>', unsafe_allow_html=True)
    st.caption(f"Configured signal threshold: {confidence_threshold:.0%}. A class probability below this threshold results in HOLD.")
    st.markdown("#### What influenced the prediction?")
    latest_ind = engineered.iloc[-1]
    explanations = [
        ("RSI (14)", f"{latest_ind.rsi_14:.1f}", "Above 70 can indicate strong/overbought momentum; below 30 can indicate weak/oversold momentum."),
        ("MACD histogram", f"{latest_ind.macd_hist:,.2f}", "Positive values indicate MACD is above its signal line; negative values indicate the reverse."),
        ("EMA 9 vs EMA 21", "Above" if latest_ind.ema_9 > latest_ind.ema_21 else "Below", "A short-term trend relationship, not a standalone signal."),
        ("Price vs SMA 20", f"{(last_close/latest_ind.sma_20-1)*100:+.2f}%", "Shows the latest close's distance from its 20-session average."),
        ("10-day momentum", f"{latest_ind.momentum_10d*100:+.2f}%", "Recent price change over ten sessions."),
        ("10-day volatility", f"{latest_ind.volatility_10d*100:.2f}%", "Rolling standard deviation of daily returns."),
    ]
    expl_df = pd.DataFrame(explanations, columns=["Indicator", "Latest reading", "Interpretation"])
    st.dataframe(expl_df, use_container_width=True, hide_index=True)
    st.markdown('<div class="note">These are contextual indicator readings, not a causal explanation of the model. The fitted model estimates class probabilities from the full engineered feature vector.</div>', unsafe_allow_html=True)
    prob_fig = go.Figure(go.Bar(x=list(probabilities.values()), y=list(probabilities.keys()), orientation="h",
                                marker_color=[CLASS_COLORS[k] for k in probabilities], text=[f"{v:.1%}" for v in probabilities.values()], textposition="outside"))
    prob_fig.update_layout(title="Probability distribution", xaxis=dict(range=[0, 1], tickformat=".0%"))
    st.plotly_chart(style_chart(prob_fig, 300), use_container_width=True)

elif page == "Technical Analysis":
    st.markdown('<div class="section-kicker">TECHNICAL INDICATOR SUITE</div>', unsafe_allow_html=True)
    ti = engineered.iloc[-1]
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("RSI (14)", f"{ti.rsi_14:.2f}", "Momentum")
    t2.metric("MACD", f"{ti.macd:.2f}")
    t3.metric("EMA 9 / 21", "Bullish" if ti.ema_9 > ti.ema_21 else "Bearish")
    t4.metric("Volatility (10d)", f"{ti.volatility_10d:.2%}")
    price = go.Figure()
    price.add_trace(go.Scatter(x=engineered.index, y=engineered.Close, name="Close", line=dict(color="#eaf2ff", width=1.5)))
    for col, color in [("ema_9", "#35d49a"), ("ema_21", "#4ba3ff"), ("ema_50", "#e7b957"), ("sma_20", "#c18cff")]:
        price.add_trace(go.Scatter(x=engineered.index, y=engineered[col], name=col.upper(), line=dict(color=color, width=1.2)))
    price.update_layout(title="Price and moving averages")
    st.plotly_chart(style_chart(price, 390), use_container_width=True)
    ind = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.08,
                        subplot_titles=("RSI (14)", "MACD", "Daily volume"))
    ind.add_trace(go.Scatter(x=engineered.index, y=engineered.rsi_14, name="RSI", line=dict(color="#c18cff")), row=1, col=1)
    ind.add_hline(y=70, line_dash="dot", line_color="#ff5d6c", row=1, col=1)
    ind.add_hline(y=30, line_dash="dot", line_color="#35d49a", row=1, col=1)
    ind.add_trace(go.Scatter(x=engineered.index, y=engineered.macd, name="MACD", line=dict(color="#4ba3ff")), row=2, col=1)
    ind.add_trace(go.Scatter(x=engineered.index, y=engineered.macd_signal, name="Signal", line=dict(color="#e7b957")), row=2, col=1)
    ind.add_trace(go.Bar(x=engineered.index, y=engineered.Volume, name="Volume", marker_color="#355779"), row=3, col=1)
    ind.update_layout(height=720, title="Momentum and participation indicators")
    st.plotly_chart(style_chart(ind, 720), use_container_width=True)
    st.markdown("#### Latest indicator readings")
    st.dataframe(pd.DataFrame({
        "Indicator": ["RSI 14", "MACD", "MACD signal", "EMA 9", "EMA 21", "EMA 50", "SMA 20", "Volume", "Volatility 5d", "Volatility 10d", "High-low range"],
        "Value": [ti.rsi_14, ti.macd, ti.macd_signal, ti.ema_9, ti.ema_21, ti.ema_50, ti.sma_20, ti.Volume, ti.volatility_5d, ti.volatility_10d, ti.range_pct]
    }), use_container_width=True, hide_index=True)

elif page == "Model Performance":
    st.markdown('<div class="section-kicker">CHRONOLOGICAL HOLDOUT EVALUATION</div>', unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    q2.metric("Macro precision", f"{metrics['precision']:.1%}")
    q3.metric("Macro recall", f"{metrics['recall']:.1%}")
    q4.metric("Macro F1", f"{metrics['f1']:.1%}")
    st.markdown("#### Candidate model selection")
    st.caption("Candidates are ranked on the validation segment only. Final metrics below are from the later, untouched test segment.")
    st.dataframe(pd.DataFrame(model_scores).sort_values("Validation macro F1", ascending=False), use_container_width=True, hide_index=True)
    st.metric("Simple majority-class baseline · test macro F1", f"{baseline_f1:.1%}")
    cm = confusion_matrix(test.target, test_pred, labels=LABELS)
    cm_fig = go.Figure(go.Heatmap(z=cm, x=[CLASS_NAMES[i] for i in LABELS], y=[CLASS_NAMES[i] for i in LABELS],
                                  text=cm, texttemplate="%{text}", colorscale="Blues"))
    cm_fig.update_layout(title="Confusion matrix · actual rows × predicted columns")
    st.plotly_chart(style_chart(cm_fig, 360), use_container_width=True)
    st.markdown("#### Class-wise report")
    report = classification_report(test.target, test_pred, labels=LABELS,
                                   target_names=[CLASS_NAMES[i] for i in LABELS],
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)
    st.markdown("#### Feature importance / model coefficients")
    underlying = eval_model.named_steps["model"] if isinstance(eval_model, Pipeline) else eval_model
    if hasattr(underlying, "feature_importances_"):
        importance = underlying.feature_importances_
        fi = pd.DataFrame({"Feature": FEATURES, "Importance": importance}).sort_values("Importance")
        ff = go.Figure(go.Bar(x=fi.Importance, y=fi.Feature, orientation="h", marker_color="#4ba3ff"))
        ff.update_layout(title="Selected model feature importance")
        st.plotly_chart(style_chart(ff, 460), use_container_width=True)
    elif hasattr(underlying, "coef_"):
        coef = np.mean(np.abs(underlying.coef_), axis=0)
        fi = pd.DataFrame({"Feature": FEATURES, "Mean absolute coefficient": coef}).sort_values("Mean absolute coefficient")
        ff = go.Figure(go.Bar(x=fi.iloc[:, 1], y=fi.Feature, orientation="h", marker_color="#4ba3ff"))
        ff.update_layout(title="Logistic regression · mean absolute standardized coefficient")
        st.plotly_chart(style_chart(ff, 460), use_container_width=True)
    else:
        st.info("This estimator does not expose a directly comparable native feature-importance measure.")

elif page == "Strategy & Backtest":
    st.markdown('<div class="section-kicker">HISTORICAL SIMULATION · FINAL TEST PERIOD</div>', unsafe_allow_html=True)
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Strategy return", f"{bt_metrics['net_return']:.2f}%")
    r2.metric("Buy & hold", f"{bt_metrics['buy_hold']:.2f}%")
    r3.metric("Win rate", f"{bt_metrics['win_rate']:.1f}%")
    r4.metric("Sharpe (annualized)", f"{bt_metrics['sharpe']:.2f}")
    r5.metric("Max drawdown", f"{bt_metrics['max_drawdown']:.2f}%")
    pf = bt_metrics["profit_factor"]
    st.metric("Profit factor", f"{pf:.2f}" if np.isfinite(pf) else "N/A")
    eq = go.Figure()
    eq.add_trace(go.Scatter(x=bt.index, y=bt.strategy_equity, name="AI strategy", line=dict(color="#35d49a", width=2)))
    eq.add_trace(go.Scatter(x=bt.index, y=bt.buy_hold_equity, name="Buy & hold", line=dict(color="#4ba3ff", width=1.7)))
    eq.update_layout(title="Growth of ₹1 · compounded test-period equity")
    st.plotly_chart(style_chart(eq, 400), use_container_width=True)
    dd = go.Figure(go.Scatter(x=bt.index, y=bt.drawdown * 100, fill="tozeroy", name="Drawdown", line=dict(color="#ff6878")))
    dd.update_layout(title="Strategy drawdown", yaxis_title="Drawdown (%)")
    st.plotly_chart(style_chart(dd, 280), use_container_width=True)
    sig = go.Figure()
    sig.add_trace(go.Scatter(x=bt.index, y=bt.Close, name="NIFTY close", line=dict(color="#dce8f8")))
    for cls, marker, color in [(2, "triangle-up", CLASS_COLORS["BULLISH"]), (0, "triangle-down", CLASS_COLORS["BEARISH"])]:
        part = bt[bt.predicted == cls]
        sig.add_trace(go.Scatter(x=part.index, y=part.Close, mode="markers", name=CLASS_NAMES[cls],
                                 marker=dict(symbol=marker, size=9, color=color)))
    sig.update_layout(title="Historical model signals on the held-out test segment")
    st.plotly_chart(style_chart(sig, 350), use_container_width=True)
    st.markdown("#### Recent test-period predictions")
    recent = bt[["Close", "actual_trend", "predicted_trend", "next_return", "position", "strategy_return"]].tail(20).copy()
    recent["next_return"] = recent.next_return.map(lambda v: f"{v:.2%}")
    recent["strategy_return"] = recent.strategy_return.map(lambda v: f"{v:.2%}")
    recent["position"] = recent.position.map({1.0: "LONG", -1.0: "SHORT", 0.0: "FLAT"})
    st.dataframe(recent.sort_index(ascending=False), use_container_width=True)
    st.markdown('<div class="note">Simulation convention: prediction at close t sets exposure to the close-to-close return t→t+1. Turnover cost plus slippage are deducted when the position changes. Shorting is modelled symmetrically and may not be available or practical for every investor. Taxes, financing, market impact, execution latency and instrument tracking error are excluded.</div>', unsafe_allow_html=True)

elif page == "Methodology":
    st.markdown('<div class="section-kicker">REVIEW 2 · TECHNICAL WALKTHROUGH</div>', unsafe_allow_html=True)
    st.markdown("### 1 · Mathematical target formulation")
    st.latex(r"r_{t+1}=\\frac{C_{t+1}}{C_t}-1")
    st.markdown(rf"""
Trend boundary is configurable at **±{threshold_pct:.2f}%**:
- Bearish (0): \(r_{{t+1}} < -\\tau\)
- Sideways (1): \(-\\tau \\leq r_{{t+1}} \\leq \\tau\)
- Bullish (2): \(r_{{t+1}} > \\tau\)

The final observation, which has no known next-session return, is excluded from supervised training.
""")
    st.markdown("### 2 · Feature engineering")
    st.markdown("""
Features are computed from data available at the close of session *t*: daily/3-day/5-day returns,
5- and 10-session rolling volatility, EMA gaps (9/21 and 21/50), price distance from SMA 20,
RSI 14, normalized MACD and signal/histogram, high-low range, volume percentage change and
10-session momentum. No future-return column is included in the feature list.
""")
    st.latex(r"RSI_t=100-\\frac{100}{1+RS_t},\\qquad RS_t=\\frac{EMA_{14}(gain)}{EMA_{14}(loss)}")
    st.latex(r"MACD_t=EMA_{12}(C_t)-EMA_{26}(C_t)")
    st.markdown("### 3 · Model architecture and selection")
    st.markdown(f"""
The application retains **Random Forest** as the baseline model and compares it with Logistic
Regression and Histogram Gradient Boosting. Candidate models are trained on the earliest
{train_fraction:.0%} of labelled observations and ranked on a later validation segment using
macro F1. The selected candidate is refit on train + validation and evaluated once on the final
chronological test segment. The latest prediction model is then refit on all labelled observations.
Scaling is included in the pipelines where appropriate. No random shuffling is used.
""")
    st.markdown("### 4 · Signal generation")
    st.markdown(f"""
- BULLISH with class probability ≥ {confidence_threshold:.0%} → **BUY**
- BEARISH with class probability ≥ {confidence_threshold:.0%} → **SELL**
- SIDEWAYS or probability below the threshold → **HOLD**

The model's probability is a classifier score, not a guarantee or necessarily a calibrated probability.
""")
    st.markdown("### 5 · Evaluation")
    st.markdown("""
Final test metrics include accuracy, macro precision, macro recall, macro F1, confusion matrix and
class-wise precision/recall/F1. A majority-class baseline is also reported. Macro averaging gives
each class equal weight and is useful when class frequencies differ.
""")
    st.markdown("### 6 · Backtest assumptions")
    st.markdown("""
The test-period prediction at t is applied to the next close-to-close return. A position is long
(+1), flat (0) or short (-1). Position changes incur configurable transaction costs and slippage.
Equity is compounded from daily strategy returns. Sharpe is annualized using √252; it is an
approximation and assumes daily observations. No parameter search uses the final test set.
""")
    st.markdown("### Review 2 evidence map")
    st.dataframe(pd.DataFrame([
        ["Data preprocessing & features", "Duplicate/OHLC validation, missing-value handling, returns, volatility, RSI/MACD/EMA/SMA, volume/range features"],
        ["Mathematical/statistical modelling", "Threshold-based 3-class target, rolling volatility, momentum and indicator formulas"],
        ["AI/ML model development", "Random Forest baseline + Logistic Regression + HistGradientBoosting comparison"],
        ["Trading strategy/signals", "Confidence-gated BUY/SELL/HOLD and position-based simulation"],
        ["Training/testing/optimization", "Chronological train-validation-test split; validation macro-F1 model selection"],
        ["Code & functional progress", "Interactive Streamlit pages, live data fetch, charts, metrics and configurable assumptions"],
    ], columns=["Criterion", "Demonstration evidence"]), use_container_width=True, hide_index=True)

st.divider()
st.caption("ASTRAQUANT X 2026 · Educational research prototype · Yahoo Finance data can be delayed, revised, incomplete or unavailable. This app does not execute trades.")
