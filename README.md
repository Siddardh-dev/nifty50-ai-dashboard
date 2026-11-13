# ASTRAQUANT X — NIFTY 50 AI Trend Intelligence

A Streamlit research dashboard for short-horizon NIFTY 50 trend classification. It uses Yahoo Finance daily data (`^NSEI`), engineered technical features, chronological model selection, confidence-gated signals, and a simplified test-period backtest.

## What's included

- **Overview:** latest available completed candle, daily change, predicted class, model confidence, price chart and probabilities.
- **AI Prediction:** bullish/bearish/sideways probabilities, confidence-gated BUY/SELL/HOLD and indicator context.
- **Technical Analysis:** RSI(14), MACD/signal, EMA 9/21/50, SMA 20, volume, rolling volatility and price charts.
- **Model Performance:** Random Forest baseline compared with Logistic Regression and Histogram Gradient Boosting; chronological validation selection; untouched final test metrics, confusion matrix, class report and feature importance where supported.
- **Strategy & Backtest:** test-period strategy and buy-and-hold equity, drawdown, win rate, profit factor, Sharpe and historical signals.
- **Methodology:** mathematical target, features, training protocol, signal rules, backtest assumptions and Review 2 evidence map.

## Installation — macOS / VS Code

1. Install Python 3.10 or newer.
2. Open this folder in VS Code.
3. Open Terminal → New Terminal.
4. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Upgrade pip and install packages:

   ```bash
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements.txt
   ```

6. Launch the dashboard:

   ```bash
   python3 -m streamlit run nifty50_app.py
   ```

7. Streamlit will print a local URL, normally `http://localhost:8501`. Open it in your browser.

If `pip` is reported as “command not found”, use `python3 -m pip` as shown above. In VS Code, select the `.venv` interpreter via **Cmd+Shift+P → Python: Select Interpreter**.

## Data source and refresh

The app downloads daily adjusted OHLCV data for Yahoo Finance ticker `^NSEI`. The sidebar lets you choose a history window and **Refresh data & retrain**. Data can be delayed, revised, incomplete, or unavailable; Yahoo Finance is not an exchange-grade feed. The latest value shown is the last available daily close, not a guaranteed live quote.

## ML architecture

1. Construct indicators and lagged/rolling features from data available at close *t*.
2. Define the label from close-to-close return at *t+1*: bearish below `-threshold`, bullish above `+threshold`, otherwise sideways.
3. Drop rows without a known future label from supervised training. Keep the latest feature row separately for inference.
4. Split labelled rows chronologically into training, validation and final test segments.
5. Compare Random Forest, Logistic Regression and Histogram Gradient Boosting on validation macro-F1.
6. Refit the selected estimator on train + validation and report final test metrics once.
7. Refit on all labelled rows for the latest next-session prediction.

The model is not guaranteed to outperform a baseline. The UI reports measured results for the currently downloaded dataset and current settings.

## Mathematical methodology

For close price `C_t`, next-session return is `r_(t+1) = C_(t+1) / C_t - 1`. With configurable boundary `τ`, label is bearish for `r < -τ`, sideways for `-τ ≤ r ≤ τ`, and bullish for `r > τ`.

Features include 1/3/5-session returns, rolling 5/10-session volatility, EMA gaps, distance from SMA20, RSI14, MACD and signal/histogram normalized by close, high-low range, volume change and 10-session momentum. No future-return field is included in the feature vector.

## Backtest assumptions and limitations

A prediction formed at close `t` is assigned to the `t → t+1` close-to-close return. Bullish maps to +1 exposure, bearish to -1, sideways to 0. A configurable transaction-cost plus slippage charge is applied to absolute position changes. This is a simplified research simulation: it omits taxes, financing, market impact, order execution, latency, instrument tracking error and realistic short-sale constraints. Sharpe uses √252 and is approximate. Historical performance is not predictive.

## Review presentation demo guide

1. **Problem framing (30 sec):** explain the next-session, three-class NIFTY trend prediction task.
2. **Data & preprocessing (45 sec):** show data source, date range, validation rules and technical features.
3. **Mathematical model (45 sec):** explain future return threshold and the three labels.
4. **ML architecture (60 sec):** explain RF baseline and model comparison; show chronological train/validation/test split.
5. **Evaluation (60 sec):** show final test accuracy, macro F1, class-wise report and confusion matrix. Compare the majority-class baseline.
6. **Signal logic (45 sec):** adjust confidence threshold and show BUY/SELL/HOLD behavior.
7. **Backtest (60 sec):** explain prediction/return alignment, costs, equity curve, buy-and-hold and drawdown.
8. **Limitations (30 sec):** mention Yahoo data limitations, probability calibration, regime changes and omitted execution costs.

## Questions judges may ask

**Why use a three-class target?**  
It distinguishes meaningful directional moves from small changes within a configurable neutral band.

**Why chronological splitting instead of random splitting?**  
Time-series observations are ordered. Random shuffling can let future regimes influence training and overstate generalization.

**How do you avoid leakage?**  
Features at time `t` use only current and historical market observations. The target uses `t+1` return and is excluded from the feature list. The last unlabeled observation is not used for supervised fitting.

**Why macro F1?**  
It weights all three classes equally, reducing the dominance of a frequent class in the headline score.

**Why compare models?**  
It tests whether a simpler linear model or a boosting model generalizes better than the retained Random Forest baseline, using validation data only.

**Does confidence mean certainty?**  
No. It is the estimator's class probability output and may not be calibrated. It is used only as a configurable gate.

**Can this place trades?**  
No. It is an educational analytics and backtesting prototype, not a brokerage-connected execution system.

**What are the backtest limitations?**  
Simplified close-to-close fills, assumed symmetric short exposure, fixed friction per position change, and omission of taxes, financing, market impact and execution constraints.

**What would you improve next?**  
Add walk-forward evaluation, probability calibration, more robust market-regime analysis, benchmark comparisons and execution-aware paper trading before considering any real-world deployment.
