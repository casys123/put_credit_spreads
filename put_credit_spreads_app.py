# Create a ready-to-deploy GitHub-style project for Streamlit Cloud
import os, textwrap, json, pandas as pd

project_dir = "/mnt/data/put-credit-spreads-app"
os.makedirs(project_dir, exist_ok=True)

app_py = r'''
# put_credit_spreads_app.py
# Streamlit app to screen weekly/biweekly Put Credit Spreads with ≥65% POP, avoiding earnings & ex-div when possible.
# Data: Yahoo Finance via yfinance (free); optional calendars via FMP/Finnhub if you add free keys in Streamlit Secrets.
# Deploy: push this file + requirements.txt to GitHub and point Streamlit Cloud to the repo (main file = put_credit_spreads_app.py)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta, timezone
import math

st.set_page_config(page_title="Put Credit Spread Finder", layout="wide")

st.title("📉 Put Credit Spread Finder (Weekly/Biweekly)")
st.caption("Targets ≥ 65% POP, avoids earnings & ex-div dates when possible. Uses free sources: Yahoo (yfinance); optional FMP/Finnhub keys via Streamlit Secrets.")

# ------------------------------
# Helpers
# ------------------------------

def parse_tickers(raw: str):
    if not raw:
        return []
    parts = [p.strip().upper() for p in raw.replace("\\n", ",").replace(" ", ",").split(",")]
    return sorted(list({p for p in parts if p}))

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1 * delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def pop_above_breakeven(S0: float, B: float, iv: float, T: float, r: float = 0.0) -> float:
    # Approx probability S_T > B under GBM with drift r - 0.5*iv^2
    if S0 <= 0 or B <= 0 or iv <= 0 or T <= 0:
        return float("nan")
    mu = r - 0.5 * iv * iv
    z = (math.log(S0 / B) + mu * T) / (iv * math.sqrt(T))
    return 1.0 - norm_cdf(-z)

def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) - 1.0  # put delta

@st.cache_data(show_spinner=False)
def get_hist(ticker: str, period: str = "1y") -> pd.DataFrame:
    return yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)

def expected_move_from_chain(chain_calls: pd.DataFrame, chain_puts: pd.DataFrame, S0: float) -> float:
    if chain_calls is None or chain_puts is None or chain_calls.empty or chain_puts.empty:
        return float("nan")
    calls = chain_calls.copy()
    puts = chain_puts.copy()
    calls["dist"] = (calls["strike"] - S0).abs()
    puts["dist"] = (puts["strike"] - S0).abs()
    try:
        c = calls.sort_values("dist").iloc[0]
        p = puts.sort_values("dist").iloc[0]
        def mid_or_last(row):
            bid = row.get("bid", float("nan"))
            ask = row.get("ask", float("nan"))
            last = row.get("lastPrice", float("nan"))
            if pd.notna(bid) and pd.notna(ask):
                return (bid + ask) / 2
            return last
        cpx = mid_or_last(c)
        ppx = mid_or_last(p)
        if pd.isna(cpx) or pd.isna(ppx):
            return float("nan")
        straddle = cpx + ppx
        return 0.85 * straddle
    except Exception:
        return float("nan")

def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

def fmp_next_exdiv(ticker: str, key: str):
    if not key:
        return None
    try:
        url = f"https://financialmodelingprep.com/stable/dividends-calendar?symbol={ticker}&apikey={key}"
        data = requests.get(url, timeout=10).json()
        if isinstance(data, list) and data:
            today = datetime.now(timezone.utc).date()
            fut = [pd.to_datetime(x.get("exDividendDate", "")).date()
                   for x in data if x.get("exDividendDate")]
            fut = [d for d in fut if d and d >= today]
            if fut:
                return min(fut)
    except Exception:
        return None
    return None

def fmp_next_earnings(ticker: str, key: str):
    if not key:
        return None
    try:
        url = f"https://financialmodelingprep.com/api/v3/earnings-calendar?symbol={ticker}&limit=10&apikey={key}"
        data = requests.get(url, timeout=10).json()
        if isinstance(data, list) and data:
            today = datetime.now(timezone.utc).date()
            fut = [pd.to_datetime(x.get("date","")).date()
                   for x in data if x.get("date")]
            fut = [d for d in fut if d and d >= today]
            if fut:
                return min(fut)
    except Exception:
        return None
    return None

def finnhub_next_earnings(ticker: str, key: str):
    if not key:
        return None
    try:
        url = f"https://finnhub.io/api/v1/calendar/earnings?symbol={ticker}&from={datetime.utcnow().date()}&to={datetime.utcnow().date()+timedelta(days=120)}&token={key}"
        data = requests.get(url, timeout=10).json()
        items = data.get("earningsCalendar", [])
        if items:
            fut = [pd.to_datetime(x.get("date","")).date() for x in items if x.get("date")]
            fut = [d for d in fut if d and d >= datetime.utcnow().date()]
            if fut:
                return min(fut)
    except Exception:
        return None
    return None

def find_spreads_for_ticker(ticker: str,
                            pop_target: float,
                            dte_windows: list,
                            max_width: float,
                            min_credit: float,
                            avoid_earnings: bool,
                            avoid_exdiv: bool,
                            use_mid: bool,
                            r: float,
                            fmp_key: str,
                            finnhub_key: str):
    tk = yf.Ticker(ticker)
    hist = get_hist(ticker, "1y")
    if hist is None or hist.empty:
        return []
    spot = float(hist["Close"].iloc[-1])
    rv20 = float(hist["Close"].pct_change().rolling(20).std().iloc[-1] * (252 ** 0.5))
    rsi14 = float(rsi(hist["Close"]).iloc[-1])
    ma50 = float(hist["Close"].rolling(50).mean().iloc[-1])
    trend = "uptrend" if hist["Close"].iloc[-1] > ma50 else "down/sideways"
    today = datetime.utcnow().date()

    next_earn = fmp_next_earnings(ticker, fmp_key) or finnhub_next_earnings(ticker, finnhub_key)
    next_exdiv = fmp_next_exdiv(ticker, fmp_key)

    results = []
    try:
        expirations = tk.options
    except Exception:
        expirations = []

    if not expirations:
        return results

    ranges = []
    if "7-10" in dte_windows:
        ranges.append((7, 10))
    if "12-16" in dte_windows:
        ranges.append((12, 16))

    for exp in expirations:
        try:
            edate = pd.to_datetime(exp).date()
        except Exception:
            continue
        dte = (edate - today).days
        if not any(lo <= dte <= hi for lo, hi in ranges):
            continue

        try:
            opt = tk.option_chain(exp)
            calls, puts = opt.calls.copy(), opt.puts.copy()
        except Exception:
            continue

        exp_move = expected_move_from_chain(calls, puts, spot)

        for _, row in puts.iterrows():
            try:
                K = float(row["strike"])
                if K >= spot:
                    continue
                iv = float(row.get("impliedVolatility", float("nan")))
                if not np.isfinite(iv) or iv <= 0:
                    continue
                T = max(dte / 365.0, 1e-6)
                delta = bs_put_delta(spot, K, T, r, iv)
                if np.isnan(delta) or delta < -0.35 or delta > -0.10:
                    continue

                below = puts[(puts["strike"] < K) & (puts["strike"] >= K - max_width)]
                if below.empty:
                    continue
                L = float(below["strike"].max())

                short_bid = float(row.get("bid", float("nan")))
                short_ask = float(row.get("ask", float("nan")))
                long_row = below[below["strike"] == L].iloc[0]
                long_bid = float(long_row.get("bid", float("nan")))
                long_ask = float(long_row.get("ask", float("nan")))
                if any(pd.isna(x) for x in [short_bid, short_ask, long_bid, long_ask]):
                    continue

                credit = ((short_bid + short_ask) / 2) - ((long_bid + long_ask) / 2) if use_mid else (short_bid - long_ask)
                if credit < min_credit:
                    continue

                width = K - L
                max_loss = width - credit
                if max_loss <= 0:
                    continue

                roi = credit / max_loss
                breakeven = K - credit
                pop = pop_above_breakeven(spot, breakeven, iv, T, r=r)
                if not np.isfinite(pop) or pop < pop_target:
                    continue

                avoid_note = []
                if avoid_earnings and next_earn and (today <= next_earn <= edate):
                    avoid_note.append("earnings")
                if avoid_exdiv and next_exdiv and (today <= next_exdiv <= edate):
                    avoid_note.append("ex-div")

                results.append({
                    "Ticker": ticker,
                    "DTE": int(dte),
                    "Expiry": edate.isoformat(),
                    "Spot": round(spot, 2),
                    "RV20": round(rv20, 3) if np.isfinite(rv20) else None,
                    "RSI14": round(rsi14, 1) if np.isfinite(rsi14) else None,
                    "Trend": trend,
                    "ExpectedMove~": round(exp_move, 2) if np.isfinite(exp_move) else None,
                    "ShortK": K,
                    "LongK": L,
                    "Width": width,
                    "Credit": round(credit, 2),
                    "MaxLoss": round(max_loss, 2),
                    "ROI": round(roi, 3),
                    "POP": round(pop, 3),
                    "IV_used": round(iv, 3),
                    "Avoid": ", ".join(avoid_note) if avoid_note else "",
                    "SuggestOpen": "Today" if (np.isfinite(rv20) and np.isfinite(iv) and iv > rv20 and rsi14 < 65) else "Wait for higher IV / pullback",
                    "SuggestClose": (edate - timedelta(days=3)).isoformat() + " (≈ 3 DTE) or 50% profit"
                })
            except Exception:
                continue

    results = sorted(results, key=lambda x: (x["Avoid"] != "", -x["ROI"], -x["POP"]))
    return results

# ------------------------------
# Sidebar controls
# ------------------------------

with st.sidebar:
    st.header("Settings")
    tickers = st.text_area("Symbols (comma/space/newline separated)", "SPY, AAPL, MSFT").upper()
    pop_target = st.slider("Minimum POP", 0.50, 0.90, 0.65, 0.01)
    max_width = st.number_input("Max spread width ($)", 1.0, 50.0, 5.0, 0.5)
    min_credit = st.number_input("Minimum net credit ($)", 0.01, 10.0, 0.25, 0.05)
    dte_choices = st.multiselect("Target DTE windows", ["7-10", "12-16"], default=["7-10", "12-16"])
    avoid_earnings = st.checkbox("Avoid earnings during holding window", True)
    avoid_exdiv = st.checkbox("Avoid ex-dividend during holding window", True)
    use_mid_prices = st.checkbox("Use mid prices (bid/ask midpoint)", True)
    risk_free = st.number_input("Risk-free rate (annualized)", 0.0, 0.10, 0.03, 0.005, format="%.3f")

    st.markdown("---")
    st.caption("Optional API keys via **Settings → Secrets** in Streamlit Cloud:")
    st.code('''
# Streamlit Secrets (example)
FMP_KEY = "YOUR_FMP_KEY"
FINNHUB_KEY = "YOUR_FINNHUB_KEY"
'''.strip())

symbols = parse_tickers(tickers)

# Read optional secrets (don't crash if not provided)
FMP_KEY = get_secret("FMP_KEY", "")
FINNHUB_KEY = get_secret("FINNHUB_KEY", "")

rows = []
if symbols:
    progress = st.progress(0.0, text="Scanning chains...")
    for i, sym in enumerate(symbols, start=1):
        try:
            rows.extend(
                find_spreads_for_ticker(
                    sym, pop_target, dte_choices, max_width, min_credit,
                    avoid_earnings, avoid_exdiv, use_mid_prices, risk_free,
                    FMP_KEY, FINNHUB_KEY
                )
            )
        except Exception as e:
            st.warning(f"{sym}: {e}")
        progress.progress(i / len(symbols), text=f"Processed {i}/{len(symbols)}")

df = pd.DataFrame(rows)
if df.empty:
    st.info("No candidates met your filters. Try widening strikes, lowering POP threshold, or disabling some avoid rules.")
else:
    fmt = df.copy()
    money_cols = ["Credit", "MaxLoss"]
    for c in money_cols:
        fmt[c] = fmt[c].map(lambda x: f"${x:,.2f}")
    fmt["ROI"] = fmt["ROI"].map(lambda x: f"{x*100:.1f}%")
    fmt["POP"] = fmt["POP"].map(lambda x: f"{x*100:.1f}%")
    fmt["IV_used"] = fmt["IV_used"].map(lambda x: f"{x*100:.1f}%")
    cols = ["Ticker","DTE","Expiry","Spot","ExpectedMove~","ShortK","LongK","Width","Credit","MaxLoss","ROI","POP","IV_used","RV20","RSI14","Trend","Avoid","SuggestOpen","SuggestClose"]
    st.dataframe(fmt[cols], use_container_width=True, height=520)

    # Downloads
    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="put_credit_spreads_results.csv",
        mime="text/csv",
    )
    st.download_button(
        "⬇️ Download JSON",
        data=df.to_json(orient="records").encode("utf-8"),
        file_name="put_credit_spreads_results.json",
        mime="application/json",
    )

st.markdown("---")
st.subheader("🔎 Notes & Methodology")
st.markdown("""
- **POP**: Probability the underlying finishes **above breakeven** at expiration under a lognormal model using the short leg IV. This is an estimate, not a guarantee.
- **Credit/Max Loss**: Uses mid-prices when enabled; otherwise conservative bid/ask. ROI = Credit / (Width - Credit).
- **Expected Move~**: Approximate 1-week expected move = ATM straddle × 0.85 (rule of thumb) from the current chain.
- **Filters**: Short-put delta (Black–Scholes) between ~−0.10 and −0.35. Avoid flags if earnings/ex-div fall inside the holding window.
- **Timing**: Open when **IV > RV20** and price is above 50-DMA but after a minor dip (RSI < 65). Close around **~3 DTE or 50% profit** to reduce gamma risk.
- **APIs**: Price/chain via `yfinance` (Yahoo Finance). Optional calendars via FMP and/or Finnhub if keys are supplied.
- **Disclaimer**: Educational only. Double-check data in your broker before trading.
""")
'''

requirements_txt = """streamlit
yfinance
pandas
numpy
requests
"""

readme_md = """
# Put Credit Spread Finder (Streamlit)

A Streamlit app that screens **weekly / biweekly** put credit spreads with a **≥ 65% POP** target, and optionally avoids **earnings** and **ex-dividend** dates using free APIs.

## Deploy to Streamlit Cloud (from GitHub)
1. Create a new GitHub repo and add:
   - `put_credit_spreads_app.py`
   - `requirements.txt`
   - (optional) `README.md`
2. On [share.streamlit.io](https://share.streamlit.io) → **New app**, select your repo.
3. **Main file**: `put_credit_spreads_app.py`
4. (Optional) **Secrets** → add:
   ```toml
   FMP_KEY = "YOUR_FMP_KEY"
   FINNHUB_KEY = "YOUR_FINNHUB_KEY"
