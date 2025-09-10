# Create an upgraded Streamlit app with more controls, liquidity filters, IV proxies, and better suggestions
from datetime import datetime
import os, textwrap

improved_code = r'''
# put_credit_spreads_app.py — v2 (Enhanced)
# Improvements:
# - Configurable delta band
# - Liquidity filters (min OI, min volume, max bid-ask %)
# - Better IV proxy (near-ATM blended IV) + IV Percentile (proxy) over last 60 sessions using HV as fallback
# - Upload tickers via CSV or paste list
# - Optional export of "trade plan" with entry/exit guidance
# - Per‑ticker details in expanders (top candidates by ROI/POP)
# - Robust error handling & caching

import io
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Put Credit Spread Finder (Enhanced)", layout="wide")

st.title("📉 Put Credit Spread Finder — Enhanced (Weekly/Biweekly)")
st.caption("Targets ≥ 65% POP, adds liquidity filters, IV proxy & percentile, uploadable tickers, and trade‑plan exports. Data via Yahoo (yfinance); optional calendars via FMP/Finnhub.")

# ------------------------------
# Helpers
# ------------------------------

def parse_tickers_text(raw: str):
    if not raw:
        return []
    parts = [p.strip().upper() for p in raw.replace("\\n", ",").replace(" ", ",").split(",")]
    return sorted(list({p for p in parts if p}))

def parse_ticker_file(file) -> list:
    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
            col = [c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()]
            if col:
                vals = df[col[0]].astype(str).tolist()
            else:
                # assume single column of tickers
                vals = df.iloc[:,0].astype(str).tolist()
            return sorted(list({v.strip().upper() for v in vals if isinstance(v, str) and v.strip()}))
        elif file.name.lower().endswith(".txt"):
            txt = file.read().decode("utf-8", errors="ignore")
            return parse_tickers_text(txt)
    except Exception:
        pass
    return []

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

def realized_vol(series: pd.Series, lookback=20) -> float:
    rv = series.pct_change().rolling(lookback).std().iloc[-1] * (252 ** 0.5)
    return float(rv) if pd.notna(rv) else float("nan")

def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) - 1.0

def pop_above_breakeven(S0: float, B: float, iv: float, T: float, r: float = 0.0) -> float:
    if S0 <= 0 or B <= 0 or iv <= 0 or T <= 0:
        return float("nan")
    mu = r - 0.5 * iv * iv
    z = (math.log(S0 / B) + mu * T) / (iv * math.sqrt(T))
    return 1.0 - norm_cdf(-z)

@st.cache_data(show_spinner=False)
def get_hist(ticker: str, period: str = "1y"):
    return yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)

def iv_proxy_from_chain(calls: pd.DataFrame, puts: pd.DataFrame, S0: float) -> float:
    """Blend near-ATM option IVs (calls & puts) to estimate current IV level."""
    if calls is None or puts is None or calls.empty or puts.empty:
        return float("nan")
    try:
        calls = calls.assign(dist=(calls["strike"] - S0).abs()).sort_values("dist").head(3)
        puts = puts.assign(dist=(puts["strike"] - S0).abs()).sort_values("dist").head(3)
        ivs = []
        for _, r in pd.concat([calls, puts]).iterrows():
            iv = float(r.get("impliedVolatility", float("nan")))
            if np.isfinite(iv) and iv > 0:
                ivs.append(iv)
        return float(np.median(ivs)) if ivs else float("nan")
    except Exception:
        return float("nan")

def expected_move_from_chain(chain_calls: pd.DataFrame, chain_puts: pd.DataFrame, S0: float):
    if chain_calls is None or chain_puts is None or chain_calls.empty or chain_puts.empty:
        return float("nan")
    calls = chain_calls.copy()
    puts = chain_puts.copy()
    calls["dist"] = (calls["strike"] - S0).abs()
    puts["dist"]  = (puts["strike"] - S0).abs()
    try:
        c = calls.sort_values("dist").iloc[0]
        p = puts.sort_values("dist").iloc[0]
        def mid_or_last(row):
            bid = row.get("bid", float("nan")); ask = row.get("ask", float("nan")); last = row.get("lastPrice", float("nan"))
            if pd.notna(bid) and pd.notna(ask):
                return (bid + ask) / 2
            return last
        cpx = mid_or_last(c); ppx = mid_or_last(p)
        if pd.isna(cpx) or pd.isna(ppx):
            return float("nan")
        return 0.85 * (cpx + ppx)
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

def iv_percentile_proxy(series: pd.Series, value: float) -> float:
    """Percent of past values <= current 'value' (0-1)."""
    try:
        arr = np.array(series.dropna()[-60:])  # lookback ~ 60 sessions
        if len(arr) < 10 or not np.isfinite(value):
            return float("nan")
        return float((arr <= value).mean())
    except Exception:
        return float("nan")

def bid_ask_spread_pct(bid: float, ask: float) -> float:
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        return float("inf")
    mid = 0.5*(bid+ask)
    return (ask - bid) / mid if mid > 0 else float("inf")

# ------------------------------
# Sidebar & inputs
# ------------------------------

with st.sidebar:
    st.header("Universe")
    tickers_text = st.text_area("Symbols (paste)", "SPY, AAPL, MSFT").upper()
    upfile = st.file_uploader("…or upload CSV/TXT with tickers", type=["csv","txt"])

    st.header("Trade Rules")
    pop_target = st.slider("Minimum POP", 0.50, 0.90, 0.65, 0.01)
    delta_min, delta_max = st.slider("Short Put Delta band (negative)", -0.40, -0.05, (-0.35, -0.10), 0.01)
    max_width = st.number_input("Max spread width ($)", 1.0, 50.0, 5.0, 0.5)
    min_credit = st.number_input("Minimum net credit ($)", 0.01, 10.0, 0.25, 0.05)
    dte_choices = st.multiselect("Target DTE windows", ["7-10", "12-16"], default=["7-10","12-16"])
    manage_early = st.checkbox("Plan to manage at 50% profit", True)

    st.header("Liquidity Filters")
    min_oi = st.number_input("Min Open Interest (short & long)", 0, 5000, 50, 10)
    min_vol = st.number_input("Min Today Volume (short & long)", 0, 5000, 0, 10)
    max_spread_pct = st.number_input("Max bid‑ask width (% of mid)", 1, 100, 25, 1)

    st.header("Risk & Calendars")
    avoid_earnings = st.checkbox("Avoid earnings during holding window", True)
    avoid_exdiv = st.checkbox("Avoid ex‑dividend during holding window", True)
    use_mid_prices = st.checkbox("Use mid prices (bid/ask midpoint)", True)
    risk_free = st.number_input("Risk‑free rate (annualized)", 0.0, 0.10, 0.03, 0.005, format="%.3f")

    st.header("Optional Secrets")
    st.caption("Add in Streamlit Cloud → Settings → Secrets")
   st.code("FMP_KEY = \"YOUR_FMP_KEY\"\nFINNHUB_KEY = \"YOUR_FINNHUB_KEY\"")

FINNHUB_KEY = "YOUR_FINNHUB_KEY"''')

# Ticker universe
symbols = parse_tickers_text(tickers_text)
if upfile is not None:
    symbols = sorted(list(set(symbols + parse_ticker_file(upfile))))

FMP_KEY = get_secret("FMP_KEY", "")
FINNHUB_KEY = get_secret("FINNHUB_KEY", "")

# ------------------------------
# Core scanner
# ------------------------------

def find_spreads_for_ticker(ticker: str):
    out = {"rows": [], "detail": {}}
    try:
        hist = get_hist(ticker, "1y")
        if hist is None or hist.empty:
            return out
        spot = float(hist["Close"].iloc[-1])
        rv20 = realized_vol(hist["Close"], 20)
        rsi14 = float(rsi(hist["Close"]).iloc[-1])
        ma50 = float(hist["Close"].rolling(50).mean().iloc[-1])
        trend = "uptrend" if hist["Close"].iloc[-1] > ma50 else "down/sideways"
        today = datetime.utcnow().date()

        tk = yf.Ticker(ticker)
        try:
            expirations = tk.options
        except Exception:
            expirations = []

        # map windows
        ranges = []
        if "7-10" in dte_choices: ranges.append((7,10))
        if "12-16" in dte_choices: ranges.append((12,16))

        # calendars
        next_earn = fmp_next_earnings(ticker, FMP_KEY) or finnhub_next_earnings(ticker, FINNHUB_KEY)
        next_exdiv = fmp_next_exdiv(ticker, FMP_KEY)

        # keep some per‑ticker context
        best = []

        for exp in expirations:
            edate = pd.to_datetime(exp).date()
            dte = (edate - today).days
            if not any(lo <= dte <= hi for lo,hi in ranges):
                continue

            try:
                chain = tk.option_chain(exp)
                calls, puts = chain.calls.copy(), chain.puts.copy()
            except Exception:
                continue

            iv_now = iv_proxy_from_chain(calls, puts, spot)
            exp_move = expected_move_from_chain(calls, puts, spot)

            # IV percentile proxy using RV history as fallback (not perfect but indicative)
            hv_series = hist["Close"].pct_change().rolling(20).std()*(252**0.5)
            ivp = iv_percentile_proxy(hv_series, iv_now)  # proxy

            # scan puts
            for _, row in puts.iterrows():
                try:
                    K = float(row["strike"])
                    if K >= spot:
                        continue
                    iv = float(row.get("impliedVolatility", float("nan")))
                    if not np.isfinite(iv) or iv <= 0:
                        continue
                    T = max(dte/365.0, 1e-6)
                    delta = bs_put_delta(spot, K, T, risk_free, iv)
                    if not np.isfinite(delta) or delta < delta_min or delta > delta_max:
                        continue

                    # Liquidity
                    sbid, sask = float(row.get("bid", np.nan)), float(row.get("ask", np.nan))
                    soi = int(row.get("openInterest", 0)); svol = int(row.get("volume", 0))
                    if any(pd.isna(x) for x in [sbid, sask]) or soi < min_oi or svol < min_vol:
                        continue
                    if bid_ask_spread_pct(sbid, sask) * 100 > max_spread_pct:
                        continue

                    below = puts[(puts["strike"] < K) & (puts["strike"] >= K - max_width)]
                    if below.empty:
                        continue
                    L = float(below["strike"].max())
                    lrow = below[below["strike"]==L].iloc[0]
                    lbid, lask = float(lrow.get("bid", np.nan)), float(lrow.get("ask", np.nan))
                    loi = int(lrow.get("openInterest", 0)); lvol = int(lrow.get("volume", 0))
                    if any(pd.isna(x) for x in [lbid, lask]) or loi < min_oi or lvol < min_vol:
                        continue
                    if bid_ask_spread_pct(lbid, lask) * 100 > max_spread_pct:
                        continue

                    credit = ( (sbid + sask)/2 - (lbid + lask)/2 ) if use_mid_prices else (sbid - lask)
                    if credit < min_credit:
                        continue
                    width = K - L
                    max_loss = width - credit
                    if max_loss <= 0:
                        continue

                    roi = credit / max_loss
                    breakeven = K - credit
                    pop = pop_above_breakeven(spot, breakeven, iv, T, r=risk_free)
                    if not np.isfinite(pop) or pop < pop_target:
                        continue

                    avoid_note = []
                    if avoid_earnings and next_earn and (today <= next_earn <= edate):
                        avoid_note.append("earnings")
                    if avoid_exdiv and next_exdiv and (today <= next_exdiv <= edate):
                        avoid_note.append("ex-div")

                    open_hint = "Today" if (np.isfinite(iv_now) and np.isfinite(rv20) and iv_now > rv20 and 40 <= rsi14 <= 65) else "Wait for higher IV / better pullback"
                    close_hint = (edate - timedelta(days=3)).isoformat() + (" (≈ 3 DTE)" + (" or 50% profit" if manage_early else ""))

                    rowd = {
                        "Ticker": ticker, "DTE": int(dte), "Expiry": edate.isoformat(), "Spot": round(spot,2),
                        "ShortK": K, "LongK": L, "Width": width, "Credit": round(credit,2),
                        "MaxLoss": round(max_loss,2), "ROI": round(roi,3), "POP": round(pop,3),
                        "IV_used": round(iv,3), "IV_proxy": round(iv_now,3) if np.isfinite(iv_now) else None,
                        "IVP_proxy": round(ivp,3) if np.isfinite(ivp) else None,
                        "RV20": round(rv20,3) if np.isfinite(rv20) else None, "RSI14": round(rsi14,1) if np.isfinite(rsi14) else None,
                        "Trend": trend, "ExpectedMove~": round(exp_move,2) if np.isfinite(exp_move) else None,
                        "Avoid": ", ".join(avoid_note) if avoid_note else "",
                        "SuggestOpen": open_hint, "SuggestClose": close_hint
                    }
                    out["rows"].append(rowd)
                    best.append(rowd)
                except Exception:
                    continue

        # keep top 5 by ROI then POP
        out["detail"]["best"] = sorted(best, key=lambda x: (x["Avoid"]!="", -x["ROI"], -x["POP"]))[:5]
        out["detail"]["context"] = {
            "rv20": rv20, "rsi14": rsi14, "trend": trend
        }
        return out
    except Exception:
        return out

rows = []
per_ticker = {}

if not symbols:
    st.info("Add tickers in the sidebar (paste or upload).")
else:
    progress = st.progress(0.0, text="Scanning…")
    for i, sym in enumerate(symbols, start=1):
        res = find_spreads_for_ticker(sym)
        rows.extend(res["rows"])
        per_ticker[sym] = res["detail"]
        progress.progress(i/len(symbols), text=f"{i}/{len(symbols)} processed")

df = pd.DataFrame(rows)
if df.empty:
    st.warning("No candidates met your filters. Try widening strikes, lowering POP threshold, relaxing liquidity filters, or disabling some avoid rules.")
else:
    # pretty table
    fmt = df.copy()
    for c in ["Credit","MaxLoss"]:
        fmt[c] = fmt[c].map(lambda x: f"${x:,.2f}")
    for c in ["ROI","POP","IV_used","IV_proxy","IVP_proxy"]:
        fmt[c] = fmt[c].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "")
    cols = ["Ticker","DTE","Expiry","Spot","ExpectedMove~","ShortK","LongK","Width","Credit","MaxLoss","ROI","POP","IV_used","IV_proxy","IVP_proxy","RV20","RSI14","Trend","Avoid","SuggestOpen","SuggestClose"]
    st.dataframe(fmt[cols], use_container_width=True, height=520)

    # downloads
    st.download_button("⬇️ Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="put_credit_spreads_results.csv", mime="text/csv")
    st.download_button("⬇️ Download JSON", data=df.to_json(orient="records").encode("utf-8"),
                       file_name="put_credit_spreads_results.json", mime="application/json")

    # trade plan export
    plan_cols = ["Ticker","Expiry","ShortK","LongK","Width","Credit","MaxLoss","ROI","POP","SuggestOpen","SuggestClose","Avoid"]
    plan = df[plan_cols].sort_values(["Ticker","Expiry"])
    st.download_button("📝 Download Trade Plan (CSV)", data=plan.to_csv(index=False).encode("utf-8"),
                       file_name="trade_plan_put_spreads.csv", mime="text/csv")

# Per‑ticker detail
if per_ticker:
    st.markdown("---")
    st.subheader("Per‑Ticker Candidates")
    for sym, det in per_ticker.items():
        with st.expander(f"{sym} — Top candidates & context"):
            ctx = det.get("context", {})
            st.write({k: (round(v,3) if isinstance(v,(int,float)) else v) for k,v in ctx.items()})
            best = pd.DataFrame(det.get("best", []))
            if not best.empty:
                show = best.copy()
                for c in ["Credit","MaxLoss"]:
                    show[c] = show[c].map(lambda x: f"${x:,.2f}")
                for c in ["ROI","POP","IV_used","IV_proxy","IVP_proxy"]:
                    if c in show.columns:
                        show[c] = show[c].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "")
                st.dataframe(show[["Ticker","DTE","Expiry","ShortK","LongK","Width","Credit","ROI","POP","IV_proxy","IVP_proxy","Avoid","SuggestOpen","SuggestClose"]], use_container_width=True)

st.markdown("---")
st.subheader("Notes & Methodology")
st.markdown("""
**POP** uses a lognormal model vs. **breakeven** (short strike − credit) and the short‑leg IV. **Delta band** selects OTM puts with a probability tilt (more negative delta ≈ more premium, less probability).  
**IV Proxy** blends near‑ATM option IVs; **IV Percentile (proxy)** compares that to the past ~60 trading sessions' 20‑day HV as a rough gauge (not true IVR).  
**Timing**: Favor entries when **IV_proxy > RV20** and **RSI ~40–65** in an **uptrend**; plan exits **~3 DTE** or at **50% profit** to curb gamma risk.  
**Liquidity filters** attempt to reduce slippage/assignment headaches by enforcing **OI/Volume** minimums and a **max bid‑ask %** on both legs.  
Data are best‑effort & for education—always verify in your broker.
""")
'''

path = "/mnt/data/put_credit_spreads_app_v2.py"
with open(path, "w") as f:
    f.write(improved_code)

path
