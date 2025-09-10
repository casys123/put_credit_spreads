# pcs-final-columns.py — Put Credit Spreads with requested columns
# Adds table columns:
# Symbol | Price~ | Exp Date | Short | Bid1 | Long | Ask2 | BE | BE% | Max Profit | Max Loss | Max Profit% | Risk/Reward | IV Rank | Loss Prob | Links

import io
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Put Credit Spreads — Requested Columns", layout="wide")
st.title("📉 Put Credit Spreads — Custom Columns View")

# ------------------------------
# Helpers
# ------------------------------

def parse_tickers_text(raw: str):
    if not raw:
        return []
    parts = [p.strip().upper() for p in raw.replace("\n", ",").replace(" ", ",").split(",")]
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

def iv_percentile_proxy(series: pd.Series, value: float) -> float:
    try:
        arr = np.array(series.dropna()[-60:])
        if len(arr) < 10 or not np.isfinite(value):
            return float("nan")
        return float((arr <= value).mean())
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
            bid = row.get("bid", float("nan"))
            ask = row.get("ask", float("nan"))
            last = row.get("lastPrice", float("nan"))
            if pd.notna(bid) and pd.notna(ask) and ask >= bid and bid > 0:
                return (bid + ask) / 2
            if pd.notna(last): return last
            if pd.notna(bid): return bid
            if pd.notna(ask): return ask
            return float("nan")
        cpx = mid_or_last(c); ppx = mid_or_last(p)
        if pd.isna(cpx) or pd.isna(ppx): return float("nan")
        return 0.85 * (cpx + ppx)
    except Exception:
        return float("nan")

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("Inputs")
    tickers_text = st.text_area("Symbols (paste)", "AAPL, MSFT, SPY").upper()
    pop_target = st.slider("Minimum POP", 0.50, 0.99, 0.65, 0.01)
    delta_min, delta_max = st.slider("Short Put Delta band (negative)", -0.40, -0.05, (-0.35, -0.10), 0.01)
    max_width = st.number_input("Max spread width ($)", 1.0, 50.0, 5.0, 0.5)
    min_credit = st.number_input("Minimum net credit ($)", 0.01, 10.0, 0.25, 0.05)
    dte_choices = st.multiselect("Target DTE windows", ["7-10", "12-16"], default=["7-10","12-16"])
    risk_free = st.number_input("Risk-free rate (annualized)", 0.0, 0.10, 0.03, 0.005, format="%.3f")

symbols = [s for s in [x.strip() for x in tickers_text.replace("\n",",").split(",")] if s]

# ------------------------------
# Scanner core
# ------------------------------
def scan_one(ticker: str):
    out = []
    try:
        hist = get_hist(ticker, "1y")
        if hist is None or hist.empty:
            return out
        spot = float(hist["Close"].iloc[-1])
        hv_series = hist["Close"].pct_change().rolling(20).std()*(252**0.5)

        tk = yf.Ticker(ticker)
        try:
            expirations = tk.options
        except Exception:
            expirations = []

        ranges = []
        if "7-10" in dte_choices: ranges.append((7,10))
        if "12-16" in dte_choices: ranges.append((12,16))

        today = datetime.utcnow().date()

        for exp in expirations or []:
            try:
                edate = pd.to_datetime(exp).date()
            except Exception:
                continue
            dte = (edate - today).days
            if not any(lo <= dte <= hi for lo,hi in ranges):
                continue

            try:
                chain = tk.option_chain(exp)
                calls, puts = chain.calls.copy(), chain.puts.copy()
            except Exception:
                continue

            iv_now = iv_proxy_from_chain(calls, puts, spot)
            ivp = iv_percentile_proxy(hv_series, iv_now)

            for _, row in puts.iterrows():
                try:
                    K = float(row["strike"])
                    if K >= spot: continue
                    iv = float(row.get("impliedVolatility", float("nan")))
                    if not np.isfinite(iv) or iv <= 0: continue
                    T = max(dte/365.0, 1e-6)
                    delta = bs_put_delta(spot, K, T, risk_free, iv)
                    if not np.isfinite(delta) or delta < delta_min or delta > delta_max:
                        continue

                    # choose long strike below K within width
                    below = puts[(puts["strike"] < K) & (puts["strike"] >= K - max_width)]
                    if below.empty: continue
                    L = float(below["strike"].max())
                    lrow = below[below["strike"]==L].iloc[0]

                    # prices for requested columns
                    sbid = float(row.get("bid", np.nan))
                    sask = float(row.get("ask", np.nan))
                    lbid = float(lrow.get("bid", np.nan))
                    lask = float(lrow.get("ask", np.nan))
                    if any(pd.isna(x) for x in [sbid, sask, lbid, lask]): 
                        continue

                    credit_mid = ( (sbid + sask)/2 - (lbid + lask)/2 )
                    credit_conservative = sbid - lask
                    credit = credit_mid if credit_mid == credit_mid else credit_conservative  # prefer mid

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

                    # Requested fields
                    price_tilde = round(spot, 2)
                    exp_label = f"{edate.isoformat()} ({dte})"
                    short_lbl = f"{K:.2f}P"
                    long_lbl  = f"{L:.2f}P"
                    bid1 = round(sbid, 2)
                    ask2 = round(lask, 2)  # long ask
                    be  = round(breakeven, 2)
                    be_pct = (breakeven/spot - 1.0) * 100.0
                    max_profit = round(credit, 2)
                    max_loss_disp = round(max_loss, 2)
                    max_profit_pct = roi * 100.0
                    rr = (max_loss / credit) if credit > 0 else float("inf")
                    loss_prob = (1.0 - pop) * 100.0
                    iv_rank_disp = (ivp * 100.0) if np.isfinite(ivp) else None

                    # links (Yahoo + Barchart)
                    # Convert expiration date to Unix timestamp for Yahoo URL
                    exp_timestamp = int(datetime.combine(edate, datetime.min.time()).timestamp())
                    yurl = f"https://finance.yahoo.com/quote/{ticker}/options?date={exp_timestamp}"
                    burl = f"https://www.barchart.com/stocks/quotes/{ticker}/options"
                    links = f"[Yahoo]({yurl}) | [Barchart]({burl})"

                    out.append({
                        "Symbol": ticker,
                        "Price~": price_tilde,
                        "Exp Date": exp_label,
                        "Short": short_lbl,
                        "Bid1": bid1,
                        "Long": long_lbl,
                        "Ask2": ask2,
                        "BE": be,
                        "BE%": be_pct,
                        "Max Profit": max_profit,
                        "Max Loss": max_loss_disp,
                        "Max Profit%": max_profit_pct,
                        "Risk/Reward": rr,
                        "IV Rank": iv_rank_disp,
                        "Loss Prob": loss_prob,
                        "Links": links,
                    })
                except Exception:
                    continue
        return out
    except Exception:
        return out

# Run
symbols = symbols or []
rows = []
if not symbols:
    st.info("Add symbols on the left to scan.")
else:
    prog = st.progress(0.0, text="Scanning…")
    for i, sym in enumerate(symbols, start=1):
        rows.extend(scan_one(sym))
        prog.progress(i/len(symbols), text=f"{i}/{len(symbols)} processed")

df = pd.DataFrame(rows)
if df.empty:
    st.warning("No rows matched the filters. Try relaxing POP, widening width, or adjusting delta band.")
else:
    fmt = df.copy()
    # format numeric columns
    fmt["BE%"] = fmt["BE%"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
    fmt["Max Profit"] = fmt["Max Profit"].map(lambda x: f"${x:,.2f}")
    fmt["Max Loss"] = fmt["Max Loss"].map(lambda x: f"${x:,.2f}")
    fmt["Max Profit%"] = fmt["Max Profit%"].map(lambda x: f"{x:.2f}%")
    fmt["Risk/Reward"] = fmt["Risk/Reward"].map(lambda x: f"{x:.2f} to 1" if pd.notna(x) and np.isfinite(x) else "")
    fmt["IV Rank"] = fmt["IV Rank"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")
    fmt["Loss Prob"] = fmt["Loss Prob"].map(lambda x: f"{x:.1f}%" if pd.notna(x) else "")

    order = ["Symbol","Price~","Exp Date","Short","Bid1","Long","Ask2","BE","BE%","Max Profit","Max Loss","Max Profit%","Risk/Reward","IV Rank","Loss Prob","Links"]
    st.dataframe(fmt[order], use_container_width=True, height=560)

    st.download_button(
        "⬇️ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="put_credit_spreads_custom_columns.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Notes: BE% = (Breakeven/Spot − 1). Max Profit = net credit (uses mid-price when available). IV Rank is a proxy from near‑ATM IV vs ~60d HV.")
