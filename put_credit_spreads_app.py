# Write a ready-to-run Streamlit app for screening weekly/biweekly put credit spreads
from datetime import datetime, timedelta, timezone
import textwrap, json, os, math

app_code = r'''
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta, timezone
import math

st.set_page_config(page_title="Put Credit Spread Finder", layout="wide")

st.title("📉 Put Credit Spread Finder (Weekly/Biweekly)")
st.caption("Targets ≥ 65% POP, avoids earnings & ex-div dates when possible. Uses free data sources (Yahoo via yfinance; optional Alpha Vantage / FMP / Finnhub keys).")

with st.sidebar:
    st.header("Settings")
    tickers = st.text_area("Symbols (comma/space/newline separated)", "SPY, AAPL, MSFT").upper()
    pop_target = st.slider("Minimum POP", 0.5, 0.9, 0.65, 0.01)
    max_width = st.number_input("Max spread width ($)", 1.0, 50.0, 5.0, 0.5)
    min_credit = st.number_input("Minimum net credit ($)", 0.01, 10.0, 0.25, 0.05)
    dte_choices = st.multiselect("Target DTE windows", ["7-10", "12-16"], default=["7-10","12-16"])
    avoid_earnings = st.checkbox("Avoid earnings during holding window", True)
    avoid_exdiv = st.checkbox("Avoid ex-dividend during holding window", True)
    use_mid_prices = st.checkbox("Use mid prices (bid/ask midpoint)", True)
    risk_free = st.number_input("Risk-free rate (annualized)", 0.0, 0.10, 0.03, 0.005, format="%.3f")
    api_keys = st.expander("Optional API Keys (for better calendars/news)")
    with api_keys:
        alpha_key = st.text_input("Alpha Vantage API Key (for news sentiment)", type="password")
        fmp_key = st.text_input("Financial Modeling Prep API Key (dividends/earnings calendars)", type="password")
        finnhub_key = st.text_input("Finnhub API Key (earnings calendar backup)", type="password")

def parse_tickers(raw):
    parts = [p.strip().upper() for p in raw.replace("\n",",").replace(" ",",").split(",")]
    return sorted(list({p for p in parts if p}))

def norm_cdf(x):
    # Standard normal CDF using math.erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1*delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100/(1+rs))

def pop_above_breakeven(S0, B, iv, T, r=0.0):
    # Geometric Brownian Motion under risk-neutral measure
    if S0 <= 0 or B <= 0 or iv <= 0 or T <= 0:
        return np.nan
    mu = r - 0.5*iv*iv
    z = (math.log(S0/B) + mu*T) / (iv*math.sqrt(T))
    return 1.0 - norm_cdf(-z)  # P(S_T > B)

def bs_put_delta(S, K, T, r, sigma):
    if S<=0 or K<=0 or T<=0 or sigma<=0:
        return np.nan
    d1 = (math.log(S/K)+(r+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    # put delta (non-discounted) ~ N(d1)-1
    return norm_cdf(d1) - 1.0

@st.cache_data(show_spinner=False)
def get_hist(ticker, period="1y"):
    return yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)

def upcoming_events_yf(ticker):
    # yfinance sometimes gives earnings calendar (may be empty)
    tk = yf.Ticker(ticker)
    cal = {}
    try:
        c = tk.calendar
        if isinstance(c, pd.DataFrame) and not c.empty:
            for k,v in c.to_dict()[ticker].items():
                cal[str(k)] = str(v)
    except Exception:
        pass
    return cal

def fmp_dividends_exdate(ticker, key):
    if not key:
        return None
    try:
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{ticker}?apikey={key}"
        data = requests.get(url, timeout=10).json()
        # FMP returns history; try to fetch next announced exDate from 'dividends-calendar' endpoint
        cal_url = f"https://financialmodelingprep.com/stable/dividends-calendar?symbol={ticker}&apikey={key}"
        cal = requests.get(cal_url, timeout=10).json()
        if isinstance(cal, list) and len(cal):
            # choose nearest future exDate
            today = datetime.now(timezone.utc).date()
            fut = [pd.to_datetime(x.get("exDividendDate","")).date() for x in cal if x.get("exDividendDate")]
            fut = [d for d in fut if d>=today]
            if fut:
                return min(fut)
    except Exception:
        return None
    return None

def fmp_earnings_date(ticker, key):
    if not key:
        return None
    try:
        url = f"https://financialmodelingprep.com/api/v3/earnings-calendar?symbol={ticker}&limit=10&apikey={key}"
        data = requests.get(url, timeout=10).json()
        if isinstance(data, list) and len(data):
            # choose nearest future date
            today = datetime.now(timezone.utc).date()
            fut = [pd.to_datetime(x.get("date","")).date() for x in data if x.get("date")]
            fut = [d for d in fut if d>=today]
            if fut:
                return min(fut)
    except Exception:
        return None
    return None

def finnhub_earnings_date(ticker, key):
    if not key:
        return None
    try:
        url = f"https://finnhub.io/api/v1/calendar/earnings?symbol={ticker}&from={datetime.utcnow().date()}&to={datetime.utcnow().date()+timedelta(days=120)}&token={key}"
        data = requests.get(url, timeout=10).json()
        items = data.get("earningsCalendar", [])
        if items:
            fut = [pd.to_datetime(x.get("date","")).date() for x in items if x.get("date")]
            fut = [d for d in fut if d>=datetime.utcnow().date()]
            if fut:
                return min(fut)
    except Exception:
        return None
    return None

def expected_move_from_chain(chain_calls, chain_puts, S0):
    # approximate using ATM straddle * 0.85 (common rough heuristic)
    # find nearest strike to S0
    if chain_calls.empty or chain_puts.empty:
        return np.nan
    calls = chain_calls.copy()
    puts = chain_puts.copy()
    calls["dist"] = (calls["strike"]-S0).abs()
    puts["dist"]  = (puts["strike"]-S0).abs()
    try:
        c = calls.sort_values("dist").iloc[0]
        p = puts.sort_values("dist").iloc[0]
        # choose price: mid or last or bid/ask
        cpx = (c.get("bid",np.nan)+c.get("ask",np.nan))/2 if not np.isnan(c.get("bid",np.nan)) and not np.isnan(c.get("ask",np.nan)) else c.get("lastPrice",np.nan)
        ppx = (p.get("bid",np.nan)+p.get("ask",np.nan))/2 if not np.isnan(p.get("bid",np.nan)) and not np.isnan(p.get("ask",np.nan)) else p.get("lastPrice",np.nan)
        if np.isnan(cpx) or np.isnan(ppx):
            return np.nan
        straddle = cpx + ppx
        return 0.85 * straddle
    except Exception:
        return np.nan

def find_spreads_for_ticker(ticker, pop_target, dte_windows, max_width, min_credit, avoid_earnings, avoid_exdiv, use_mid, r):
    tk = yf.Ticker(ticker)
    spot = tk.history(period="1d")["Close"][-1]
    hist = get_hist(ticker, "1y")
    rv20 = hist["Close"].pct_change().rolling(20).std().iloc[-1] * (252**0.5)
    rsi14 = rsi(hist["Close"]).iloc[-1]
    ma50 = hist["Close"].rolling(50).mean().iloc[-1]
    trend = "uptrend" if hist["Close"].iloc[-1] > ma50 else "down/sideways"
    today = datetime.utcnow().date()

    # Calendar filters
    next_earn = fmp_earnings_date(ticker, st.session_state.get("fmp_key","")) or finnhub_earnings_date(ticker, st.session_state.get("finnhub_key",""))
    next_exdiv = fmp_dividends_exdate(ticker, st.session_state.get("fmp_key",""))

    results = []
    expirations = tk.options
    if not expirations:
        return results

    # Map dte_windows to day ranges
    ranges = []
    if "7-10" in dte_windows:
        ranges.append((7,10))
    if "12-16" in dte_windows:
        ranges.append((12,16))

    for exp in expirations:
        edate = pd.to_datetime(exp).date()
        dte = (edate - today).days
        # check if in any window
        if not any(lo <= dte <= hi for lo,hi in ranges):
            continue
        # load chain
        opt = tk.option_chain(exp)
        calls, puts = opt.calls.copy(), opt.puts.copy()
        exp_move = expected_move_from_chain(calls, puts, spot)
        # choose candidate short strikes: OTM puts with delta ~ -0.15 to -0.35
        # We'll compute delta ourselves using each row's IV
        candidates = []
        for idx, row in puts.iterrows():
            K = float(row["strike"])
            if K >= spot:  # need OTM put
                continue
            iv = float(row.get("impliedVolatility", np.nan))
            if not np.isfinite(iv) or iv <= 0:
                continue
            T = max(dte/365.0, 1e-6)
            delta = bs_put_delta(spot, K, T, r, iv)
            if np.isnan(delta):
                continue
            if delta < -0.35 or delta > -0.10:
                continue
            # pick a long strike below, within max_width
            below = puts[(puts["strike"] < K) & (puts["strike"] >= K - max_width)]
            if below.empty:
                continue
            # choose long strike as nearest below K
            L = float(below["strike"].max())
            short_bid, short_ask = float(row.get("bid", np.nan)), float(row.get("ask", np.nan))
            long_bid  = float(below[below["strike"]==L]["bid"].values[0])
            long_ask  = float(below[below["strike"]==L]["ask"].values[0])
            if any(map(lambda x: np.isnan(x), [short_bid, short_ask, long_bid, long_ask])):
                continue
            credit = (short_bid + short_ask)/2 - (long_bid + long_ask)/2 if use_mid else short_bid - long_ask
            if credit < min_credit:
                continue
            width = K - L
            max_loss = width - credit
            if max_loss <= 0:
                continue
            roi = credit / max_loss
            breakeven = K - credit
            T = max(dte/365.0, 1e-6)
            pop = pop_above_breakeven(spot, breakeven, iv, T, r=r)
            if not np.isfinite(pop):
                continue

            avoid_note = []
            if avoid_earnings and next_earn and (today <= next_earn <= edate):
                avoid_note.append("earnings")
            if avoid_exdiv and next_exdiv and (today <= next_exdiv <= edate):
                avoid_note.append("ex-div")
            results.append({
                "Ticker": ticker,
                "DTE": dte,
                "Expiry": edate,
                "Spot": round(spot,2),
                "RV20": round(rv20,3) if pd.notna(rv20) else np.nan,
                "RSI14": round(rsi14,1) if pd.notna(rsi14) else np.nan,
                "Trend": trend,
                "ExpectedMove~": round(exp_move,2) if pd.notna(exp_move) else np.nan,
                "ShortK": K,
                "LongK": L,
                "Width": width,
                "Credit": round(credit,2),
                "MaxLoss": round(max_loss,2),
                "ROI": round(roi,3),
                "POP": round(pop,3),
                "IV_used": round(iv,3),
                "Avoid": ", ".join(avoid_note) if avoid_note else "",
                "SuggestOpen": "Today" if (rv20 and iv and iv>rv20 and rsi14<65) else "Wait for higher IV / pullback",
                "SuggestClose": (edate - timedelta(days=3)).isoformat() + " (≈ 3 DTE) or 50% profit"
            })

    # Filter by POP
    results = [r for r in results if r["POP"] >= pop_target]
    # sort by ROI then POP
    results = sorted(results, key=lambda x: (x["Avoid"]!="", -x["ROI"], -x["POP"]))
    return results

# stash keys into session for helper funcs
st.session_state["fmp_key"] = fmp_key if 'fmp_key' not in st.session_state or st.session_state['fmp_key'] != fmp_key else st.session_state['fmp_key']
st.session_state["finnhub_key"] = finnhub_key if 'finnhub_key' not in st.session_state or st.session_state['finnhub_key'] != finnhub_key else st.session_state['finnhub_key']

symbols = parse_tickers(tickers)

rows = []
progress = st.progress(0.0, text="Scanning chains...")
for i, sym in enumerate(symbols, start=1):
    try:
        rows.extend(find_spreads_for_ticker(sym, pop_target, dte_choices, max_width, min_credit, avoid_earnings, avoid_exdiv, use_mid_prices, risk_free))
    except Exception as e:
        st.warning(f"{sym}: {e}")
    progress.progress(i/len(symbols), text=f"Processed {i}/{len(symbols)}")

df = pd.DataFrame(rows)
if df.empty:
    st.info("No candidates met your filters. Try widening strikes, lowering POP threshold, or disabling some avoid rules.")
else:
    # nice formatting
    fmt = df.copy()
    fmt["Credit"] = fmt["Credit"].map(lambda x: f"${x:,.2f}")
    fmt["MaxLoss"] = fmt["MaxLoss"].map(lambda x: f"${x:,.2f}")
    fmt["ROI"] = fmt["ROI"].map(lambda x: f"{x*100:.1f}%")
    fmt["POP"] = fmt["POP"].map(lambda x: f"{x*100:.1f}%")
    fmt["IV_used"] = fmt["IV_used"].map(lambda x: f"{x*100:.1f}%")
    cols = ["Ticker","DTE","Expiry","Spot","ExpectedMove~","ShortK","LongK","Width","Credit","MaxLoss","ROI","POP","IV_used","RV20","RSI14","Trend","Avoid","SuggestOpen","SuggestClose"]
    st.dataframe(fmt[cols], use_container_width=True, height=500)

st.markdown("---")
st.subheader("🔎 Notes & Methodology")
st.markdown("""
- **POP**: Probability underlying finishes **above** breakeven at expiration under a lognormal model using the short leg's implied volatility. This is a model-based estimate, not a guarantee.
- **Credit/Max Loss**: Uses mid-prices when enabled, otherwise conservative bid/ask. ROI = Credit / (Width - Credit).
- **Expected Move~**: Approximate 1-week expected move = ATM straddle * 0.85 (rule of thumb) from current chain.
- **Filters**: Short strike delta (via Black–Scholes) between ~-0.10 and -0.35. Avoid flags are applied if an earnings or ex-dividend date falls inside the holding window.
- **Timing**: Suggest opening when **IV > RV20** and price is above 50DMA but after a minor dip (RSI < 65). Suggested close is **~3 DTE or 50% profit**, to reduce gamma risk into expiry.
- **APIs**: Price/chain via `yfinance` (Yahoo Finance). Optional news via Alpha Vantage; calendars via FMP and/or Finnhub if keys are supplied.
- **Disclaimers**: Data may be delayed/limited. Always double-check chains in your broker. Nothing here is financial advice.
""")
'''

path = "/mnt/data/put_credit_spreads_app.py"
with open(path, "w") as f:
    f.write(app_code)

path
