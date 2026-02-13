
# app.py (mobile-first Streamlit)
# Run: streamlit run app.py
import os
import json
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import argrelextrema

import warnings
warnings.filterwarnings("ignore")

import FinanceDataReader as fdr
import yfinance as yf

import plotly.graph_objects as go

# -----------------------------
# Indicators & pattern logic
# -----------------------------
def calculate_rsi(df, period=14):
    delta = df["Close"].diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["Close"].ewm(span=fast).mean()
    ema_slow = df["Close"].ewm(span=slow).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
    return df

def calculate_stochastic(df, period=14, d_period=3):
    low_min = df["Low"].rolling(window=period).min()
    high_max = df["High"].rolling(window=period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    df["K%"] = 100 * ((df["Close"] - low_min) / denom)
    df["D%"] = df["K%"].rolling(window=d_period).mean()
    return df

def calculate_adx(df, period=14):
    high_diff = df["High"].diff()
    low_diff = -df["Low"].diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)

    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            np.abs(df["High"] - df["Close"].shift()),
            np.abs(df["Low"] - df["Close"].shift()),
        ),
    )
    atr = pd.Series(tr, index=df.index).rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    di_diff = (plus_di - minus_di).abs()
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = 100 * (di_diff / di_sum).rolling(window=period).mean()
    return df

def find_pivots(df, order=5):
    high_idx = argrelextrema(df["High"].values, np.greater, order=order)[0]
    low_idx = argrelextrema(df["Low"].values, np.less, order=order)[0]
    pivots = []
    for idx in high_idx:
        pivots.append((int(idx), float(df["High"].iloc[idx]), "High"))
    for idx in low_idx:
        pivots.append((int(idx), float(df["Low"].iloc[idx]), "Low"))
    pivots.sort(key=lambda x: x[0])
    return pivots

def detect_harmonic_patterns(df):
    pivots = find_pivots(df, order=5)
    if len(pivots) < 4:
        return None
    patterns = []
    start_idx = max(0, len(pivots) - 5)
    for i in range(start_idx, len(pivots) - 3):
        p1, p2, p3, p4 = pivots[i : i + 4]
        _, y1, _ = p1
        _, y2, _ = p2
        _, y3, _ = p3
        _, y4, _ = p4
        xa = abs(y2 - y1)
        ab = abs(y3 - y2)
        bc = abs(y4 - y3)
        if xa == 0 or ab == 0 or bc == 0:
            continue
        ab_ratio = ab / xa
        bc_ratio = bc / ab
        if 0.45 < ab_ratio < 0.75 and 0.25 < bc_ratio < 0.55:
            patterns.append({"type":"Gartley","points":[p1,p2,p3,p4]})
        elif 0.65 < ab_ratio < 0.95 and 1.3 < bc_ratio < 2.0:
            patterns.append({"type":"Butterfly","points":[p1,p2,p3,p4]})
    return patterns if patterns else None

def detect_wolfe_wave_pattern(df):
    df = df.copy()
    df["RSI"] = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_stochastic(df)
    df = calculate_adx(df)

    rsi = float(df["RSI"].iloc[-1])
    macd = float(df["MACD"].iloc[-1])
    hist = float(df["MACD_Histogram"].iloc[-1])
    k = float(df["K%"].iloc[-1]) if not np.isnan(df["K%"].iloc[-1]) else 50.0
    adx = float(df["ADX"].iloc[-1]) if not np.isnan(df["ADX"].iloc[-1]) else 0.0

    pivots = find_pivots(df, order=5)
    if len(pivots) < 5:
        return None

    last_5 = pivots[-5:]
    types = [p[2] for p in last_5]
    bull = types == ["Low","High","Low","High","Low"]
    bear = types == ["High","Low","High","Low","High"]
    if not (bull or bear):
        return None

    p1, p2, p3, p4, p5 = last_5
    x1, y1 = p1[0], p1[1]
    x4, y4 = p4[0], p4[1]
    if (x4 - x1) == 0:
        return None

    cur_idx = len(df) - 1
    cur_price = float(df["Close"].iloc[-1])
    m = (y4 - y1) / (x4 - x1)
    c = y1 - (m * x1)
    epa = m * cur_idx + c
    disparity = ((cur_price - epa) / epa) * 100 if epa != 0 else 0.0

    status = "매수" if bull else "매도"
    strength = 1.0

    if bull and cur_price >= epa:
        return {"pattern":"목표 도달","points":last_5,"disparity":round(disparity,2),"epa_price":epa,"rsi":round(rsi,2),"strength":0.0,"macd":round(macd,4),"stoch_k":round(k,1),"adx":round(adx,1)}
    if bear and cur_price <= epa:
        return {"pattern":"목표 도달","points":last_5,"disparity":round(disparity,2),"epa_price":epa,"rsi":round(rsi,2),"strength":0.0,"macd":round(macd,4),"stoch_k":round(k,1),"adx":round(adx,1)}

    # Strength (keep similar flavor)
    if bull:
        if rsi <= 40: strength += 1
        if rsi <= 30: strength += 1
        if macd > 0 and hist > 0: strength += 1
        if k < 20: strength += 1
        if adx > 25: strength += 0.5
    else:
        if rsi >= 60: strength += 1
        if rsi >= 70: strength += 1
        if macd < 0 and hist < 0: strength += 1
        if k > 80: strength += 1
        if adx > 25: strength += 0.5

    if detect_harmonic_patterns(df):
        strength += 1

    strength = float(min(5.0, strength))
    if strength >= 4:
        status = "매우 적극" + status
    elif strength >= 2.5:
        status = "적극" + status

    return {
        "pattern": status,
        "points": last_5,
        "disparity": round(float(disparity), 2),
        "epa_price": float(epa),
        "rsi": round(rsi, 2),
        "strength": strength,
        "macd": round(macd, 4),
        "stoch_k": round(k, 1),
        "adx": round(adx, 1),
    }

# -----------------------------
# Data fetch (FDR -> yfinance)
# -----------------------------
def get_stock_data(code: str, start_year: int):
    start_date = datetime.datetime(start_year, 1, 1)
    try:
        df = fdr.DataReader(code, start_date)
        if df is not None and not df.empty and len(df) > 80:
            return df
    except Exception:
        pass

    try:
        for suffix in [".KS", ".KQ"]:
            ticker = f"{code}{suffix}"
            df = yf.Ticker(ticker).history(start=start_date, auto_adjust=False, actions=False)
            if df is not None and not df.empty and len(df) > 80:
                return df
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=60*60*6, show_spinner=False)
def get_universe(limit_kospi=200):
    stocks = fdr.StockListing("KOSPI").head(int(limit_kospi))[["Code", "Name"]]
    etf_codes = [
        ("069500", "KODEX 200"), ("102110", "TIGER 200"), ("252710", "KODEX KOSPI100"),
        ("122630", "KODEX 레버리지"), ("123310", "KODEX 인버스"), ("130680", "KODEX 코스닥150"),
        ("152100", "KODEX 200레버리지"), ("305540", "KODEX 나스닥100"), ("273130", "TIGER 미국나스닥100")
    ]
    etf_df = pd.DataFrame(etf_codes, columns=["Code","Name"])
    combined = pd.concat([stocks, etf_df], ignore_index=True).drop_duplicates(subset=["Code"], keep="first")
    combined["Code"] = combined["Code"].astype(str).str.zfill(6)
    return combined.reset_index(drop=True)

# -----------------------------
# Signal history
# -----------------------------
def history_path():
    os.makedirs("data", exist_ok=True)
    return os.path.join("data", "signal_history.json")

def load_hist():
    p = history_path()
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_hist(h):
    p = history_path()
    with open(p, "w", encoding="utf-8") as f:
        json.dump(h, f, ensure_ascii=False, indent=2)

def update_hist(h, code, signal, date):
    h.setdefault(code, [])
    h[code] = h[code][-9:] + [{"date":date, "signal":signal}]

def check_consec(h, code, signal, required_days=2):
    if code not in h:
        return (False, 1)
    s = h[code]
    if len(s) < required_days - 1:
        return (False, len(s) + 1)
    recent = s[-(required_days-1):]
    if any(e["signal"] != signal for e in recent):
        return (False, 1)
    cnt = 1
    for e in reversed(s):
        if e["signal"] == signal: cnt += 1
        else: break
    return (True, cnt)

# -----------------------------
# Plotly chart (mobile friendly: pinch-zoom)
# -----------------------------
def make_chart(df, points, name, pattern):
    df = df.copy()
    df = df.reset_index()
    if "Date" not in df.columns:
        df.rename(columns={"index":"Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])

    # focus around pivots
    p_idx = [p[0] for p in points]
    start = max(0, min(p_idx) - 20)
    end = min(len(df), max(p_idx) + 140)
    view = df.iloc[start:end].copy()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=view["Date"],
        open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
        name="OHLC"
    ))
    # pivot markers
    px = [df["Date"].iloc[p[0]] for p in points]
    py = [p[1] for p in points]
    fig.add_trace(go.Scatter(x=px, y=py, mode="markers+text", text=[str(i) for i in range(1,6)],
                             textposition="top center", name="Pivots"))
    fig.update_layout(
        title=f"{name} · {pattern}",
        height=520,
        margin=dict(l=10,r=10,t=50,b=10),
        xaxis_rangeslider_visible=False,
        showlegend=False
    )
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    return fig

# -----------------------------
# Mobile-first UI
# -----------------------------
st.set_page_config(page_title="KOSPI Wolfe (Mobile)", layout="centered")

# small CSS to improve mobile spacing
st.markdown("""
<style>
section.main > div { padding-top: 0.8rem; }
div[data-testid="stSidebar"] { padding-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.title("KOSPI Wolfe 스캐너 (모바일용)")

with st.expander("설정", expanded=True):
    limit_kospi = st.slider("KOSPI 상위 N개", 50, 300, 200, 10)
    start_year = st.selectbox("데이터 시작 연도", [datetime.datetime.now().year - i for i in range(1, 6)], index=1)
    required_days = st.selectbox("연속 신호 기준(거래일)", [2, 3], index=0)
    max_workers = st.slider("병렬 스캔 작업수", 2, 20, 10, 1)
    name_filter = st.text_input("종목명 검색(선택)", "")
    run = st.button("스캔 실행", type="primary", use_container_width=True)

universe = get_universe(limit_kospi=limit_kospi)
if name_filter.strip():
    universe = universe[universe["Name"].str.contains(name_filter.strip(), case=False, na=False)].reset_index(drop=True)

st.caption(f"대상 종목 수: {len(universe)}")

def scan(universe, start_year, max_workers, required_days):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    hist = load_hist()
    results, pending = [], []
    dfs_cache = {}

    prog = st.progress(0)
    msg = st.empty()
    total = len(universe)

    def process(row):
        code = str(row["Code"]).zfill(6)
        name = str(row["Name"])
        df = get_stock_data(code, start_year)
        if df is None or df.empty or len(df) < 80:
            return None
        ans = detect_wolfe_wave_pattern(df)
        if not ans:
            return None

        pat = ans["pattern"]
        sig = "BUY" if "매수" in pat else ("SELL" if "매도" in pat else "NEUTRAL")
        update_hist(hist, code, sig, today)
        is_con, days = check_consec(hist, code, sig, required_days=required_days)

        item = {
            "Code": code, "종목명": name, "현재가": int(float(df["Close"].iloc[-1])),
            "패턴유형": pat, "연속일": days,
            "강도": float(ans.get("strength", 0.0)),
            "RSI": float(ans.get("rsi", 0.0)),
            "괴리율(%)": float(ans.get("disparity", 0.0)),
            "points": ans["points"],
        }
        return (is_con, item, df)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process, row): row for _, row in universe.iterrows()}
        done = 0
        for f in as_completed(futs):
            done += 1
            row = futs[f]
            prog.progress(done / total)
            msg.write(f"스캔 중… {done}/{total} · {row['Name']}")
            try:
                out = f.result()
                if out is None:
                    continue
                is_con, item, df = out
                dfs_cache[item["Code"]] = df
                (results if is_con else pending).append(item)
            except Exception:
                continue

    save_hist(hist)
    msg.write("완료")
    prog.empty()

    res_df = pd.DataFrame(results).sort_values(["강도","괴리율(%)"], ascending=[False, True]) if results else pd.DataFrame()
    pend_df = pd.DataFrame(pending).sort_values(["강도","괴리율(%)"], ascending=[False, True]) if pending else pd.DataFrame()
    return res_df, pend_df, dfs_cache

if run:
    res_df, pend_df, dfs_cache = scan(universe, start_year, max_workers, required_days)
    st.session_state["res_df"] = res_df
    st.session_state["pend_df"] = pend_df
    st.session_state["dfs_cache"] = dfs_cache

res_df = st.session_state.get("res_df", pd.DataFrame())
pend_df = st.session_state.get("pend_df", pd.DataFrame())
dfs_cache = st.session_state.get("dfs_cache", {})

tab1, tab2, tab3 = st.tabs([f"연속({required_days}일)", "대기(1일차)", "차트"])

with tab1:
    if res_df.empty:
        st.info("연속 신호 결과가 없습니다.")
    else:
        st.dataframe(
            res_df[["Code","종목명","현재가","패턴유형","연속일","강도","RSI","괴리율(%)"]],
            use_container_width=True,
            hide_index=True,
            height=420
        )

with tab2:
    if pend_df.empty:
        st.write("없음")
    else:
        st.dataframe(
            pend_df[["Code","종목명","현재가","패턴유형","연속일","강도","RSI","괴리율(%)"]],
            use_container_width=True,
            hide_index=True,
            height=420
        )

with tab3:
    codes = []
    if not res_df.empty: codes += list(res_df["Code"].astype(str))
    if not pend_df.empty: codes += list(pend_df["Code"].astype(str))
    codes = list(dict.fromkeys(codes))

    if not codes:
        st.caption("스캔 실행 후 종목을 선택할 수 있습니다.")
    else:
        code = st.selectbox("종목 선택", codes)
        row = None
        if not res_df.empty and code in set(res_df["Code"]):
            row = res_df[res_df["Code"] == code].iloc[0].to_dict()
        elif not pend_df.empty and code in set(pend_df["Code"]):
            row = pend_df[pend_df["Code"] == code].iloc[0].to_dict()

        if row:
            st.markdown(f"**{row['종목명']} ({row['Code']})**  ·  {row['패턴유형']}")
            st.caption(f"현재가 {row['현재가']:,}  |  강도 {row['강도']:.1f}  |  RSI {row['RSI']:.1f}  |  괴리율 {row['괴리율(%)']:.2f}%")
            st.markdown(f"[네이버증권 열기](https://finance.naver.com/item/main.naver?code={row['Code']})")

            df = dfs_cache.get(code) or get_stock_data(code, start_year)
            fig = make_chart(df, row["points"], row["종목명"], row["패턴유형"])
            st.plotly_chart(fig, use_container_width=True)
