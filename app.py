
# app.py - Streamlit Cloud safe + KOSPI200 universe auto (via Naver Finance crawl)
# Why: KRX(data.krx.co.kr) can be "Access Denied" on Streamlit Cloud.
# Source idea: Naver Finance provides KOSPI200 constituent table pages (/sise/entryJongmok.nhn?page=)
import os, json, datetime, re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoup
import yfinance as yf

# Optional: KRX/Naver-backed data source (more stable for KR tickers on Cloud)
try:
    import FinanceDataReader as fdr
except Exception:
    fdr = None
import plotly.graph_objects as go

# -----------------------------
# Pivot detection (no scipy)
# -----------------------------
def find_pivots(df, order=5):
    highs = df["High"].values
    lows = df["Low"].values
    pivots = []
    n = len(df)
    for i in range(order, n - order):
        h = highs[i]
        l = lows[i]
        if h == np.max(highs[i-order:i+order+1]) and h > np.max(highs[i-order:i]) and h >= np.max(highs[i+1:i+order+1]):
            pivots.append((i, float(h), "High"))
        if l == np.min(lows[i-order:i+order+1]) and l < np.min(lows[i-order:i]) and l <= np.min(lows[i+1:i+order+1]):
            pivots.append((i, float(l), "Low"))
    pivots.sort(key=lambda x: x[0])
    dedup = {}
    for p in pivots:
        dedup[p[0]] = p
    return [dedup[k] for k in sorted(dedup.keys())]

# -----------------------------
# Indicators
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



# -----------------------------
# AI 예측(경량, 순수 NumPy)
# - 목표: "현재 기술지표 상태"로 N일 후 상승확률을 추정해 신호를 보정
# - 외부 ML 라이브러리 없이(Cloud/파이썬 버전 이슈 방지) 직접 로지스틱 회귀 학습
# -----------------------------
def _zscore(x: np.ndarray):
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (x - mu) / sd, mu, sd

def _sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))

def _train_logreg(X: np.ndarray, y: np.ndarray, lr=0.08, steps=600, l2=0.6):
    """
    아주 가벼운 배치 GD 로지스틱 회귀
    """
    n, d = X.shape
    w = np.zeros(d, dtype=float)
    b = 0.0
    for _ in range(steps):
        p = _sigmoid(X @ w + b)
        # gradients
        dw = (X.T @ (p - y)) / n + l2 * w
        db = float(np.mean(p - y))
        w -= lr * dw
        b -= lr * db
    return w, b

def build_ai_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    AI 학습용 feature table 생성 (지표 + 가격 포지션)
    """
    d = df.copy()
    if "RSI" not in d.columns:
        d["RSI"] = calculate_rsi(d)
    if "MACD" not in d.columns:
        d = calculate_macd(d)
    if "K%" not in d.columns:
        d = calculate_stochastic(d)
    if "ADX" not in d.columns:
        d = calculate_adx(d)

    # 추가 features (v3에 없던 것들을 간단히)
    d["ROC_10"] = d["Close"].pct_change(10) * 100.0
    ma20 = d["Close"].rolling(20).mean()
    sd20 = d["Close"].rolling(20).std()
    d["BB_pos"] = (d["Close"] - ma20) / (sd20.replace(0, np.nan))  # 0이면 NaN
    d["Vol_20"] = d["Close"].pct_change().rolling(20).std() * np.sqrt(252) * 100.0

    feats = d[["RSI", "MACD", "MACD_Histogram", "K%", "ADX", "ROC_10", "BB_pos", "Vol_20"]].copy()
    return feats

@st.cache_data(ttl=60*60, show_spinner=False)
def ai_probability_up(code: str, df: pd.DataFrame, horizon: int = 10):
    """
    종목별로 (feature -> horizon일 후 상승확률) 추정
    - 학습데이터: 최근 2년 내에서 가능한 구간
    - label: horizon일 후 수익률 > 0
    """
    if df is None or df.empty or len(df) < 150:
        return None

    feats = build_ai_features(df)
    # label
    future_ret = df["Close"].shift(-horizon) / df["Close"] - 1.0
    y = (future_ret > 0).astype(float)

    # drop NaN
    data = feats.copy()
    data["y"] = y
    data = data.dropna()
    if len(data) < 120:
        return None

    # time split (최근 20%는 검증, 나머지 학습)
    X = data[["RSI","MACD","MACD_Histogram","K%","ADX","ROC_10","BB_pos","Vol_20"]].to_numpy(dtype=float)
    yv = data["y"].to_numpy(dtype=float)
    split = int(len(data) * 0.8)
    Xtr, ytr = X[:split], yv[:split]
    Xte, yte = X[split:], yv[split:]

    Xtr_z, mu, sd = _zscore(Xtr)
    Xte_z = (Xte - mu) / sd

    w, b = _train_logreg(Xtr_z, ytr)

    # 검증 성능(간단)
    p_te = _sigmoid(Xte_z @ w + b)
    acc = float(np.mean((p_te >= 0.5) == (yte >= 0.5)))
    # 최신 확률
    last_x = X[-1:]
    last_z = (last_x - mu) / sd
    p_up = float(_sigmoid(last_z @ w + b)[0])

    return {"p_up": p_up, "acc": acc, "n": int(len(data)), "horizon": int(horizon)}

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
            patterns.append({"type":"Gartley"})
        elif 0.65 < ab_ratio < 0.95 and 1.3 < bc_ratio < 2.0:
            patterns.append({"type":"Butterfly"})
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

    if bull and cur_price >= epa:
        return {"pattern":"목표 도달","points":last_5,"disparity":round(disparity,2),"rsi":round(rsi,2),"strength":0.0}
    if bear and cur_price <= epa:
        return {"pattern":"목표 도달","points":last_5,"disparity":round(disparity,2),"rsi":round(rsi,2),"strength":0.0}

    status = "매수" if bull else "매도"
    strength = 1.0

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

    return {"pattern": status, "points": last_5, "disparity": round(float(disparity), 2), "rsi": round(rsi, 2), "strength": strength}


# -----------------------------
# Fallback signal (indicator-based)
# -----------------------------
def basic_signal(df: pd.DataFrame):
    """Return a relaxed BUY/SELL/NEUTRAL signal even when Wolfe/Harmonic patterns are not detected.
    This prevents scan results from becoming empty."""
    close = df["Close"].astype(float)

    rsi_series = calculate_rsi(close, period=14)
    rsi = float(rsi_series.iloc[-1]) if len(rsi_series) else 50.0

    macd, signal, hist = calculate_macd(close)
    macd_v = float(macd.iloc[-1]) if len(macd) else 0.0
    hist_v = float(hist.iloc[-1]) if len(hist) else 0.0

    k, d = calculate_stochastic(df, 14, 3)
    k_v = float(k.iloc[-1]) if len(k) else 50.0

    adx_series = calculate_adx(df, 14)
    adx = float(adx_series.iloc[-1]) if len(adx_series) else 0.0

    # Relaxed scoring (designed to produce results similar to pre-AI versions)
    buy_s = 0.0
    if rsi <= 45: buy_s += 1
    if rsi <= 35: buy_s += 1
    if macd_v > 0 and hist_v > 0: buy_s += 1
    if k_v < 30: buy_s += 1
    if adx >= 20: buy_s += 0.5

    sell_s = 0.0
    if rsi >= 55: sell_s += 1
    if rsi >= 65: sell_s += 1
    if macd_v < 0 and hist_v < 0: sell_s += 1
    if k_v > 70: sell_s += 1
    if adx >= 20: sell_s += 0.5

    if buy_s >= max(2.0, sell_s + 0.5):
        signal_txt = "BUY"
        status = f"지표 기반 강한 매수 신호 (RSI {rsi:.1f})"
        strength = buy_s
    elif sell_s >= max(2.0, buy_s + 0.5):
        signal_txt = "SELL"
        status = f"지표 기반 강한 매도 신호 (RSI {rsi:.1f})"
        strength = sell_s
    else:
        signal_txt = "NEUTRAL"
        status = f"지표 기반 중립 (RSI {rsi:.1f})"
        strength = max(buy_s, sell_s)

    ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else np.nan
    disparity = float((close.iloc[-1] / ma20 - 1) * 100) if ma20 == ma20 and ma20 != 0 else 0.0

    return {
        "pattern": status,
        "signal": signal_txt,
        "points": {},
        "disparity": round(float(disparity), 2),
        "rsi": round(float(rsi), 2),
        "strength": float(min(5.0, strength)),
    }

# -----------------------------
# Data fetch (yfinance)
# -----------------------------
@st.cache_data(ttl=60*60, show_spinner=False)
def get_stock_data(code: str, start_year: int):
    start_date = datetime.datetime(start_year, 1, 1)
    end_date = datetime.datetime.now()
    code6 = str(code).zfill(6)

    # 1) Prefer FinanceDataReader (KRX/Naver-backed) if available
    if fdr is not None:
        try:
            df = fdr.DataReader(code6, start_date, end_date)
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index)
                # Normalize columns to match yfinance style
                col_map = {c: c.title() for c in df.columns}
                df = df.rename(columns=col_map)
                # Ensure required columns exist
                if "Close" in df.columns:
                    return df
        except Exception:
            pass

    # 2) Fallback to yfinance (.KS/.KQ)
    for ticker in [f"{code6}.KS", f"{code6}.KQ"]:
        try:
            df = yf.Ticker(ticker).history(start=start_date, auto_adjust=False, actions=False, progress=False)
            if df is not None and not df.empty:
                return df
        except Exception:
            continue

    return pd.DataFrame()

# -----------------------------
# KOSPI200 list (Naver crawl -> cached csv)
# -----------------------------
KOSPI200_CSV = "kospi200.csv"

def load_kospi200_csv():
    if os.path.exists(KOSPI200_CSV):
        try:
            df = pd.read_csv(KOSPI200_CSV, dtype={"Code": str})
            if "Code" in df.columns:
                if "Name" not in df.columns:
                    df["Name"] = df["Code"]
                df["Code"] = df["Code"].astype(str).str.zfill(6)
                return df[["Code","Name"]].drop_duplicates(subset=["Code"]).reset_index(drop=True)
        except Exception:
            pass
    return pd.DataFrame(columns=["Code","Name"])

@st.cache_data(ttl=60*60*24, show_spinner=False)
def fetch_kospi200_from_naver():
    # Naver Finance entry list pages: 20 pages * 10 rows ≈ 200
    base = "https://finance.naver.com/sise/entryJongmok.nhn?page={}"
    headers = {"User-Agent":"Mozilla/5.0"}
    out = []
    for page in range(1, 30):  # a bit extra in case pagination changes
        r = requests.get(base.format(page), headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        # rows: <td class="ctg"><a href="/item/main.naver?code=005930">삼성전자</a></td>
        for td in soup.select("td.ctg"):
            a = td.find("a")
            if not a or not a.get("href"):
                continue
            m = re.search(r"code=(\d+)", a["href"])
            if not m:
                continue
            code = m.group(1).zfill(6)
            name = a.get_text(strip=True)
            out.append((code, name))
        # stop early if already >= 200 and no new additions
        if len(dict(out)) >= 200:
            break
    df = pd.DataFrame(list(dict(out).items()), columns=["Code","Name"])
    df["Code"] = df["Code"].astype(str).str.zfill(6)
    return df

def save_kospi200_csv(df):
    df.to_csv(KOSPI200_CSV, index=False, encoding="utf-8")

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
# Plotly chart
# -----------------------------
def make_chart(df, points, name, pattern):
    df = df.copy().reset_index()
    if "Date" not in df.columns:
        df.rename(columns={"index":"Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    p_idx = [p[0] for p in points]
    start = max(0, min(p_idx) - 20)
    end = min(len(df), max(p_idx) + 140)
    view = df.iloc[start:end].copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=view["Date"], open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"]))
    px = [df["Date"].iloc[p[0]] for p in points]
    py = [p[1] for p in points]
    fig.add_trace(go.Scatter(x=px, y=py, mode="markers+text", text=[str(i) for i in range(1,6)], textposition="top center"))
    fig.update_layout(title=f"{name} · {pattern}", height=520, margin=dict(l=10,r=10,t=50,b=10), xaxis_rangeslider_visible=False, showlegend=False)
    return fig

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="KOSPI Wolfe (KOSPI200 auto)", layout="centered")
st.markdown("<style>section.main > div { padding-top: 0.8rem; }</style>", unsafe_allow_html=True)

st.title("KOSPI Wolfe 스캐너 (모바일/Cloud)")

st.info(
    "이 버전은 **KRX(data.krx.co.kr) 호출을 하지 않습니다.**\\n"
    "- 기본 스캔 대상: repo에 포함된 `kospi200.csv`\\n"
    "- 비어있으면: 네이버금융 KOSPI200 구성종목 페이지를 크롤링해 자동 생성합니다."
)

# Universe setup
with st.expander("스캔 대상", expanded=True):
    colA, colB = st.columns([1,1])
    with colA:
        refresh = st.button("코스피200 리스트 갱신(네이버에서 재수집)", width='stretch')
    with colB:
        show_list = st.checkbox("코스피200 리스트 보기", value=False)

    kospi200 = load_kospi200_csv()
    if refresh or kospi200.empty:
        with st.spinner("네이버 금융에서 코스피200 구성종목을 가져오는 중..."):
            df_new = fetch_kospi200_from_naver()
            if len(df_new) >= 150:
                save_kospi200_csv(df_new)
                kospi200 = df_new
                st.success(f"갱신 완료: {len(kospi200)}개")
            else:
                st.error(f"가져온 종목 수가 너무 적습니다({len(df_new)}). 네이버 접속이 막혔을 수 있습니다.")

    if show_list and not kospi200.empty:
        st.dataframe(kospi200, width='stretch', hide_index=True, height=300)

    universe = kospi200.copy() if not kospi200.empty else pd.DataFrame(columns=["Code","Name"])

with st.expander("분석 설정", expanded=True):
    start_year = st.selectbox("데이터 시작 연도", [datetime.datetime.now().year - i for i in range(1, 6)], index=1)
    required_days = st.selectbox("연속 신호 기준(거래일)", [1, 2, 3], index=1)
    enable_ai = st.checkbox("AI 확률 보정(경량 모델) 사용", value=True)
    ai_horizon = st.selectbox("AI 예측 기간(거래일)", [5, 10, 20], index=1)
    st.caption("AI는 종목별 최근 데이터로 상승확률을 추정해, 현재 신호(매수/매도)의 확신도를 보조합니다.")
    max_workers = st.slider("병렬 스캔 작업수", 2, 12, 6, 1)
    name_filter = st.text_input("종목명 검색(선택)", "")
    run = st.button("스캔 실행", type="primary", width='stretch')

if name_filter.strip() and not universe.empty:
    universe = universe[universe["Name"].str.contains(name_filter.strip(), case=False, na=False)].reset_index(drop=True)

st.caption(f"대상 종목 수: {len(universe)}")

def scan(universe, start_year, max_workers, required_days):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    hist = load_hist()
    results, pending, dfs_cache = [], [], {}
    ok_cnt, fail_cnt, neutral_cnt = 0, 0, 0

    prog = st.progress(0)
    msg = st.empty()
    total = max(1, len(universe))

    def process(row):
        code = str(row["Code"]).zfill(6)
        name = str(row.get("Name", code))
        df = get_stock_data(code, start_year)
        if df is None or df.empty or len(df) < 80:
            return None
        ans = detect_wolfe_wave_pattern(df)
        if not ans:
            ans = basic_signal(df)
        pat = ans["pattern"]
        sig = ans.get("signal")
        if not sig:
            sig = "BUY" if "매수" in pat else ("SELL" if "매도" in pat else "NEUTRAL")
        update_hist(hist, code, sig, today)
        is_con, days = check_consec(hist, code, sig, required_days=required_days)
        item = {
            "Code": code, "종목명": name, "현재가": int(float(df["Close"].iloc[-1])),
            "패턴유형": pat, "신호": sig, "연속일": days,
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
            msg.write(f"스캔 중… {done}/{total} · {row.get('Name', row.get('Code'))}")
            try:
                out = f.result()
                if out is None:
                    fail_cnt += 1
                    continue
                is_con, item, df = out
                ok_cnt += 1
                if item.get("신호") == "NEUTRAL":
                    neutral_cnt += 1
                dfs_cache[item["Code"]] = df
                (results if is_con else pending).append(item)
            except Exception:
                continue

    save_hist(hist)
    msg.write("완료")
    st.info(f"데이터 수집 성공 {ok_cnt} / 실패 {fail_cnt} (중립 {neutral_cnt})")
    prog.empty()

    res_df = pd.DataFrame(results).sort_values(["강도","괴리율(%)"], ascending=[False, True]) if results else pd.DataFrame()
    pend_df = pd.DataFrame(pending).sort_values(["강도","괴리율(%)"], ascending=[False, True]) if pending else pd.DataFrame()
    return res_df, pend_df, dfs_cache

if run and not universe.empty:
    res_df, pend_df, dfs_cache = scan(universe, start_year, max_workers, required_days)
    st.session_state["res_df"] = res_df
    st.session_state["pend_df"] = pend_df
    st.session_state["dfs_cache"] = dfs_cache
elif run and universe.empty:
    st.error("스캔 대상이 비어 있습니다. 코스피200 리스트 갱신을 먼저 해주세요.")

res_df = st.session_state.get("res_df", pd.DataFrame())
pend_df = st.session_state.get("pend_df", pd.DataFrame())
dfs_cache = st.session_state.get("dfs_cache", {})

tab1, tab2, tab3 = st.tabs([f"연속({required_days}일)", "대기(1일차)", "차트"])

with tab1:
    if res_df.empty:
        st.info("연속 신호 결과가 없습니다.")
    else:
        st.dataframe(res_df[["Code","종목명","현재가","패턴유형","연속일","강도","RSI","괴리율(%)"]],
                     width='stretch', hide_index=True, height=420)

with tab2:
    if pend_df.empty:
        st.write("없음")
    else:
        st.dataframe(pend_df[["Code","종목명","현재가","패턴유형","연속일","강도","RSI","괴리율(%)"]],
                     width='stretch', hide_index=True, height=420)

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
            st.markdown(f"**{row['종목명']} ({row['Code']})** · {row['패턴유형']}")
            st.caption(f"현재가 {row['현재가']:,} | 강도 {row['강도']:.1f} | RSI {row['RSI']:.1f} | 괴리율 {row['괴리율(%)']:.2f}%")
            st.markdown(f"[네이버증권 열기](https://finance.naver.com/item/main.naver?code={row['Code']})")
            df = dfs_cache.get(code)
            if df is None or df.empty:
                df = get_stock_data(code, start_year)
            if df is None or df.empty:
                st.warning(f"데이터를 불러오지 못했습니다: {code}")
                st.stop()

            # --- AI 보정(선택) ---
            if enable_ai:
                ai = ai_probability_up(code, df, horizon=ai_horizon)
                if ai:
                    p_up = ai["p_up"]
                    acc = ai["acc"]
                    n = ai["n"]
                    st.write(f"AI 상승확률( {ai_horizon}일 ): **{p_up*100:.1f}%**  · (간이검증 정확도 {acc*100:.1f}% / 표본 {n})")
                    sig = row["패턴유형"]
                    if "매수" in sig:
                        agree = p_up
                        st.caption(f"AI가 현재 '매수' 신호에 동의할 확률: {agree*100:.1f}%")
                    elif "매도" in sig:
                        agree = 1.0 - p_up
                        st.caption(f"AI가 현재 '매도' 신호에 동의할 확률: {agree*100:.1f}%")
                else:
                    st.info("AI 보정: 학습 데이터가 부족해 확률을 계산하지 못했습니다.")
            st.plotly_chart(make_chart(df, row["points"], row["종목명"], row["패턴유형"]),
                            width='stretch')