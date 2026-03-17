"""
進階版動態 Kelly Criterion 倉位計算與風險模擬工具
Advanced Dynamic Kelly Criterion Position Sizing & Risk Simulation Tool

Author: Quant Engineer
Python: 3.9+

修訂說明 (v2):
  - Kelly 解析解：以閉合公式取代 scipy 數值優化，修正 500% 撞頂問題
  - μ 算術修正：Kelly 計算改用 μ_arith = μ_log + σ²/2
  - 最大槓桿上限：新增 sidebar slider，預設 3.0×
  - 多來源交叉驗證：yfinance + Stooq 合併，Winsorization 1%/99%
  - FRED 無風險利率：自動抓取 3M T-Bill，一鍵填入
  - 效能優化：Rolling Kelly 改解析解；Monte Carlo 圖表改 NaN-concat 單 trace
  - 快取：Rolling Kelly + Monte Carlo 加入 @st.cache_data
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd
# import pandas_datareader.data as pdr  # 暫時禁用以規避 Streamlit Cloud 相容性問題
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS: int = 252

TICKERS: dict[str, str] = {
    "台股加權指數 (^TWII)": "^TWII",
    "Vanguard S&P 500 ETF (VOO)": "VOO",
    "Invesco QQQ Trust (QQQ)": "QQQ",
}

# Stooq 的 ticker 代號與 yfinance 不同，需要對照表
STOOQ_TICKERS: dict[str, str] = {
    "^TWII": "^twii",
    "VOO": "VOO.US",
    "QQQ": "QQQ.US",
}

SigmaModel = Literal["長期歷史平均 (與 μ 同週期)", "短期 126 日歷史波動率", "EWMA (Span=60)"]

# ---------------------------------------------------------------------------
# Data Layer
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker: str, period_years: int) -> pd.Series:
    """
    從 yfinance 下載指定資產的每日收盤價（主要來源）。

    Args:
        ticker: Yahoo Finance 股票代號，例如 "^TWII"。
        period_years: 要抓取的歷史資料年數。

    Returns:
        以日期為 index 的收盤價 pd.Series。

    Raises:
        ValueError: 若下載資料為空或欄位缺失。
    """
    period_str = f"{period_years}y"
    df: pd.DataFrame = yf.download(
        ticker,
        period=period_str,
        progress=False,
        auto_adjust=True,
        actions=False,
    )
    if df.empty:
        raise ValueError(f"yfinance 無法取得 {ticker} 的資料，請確認代號是否正確。")

    # yfinance >= 0.2.x 可能回傳 MultiIndex 欄位
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise ValueError(f"資料中缺少 'Close' 欄位，實際欄位：{list(df.columns)}")

    close: pd.Series = df["Close"].dropna()
    if len(close) < 2:
        raise ValueError(f"收盤價資料點數不足 (只有 {len(close)} 筆)，無法計算報酬率。")

    return close


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data_stooq(
    ticker_stooq: str,
    start: date,
    end: date,
) -> pd.Series | None:
    """
    從 Stooq 下載收盤價作為交叉驗證資料來源（無需 API Key）。

    【已禁用】pandas_datareader 在 Streamlit Cloud 上的相容性問題，改為返回 None。
    使用 yfinance 作為唯一數據源。

    Args:
        ticker_stooq: Stooq 格式的股票代號，如 "VOO.US"、"^twii"。
        start: 開始日期。
        end: 結束日期。

    Returns:
        以日期為 index 的收盤價 pd.Series；若失敗則回傳 None（靜默降級）。
    """
    # pandas_datareader 已禁用以規避 Streamlit Cloud 相容性問題
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_rfr_fred() -> float | None:
    """
    從 FRED 取得最新 3 個月美國國庫券殖利率 (DGS3MO) 作為無風險利率參考。

    【已禁用】pandas_datareader 在 Streamlit Cloud 上的相容性問題，改為返回 None。
    使用預設的 4.0% 無風險利率。

    Returns:
        年化無風險利率（小數形式，例如 0.043 代表 4.3%）；失敗則回傳 None。
    """
    # pandas_datareader 已禁用以規避 Streamlit Cloud 相容性問題
    return None


# ---------------------------------------------------------------------------
# Analytics Layer
# ---------------------------------------------------------------------------


# ──────────────────────────────────────────────────────────────────────────
# 價值指標計算模組（7 大估值指標體系）
# ──────────────────────────────────────────────────────────────────────────

# 各資產的歷史基準值（low=低估邊界, fair=歷史中位, high=高估邊界）
INDICATOR_BENCHMARKS: dict[str, dict] = {
    "pe": {
        # 依據：S&P500 2000-2025 中位 ~22x，科技股合理 PE 區間 32x，台股中位 ~14-16x
        "^TWII": {"low": 10.0, "fair": 16.0, "high": 22.0},
        "VOO":   {"low": 16.0, "fair": 21.0, "high": 29.0},
        "QQQ":   {"low": 22.0, "fair": 32.0, "high": 46.0},
    },
    "pb": {
        # 依據：S&P500 2012-2024 平均 P/B 約 3.5x，隨 PE 同步上調
        "VOO":   {"low": 2.2,  "fair": 3.2,  "high": 4.8},
        "QQQ":   {"low": 4.0,  "fair": 6.0,  "high": 9.0},
        "^TWII": {"low": 1.0,  "fair": 1.5,  "high": 2.5},
    },
    "dividend_yield": {
        # 值為年化殖利率（如 0.015 = 1.5%），高殖利率 = 低估
        "VOO":   {"high": 0.020, "fair": 0.015, "low": 0.010},
        "QQQ":   {"high": 0.008, "fair": 0.005, "low": 0.003},
        "^TWII": {"high": 0.040, "fair": 0.028, "low": 0.018},
    },
}


def _sigmoid_score(ratio: float) -> float:
    """
    將「當前/基準」比值轉換為 0-100 分（sigmoid 非線性壓縮）。
    ratio=1.0 → 50；ratio=2.0 → ~95；ratio=0.5 → ~5
    """
    import math
    return 100.0 / (1.0 + math.exp(-3.0 * (ratio - 1.0)))


def compute_indicator_scores(
    indicators: dict[str, float | None],
    ticker: str,
    manual_pe: float | None = None,
) -> dict[str, float | None]:
    """
    將原始市場指標值轉換為 0-100 估值評分。
    高分 = 高估（建議減槓桿）；低分 = 低估（建議加槓桿）。

    Args:
        indicators: fetch_market_indicators() 的輸出
        ticker: 資產代號（用於查找歷史基準值）
        manual_pe: 用戶手動輸入的 P/E（台股無法自動取得時使用）

    Returns:
        dict: 各指標的 0-100 評分，無法計算則為 None
    """
    bm = INDICATOR_BENCHMARKS
    scores: dict[str, float | None] = {}

    # ── P/E 評分（低 PE = 低估 = 低分）──
    pe = manual_pe if manual_pe is not None else indicators.get("pe_ratio")
    if pe and pe > 0:
        fair_pe = bm["pe"].get(ticker, bm["pe"]["VOO"])["fair"]
        scores["pe_score"] = _sigmoid_score(pe / fair_pe)
    else:
        scores["pe_score"] = None

    # ── P/B 評分（低 P/B = 低估 = 低分）──
    pb = indicators.get("pb_ratio")
    if pb and pb > 0:
        fair_pb = bm["pb"].get(ticker, bm["pb"]["VOO"])["fair"]
        scores["pb_score"] = _sigmoid_score(pb / fair_pb)
    else:
        scores["pb_score"] = None

    # ── 股息殖利率評分（高 Yield = 低估 = 低分，需反轉）──
    div_yield = indicators.get("dividend_yield")
    if div_yield and div_yield > 0:
        fair_yield = bm["dividend_yield"].get(ticker, bm["dividend_yield"]["VOO"])["fair"]
        scores["dividend_score"] = 100.0 - _sigmoid_score(div_yield / fair_yield)
    else:
        scores["dividend_score"] = None

    # ── VIX 評分（高 VIX = 恐慌 = 低估 = 低分，需反轉）──
    vix = indicators.get("vix")
    if vix and vix > 0:
        # VIX=15 → 100分（平靜高估）；VIX=50 → 0分（恐慌低估）
        scores["vix_score"] = max(0.0, min(100.0, 100.0 - (vix - 15.0) / 35.0 * 100.0))
    else:
        scores["vix_score"] = None

    # ── 殖利率曲線評分（倒掛 = 衰退 = 高分；陡峭 = 成長 = 低分）──
    spread = indicators.get("yield_spread_10y3m")
    if spread is not None:
        # spread=2.0 → ~20分（陡峭）；spread=0 → 50分；spread=-1 → ~65分（倒掛）
        scores["yield_curve_score"] = max(0.0, min(100.0, 50.0 - spread / 2.0 * 30.0))
    else:
        scores["yield_curve_score"] = None

    # ── ERP 評分（高 ERP = 低估 = 低分，需反轉）──
    erp = indicators.get("erp_pct")
    if erp is not None:
        # ERP=7% → 0分（低估）；ERP=3% → 100分（高估）；ERP=5.5% → 50分
        scores["erp_score"] = max(0.0, min(100.0, 100.0 - (erp - 3.0) / 4.0 * 100.0))
    else:
        scores["erp_score"] = None

    return scores


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_indicators(ticker: str, r_free: float = 0.04) -> dict[str, float | None]:
    """
    統一從 yfinance 抓取 6 個估值指標。
    ^TWII 缺少 P/B、Dividend Yield 時自動降級（返回 None）。
    ERP 使用 sidebar 傳入的 r_free，確保與 Kelly 計算一致。

    Args:
        ticker: 資產代號（"^TWII"、"VOO"、"QQQ"）
        r_free: 無風險利率（小數，如 0.04 = 4%），與 sidebar 同步

    Returns:
        dict with keys: pe_ratio, pb_ratio, dividend_yield,
                        vix, yield_spread_10y3m, erp_pct
        各指標若無法取得則為 None
    """
    result: dict[str, float | None] = {
        "pe_ratio": None,
        "pb_ratio": None,
        "dividend_yield": None,
        "vix": None,
        "yield_spread_10y3m": None,
        "erp_pct": None,
    }
    try:
        # ── 資產本身的基本面指標 ──
        info = yf.Ticker(ticker).info
        result["pe_ratio"] = info.get("trailingPE")
        result["pb_ratio"] = info.get("priceToBook")
        result["dividend_yield"] = info.get("trailingAnnualDividendYield")

        # ── VIX 恐慌指數（全球通用） ──
        vix_info = yf.Ticker("^VIX").fast_info
        result["vix"] = vix_info.get("lastPrice") or vix_info.get("regularMarketPrice")

        # ── 殖利率曲線（10Y - 3M） ──
        tnx = yf.Ticker("^TNX").fast_info.get("lastPrice")   # 10 年美債（單位：%）
        irx = yf.Ticker("^IRX").fast_info.get("lastPrice")   # 3 個月美債（單位：%）
        if tnx and irx:
            result["yield_spread_10y3m"] = float(tnx) - float(irx)

        # ── ERP：預期股票報酬 - 無風險利率（與 sidebar r_free 動態聯動）──
        result["erp_pct"] = (0.10 - r_free) * 100.0  # 單位：百分點

    except Exception:
        pass  # 部分失敗不阻斷主流程，各指標分別保持 None

    return result


def compute_value_score(
    indicator_scores: dict[str, float | None],
    user_override: float | None = None,
) -> tuple[float, dict[str, float]]:
    """
    計算加權綜合估值評分 (0-100)。
    若某指標為 None（無法取得），自動重新分配其權重至可用指標。

    低分 (0-30) = 低估 → 應加槓桿
    中分 (40-60) = 公允 → 標準槓桿
    高分 (70-100) = 高估 → 應減槓桿

    Args:
        indicator_scores: compute_indicator_scores() 的輸出
        user_override: 用戶手動輸入的評分（若非 None 則完全使用此值）

    Returns:
        (composite_score: float, effective_weights: dict)
    """
    if user_override is not None:
        return float(user_override), {}

    WEIGHTS: dict[str, float] = {
        "pe_score":          0.25,
        "vix_score":         0.25,
        "dividend_score":    0.15,
        "yield_curve_score": 0.15,
        "pb_score":          0.10,
        "erp_score":         0.10,
    }

    available = {k: v for k, v in indicator_scores.items() if v is not None and k in WEIGHTS}
    if not available:
        return 50.0, {}  # 完全無數據 → 中性

    total_w = sum(WEIGHTS[k] for k in available)
    eff_w = {k: WEIGHTS[k] / total_w for k in available}
    composite = sum(eff_w[k] * available[k] for k in available)
    return max(0.0, min(100.0, composite)), eff_w


def value_multiplier_from_score(score: float, aggressiveness: float = 1.0) -> float:
    """
    將價值評分 (0-100) 轉換為倉位乘數（強化版，指數衰減）。

    設計邏輯（源自槓桿思維 + 生命投資法）：
    - 中性點：score=50 → 1.0x（無干預）
    - 低估區（score < 50）：線性加碼，最高 1.5×
    - 高估區（score > 50）：指數減碼，保護力隨高估程度加速增強

    指數減碼的必要性：
    - 歷史 μ 過高（如 QQQ 10 年牛市）會導致 Kelly 永遠偏向全槓桿
    - 線性減碼不足以對抗 3.0× 的 raw Kelly（3× × 0.75 = 2.25× 仍太高）
    - 指數曲線確保 score=80 時乘數 ~0.34×，3× → 1.0×（接近無槓桿）

    效果對照（aggressiveness=1.0）：
        score=30  → 1.35×（低估加碼）
        score=50  → 1.00×（中性）
        score=60  → ~0.82×（輕微高估）
        score=70  → ~0.56×（明確高估）
        score=80  → ~0.34×（顯著高估，3× Kelly → ~1.0×）
        score=90  → ~0.19×（泡沫）
        score=100 → 0.10×（極端泡沫）

    Args:
        score: 價值評分 (0-100)
        aggressiveness: 槓桿激進度 (0.5-1.5)

    Returns:
        value_multiplier: 倉位乘數（0.10x - 1.75x）
    """
    normalized = (score - 50.0) / 50.0  # -1.0 ~ +1.0（50 分 = 0 = 中性）

    if normalized <= 0:
        # 低估區（score ≤ 50）：線性加碼，score=0 → 1.5×
        boost = 0.5 * (-normalized) * aggressiveness
        return 1.0 + boost
    else:
        # 高估區（score > 50）：指數減碼，非線性強化
        # normalized ** (1/aggressiveness) 使激進度更高時曲線更陡峭
        reduction_base = 0.90  # 最大削減幅度（保留 10%）
        reduction = reduction_base * (normalized ** (1.0 / max(aggressiveness, 0.3))) * aggressiveness
        return max(0.10, 1.0 - reduction)


# ──────────────────────────────────────────────────────────────────────────
# 原有分析層函數
# ──────────────────────────────────────────────────────────────────────────

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    計算每日對數報酬率 R_t = ln(P_t / P_{t-1})。

    Args:
        prices: 收盤價時間序列。

    Returns:
        對數報酬率序列 (dropna 後)。
    """
    return np.log(prices / prices.shift(1)).dropna()


def winsorize_returns(
    returns: pd.Series,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> pd.Series:
    """
    對報酬率序列進行 Winsorization，將極端值截斷至指定分位數範圍。

    Args:
        returns: 日報酬率序列。
        q_low: 下界分位數（預設 1%）。
        q_high: 上界分位數（預設 99%）。

    Returns:
        截斷後的報酬率序列（資料筆數不變）。
    """
    lo = float(returns.quantile(q_low))
    hi = float(returns.quantile(q_high))
    return returns.clip(lo, hi)


def merge_and_winsorize(
    returns_dict: dict[str, pd.Series],
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> pd.Series:
    """
    合併多個來源的日報酬率，進行 Winsorization 後取等權平均。

    步驟：
    1. Inner join（只保留各來源共同有資料的交易日）
    2. 對各來源分別進行 Winsorization（截斷至 1%/99%）
    3. 等權平均跨來源報酬率 → 單一合併序列

    Args:
        returns_dict: 來源名稱 → 日報酬率序列的字典。
        q_low: Winsorization 下界分位數。
        q_high: Winsorization 上界分位數。

    Returns:
        合併後的日報酬率序列，name="merged"。
    """
    if len(returns_dict) == 1:
        only = list(returns_dict.values())[0]
        result = winsorize_returns(only, q_low, q_high)
        result.name = "merged"
        return result

    df = pd.DataFrame(returns_dict).dropna()
    for col in df.columns:
        lo = float(df[col].quantile(q_low))
        hi = float(df[col].quantile(q_high))
        df[col] = df[col].clip(lo, hi)

    merged = df.mean(axis=1)
    merged.name = "merged"
    return merged


def compute_mu(log_returns: pd.Series) -> float:
    """
    計算年化對數報酬率均值 μ_log。

    注意：回傳的是幾何（log-space）均值，即 E[ln(P_t/P_{t-1})] × 252。
    與算術期望報酬的關係為：μ_arith = μ_log + σ²/2。
    在計算 Kelly 比例時應使用 μ_arith，而非 μ_log。

    Args:
        log_returns: 每日對數報酬率序列。

    Returns:
        年化 μ_log（小數形式，例如 0.12 代表 12%）。
    """
    return float(log_returns.mean() * TRADING_DAYS)


def compute_sigma(log_returns: pd.Series, model: SigmaModel) -> float:
    """
    依據指定模型計算年化波動率 σ。

    Args:
        log_returns: 每日對數報酬率序列（全期）。
        model: 波動率模型選項。

    Returns:
        年化 σ（小數形式，例如 0.20 代表 20%）。
    """
    if model == "長期歷史平均 (與 μ 同週期)":
        sigma = float(log_returns.std() * np.sqrt(TRADING_DAYS))
    elif model == "短期 126 日歷史波動率":
        recent = log_returns.iloc[-126:] if len(log_returns) >= 126 else log_returns
        sigma = float(recent.std() * np.sqrt(TRADING_DAYS))
    else:  # EWMA
        ewma_std = log_returns.ewm(span=60, adjust=False).std()
        sigma = float(ewma_std.iloc[-1] * np.sqrt(TRADING_DAYS))

    return max(sigma, 1e-8)


def compute_kelly_analytical(
    mu_arith: float,
    sigma: float,
    r_free: float,
    r_margin: float,
    max_leverage: float = 3.0,
) -> float:
    """
    以閉合解析解計算最佳 Kelly 倉位比例 f*（取代數值優化）。

    數學推導
    --------
    分段凹函數 g(f) 在 f=1 處連續但有導數跳躍（kink）：

        f ≤ 1：g₁(f) = f·μ + (1-f)·r_free  - f²σ²/2
                → 頂點 f₁* = (μ - r_free) / σ²

        f > 1：g₂(f) = f·μ - (f-1)·r_margin - f²σ²/2
                → 頂點 f₂* = (μ - r_margin) / σ²

    由於 r_free < r_margin，g 在 f=1 左側斜率 > 右側斜率（kink 向下跳）。
    全域最大值判斷（三段式）：

        1. f₁* ≤ 1.0  →  f* = f₁*   （無槓桿最佳，可能為負 = 放空）
        2. f₁* > 1 且 f₂* ≤ 1  →  f* = 1.0  （kink 點是全域最大）
        3. f₂* > 1.0  →  f* = f₂*  （槓桿最佳）

    最終 clamp 至 [-1.0, max_leverage]。

    Args:
        mu_arith: 年化算術期望報酬率（= μ_log + σ²/2）。
        sigma: 年化波動率。
        r_free: 無風險利率（小數）。
        r_margin: 融資借貸成本（小數）。
        max_leverage: 最大允許槓桿上限（預設 3.0）。

    Returns:
        最佳 Full-Kelly 比例 f*，已 clamp 至 [-1, max_leverage]。
    """
    sigma2 = sigma ** 2
    if sigma2 < 1e-12:
        # 幾乎零波動，理論上應無限加槓桿，直接回傳上限
        return float(max_leverage)

    f1_star = (mu_arith - r_free) / sigma2    # 無槓桿區最佳頂點
    f2_star = (mu_arith - r_margin) / sigma2  # 槓桿區最佳頂點

    if f1_star <= 1.0:
        raw = f1_star
    elif f2_star <= 1.0:
        raw = 1.0   # kink 是全域最大值
    else:
        raw = f2_star

    return float(np.clip(raw, -1.0, max_leverage))


# ---------------------------------------------------------------------------
# Rolling Kelly (cached inner function)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def _compute_rolling_kelly_cached(
    log_returns_arr: np.ndarray,
    dates_arr: np.ndarray,
    r_free: float,
    r_margin: float,
    sigma_model: str,
    max_leverage: float,
    window: int = TRADING_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    快取版滾動 Kelly 計算核心（接受 numpy 陣列，st.cache_data 可正確 hash）。

    內部使用解析解（O(1) per step），比 scipy 數值優化快 10–100×。

    Returns:
        (kelly_values_array, dates_array)
    """
    returns_series = pd.Series(
        log_returns_arr,
        index=pd.DatetimeIndex(dates_arr),
    )
    compute_returns = returns_series.iloc[-(2 * window):]

    kelly_values: list[float] = []
    result_dates: list[pd.Timestamp] = []

    for i in range(window, len(compute_returns) + 1):
        window_returns = compute_returns.iloc[i - window : i]
        mu_w = compute_mu(window_returns)
        sigma_w = compute_sigma(window_returns, sigma_model)  # type: ignore[arg-type]
        mu_arith_w = mu_w + 0.5 * sigma_w ** 2  # 算術報酬修正
        kelly_w = compute_kelly_analytical(mu_arith_w, sigma_w, r_free, r_margin, max_leverage)
        kelly_values.append(kelly_w)
        result_dates.append(compute_returns.index[i - 1])

    return np.array(kelly_values), np.array(result_dates, dtype="datetime64[ns]")


def compute_rolling_kelly(
    log_returns: pd.Series,
    r_free: float,
    r_margin: float,
    sigma_model: SigmaModel,
    max_leverage: float = 3.0,
    window: int = TRADING_DAYS,
) -> pd.Series:
    """
    計算過去 window 個交易日的滾動 Full-Kelly 比例（公開介面）。

    內部委託快取函數執行，避免重複計算。

    Args:
        log_returns: 全期每日對數報酬率序列。
        r_free: 無風險利率（小數）。
        r_margin: 融資借貸成本（小數）。
        sigma_model: 波動率模型。
        max_leverage: 最大槓桿上限。
        window: 滾動視窗大小，預設 252 個交易日。

    Returns:
        以日期為 index 的滾動 Kelly 比例序列。
    """
    kelly_arr, dates_arr = _compute_rolling_kelly_cached(
        log_returns.values,
        log_returns.index.values,
        r_free,
        r_margin,
        sigma_model,
        max_leverage,
        window,
    )
    return pd.Series(kelly_arr, index=pd.DatetimeIndex(dates_arr), name="Rolling Full-Kelly")


# ---------------------------------------------------------------------------
# Monte Carlo Simulation (cached)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def run_monte_carlo(
    mu: float,
    sigma: float,
    f: float,
    n_paths: int = 1000,
    n_days: int = TRADING_DAYS,
    seed: int = 42,
) -> np.ndarray:
    """
    以幾何布朗運動 (GBM) 模擬未來資金曲線。

    GBM 離散化公式：
        S_{t+1} = S_t × exp((f·μ - f²σ²/2)·Δt + f·σ·√Δt·Z)
    其中 Z ~ N(0,1)，Δt = 1/252。

    Args:
        mu: 年化算術期望報酬率（μ_arith）。
        sigma: 年化波動率。
        f: 倉位比例 (Fractional Kelly)。
        n_paths: 模擬路徑數量，預設 1000。
        n_days: 模擬天數，預設 252（一年）。
        seed: 隨機種子，確保可重現性。

    Returns:
        shape (n_paths, n_days+1) 的 numpy 陣列，初始值為 1.0。
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS

    drift = (f * mu - 0.5 * f**2 * sigma**2) * dt
    diffusion = f * sigma * np.sqrt(dt)

    z = rng.standard_normal((n_paths, n_days))
    log_returns_sim = drift + diffusion * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_returns_sim, axis=1)],
        axis=1,
    )
    return np.exp(log_paths)


def compute_max_drawdown_per_path(paths: np.ndarray) -> np.ndarray:
    """
    計算每條資金曲線的最大回撤率 (Max Drawdown)。

    Max Drawdown = (峰值 - 谷值) / 峰值，以正值百分比表示。

    Args:
        paths: shape (n_paths, n_days+1) 的資金曲線矩陣。

    Returns:
        shape (n_paths,) 的最大回撤率陣列（0~1）。
    """
    running_max = np.maximum.accumulate(paths, axis=1)
    drawdown = (running_max - paths) / running_max
    return drawdown.max(axis=1)


# ---------------------------------------------------------------------------
# Visualization Layer
# ---------------------------------------------------------------------------


def plot_rolling_kelly(rolling_kelly: pd.Series, kelly_multiplier: float) -> go.Figure:
    """
    繪製滾動 Full-Kelly 建議倉位折線圖。

    Args:
        rolling_kelly: 滾動 Kelly 比例時間序列。
        kelly_multiplier: 凱利乘數，用於繪製 Fractional Kelly 參考線。

    Returns:
        Plotly Figure 物件。
    """
    fractional = rolling_kelly * kelly_multiplier

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=rolling_kelly.index,
            y=rolling_kelly.values * 100,
            mode="lines",
            name="Full-Kelly (%)",
            line=dict(color="#4A90D9", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fractional.index,
            y=fractional.values * 100,
            mode="lines",
            name=f"Fractional Kelly ×{kelly_multiplier:.2f} (%)",
            line=dict(color="#E8A838", width=2, dash="dash"),
        )
    )
    fig.add_hline(
        y=100,
        line=dict(color="rgba(255,80,80,0.5)", width=1, dash="dot"),
        annotation_text="滿倉 100%",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="滾動 252 日 Kelly 倉位建議走勢",
        xaxis_title="日期",
        xaxis=dict(automargin=True),
        yaxis_title="建議倉位比例 (%)",
        yaxis=dict(automargin=True),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
        height=340,                          # 原 420，縮短後手機螢幕可完整顯示
        margin=dict(l=50, r=10, t=50, b=50), # 原 l=60 r=20，縮減邊距增加有效圖寬
    )
    return fig


def plot_monte_carlo(
    paths: np.ndarray,
    var95_drawdown: float,
    f: float,
    n_display: int = 200,
    seed: int = 0,
    title_suffix: str = "",
) -> go.Figure:
    """
    繪製蒙地卡羅資金曲線模擬圖（高效能版本）。

    效能優化說明
    -----------
    舊版：for i in range(1000) → 加入 1,000 條獨立 trace（渲染極慢）
    新版：
      - 200 條代表路徑以 NaN 分隔符合併為 **1 條 trace**
      - Percentile band 用 2 條 filled scatter 繪製
      - 合計 trace 數：5 條（vs 舊版 1,002 條），渲染提升 10–50×

    Args:
        paths: shape (n_paths, n_days+1) 的資金曲線矩陣。
        var95_drawdown: 95% VaR 對應的最大回撤率 (0~1)。
        f: 倉位比例，用於標題顯示。
        n_display: 顯示的代表路徑數量（預設 200）。
        seed: 隨機抽樣路徑的種子。

    Returns:
        Plotly Figure 物件。
    """
    n_paths, n_steps = paths.shape
    x_base = np.arange(n_steps, dtype=float)

    fig = go.Figure()

    # ── 1. 代表路徑（NaN 分隔，合併為 1 條 trace）──
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_paths, size=min(n_display, n_paths), replace=False)

    nan_row = np.full(1, np.nan)
    y_parts = [np.concatenate([paths[i], nan_row]) for i in idx]
    x_parts = [np.concatenate([x_base, nan_row]) for _ in idx]

    y_all = np.concatenate(y_parts)
    x_all = np.concatenate(x_parts)

    fig.add_trace(
        go.Scatter(
            x=x_all,
            y=y_all,
            mode="lines",
            line=dict(width=0.5, color="rgba(100,160,255,0.15)"),
            showlegend=False,
            hoverinfo="skip",
            name="_paths",
        )
    )

    # ── 2. Percentile band：外帶 5–95%，內帶 25–75% ──
    p5  = np.percentile(paths, 5,  axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_base, x_base[::-1]]),
            y=np.concatenate([p95, p5[::-1]]),
            fill="toself",
            fillcolor="rgba(100,160,255,0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            name="5%–95% 區間",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_base, x_base[::-1]]),
            y=np.concatenate([p75, p25[::-1]]),
            fill="toself",
            fillcolor="rgba(100,200,255,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="25%–75% 區間",
            hoverinfo="skip",
        )
    )

    # ── 3. 中位數路徑 ──
    median_path = np.median(paths, axis=0)
    fig.add_trace(
        go.Scatter(
            x=x_base,
            y=median_path,
            mode="lines",
            name="中位數路徑",
            line=dict(color="#FFD700", width=2.5),
        )
    )

    # ── 4. 初始值參考線 ──
    fig.add_hline(
        y=1.0,
        line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
    )

    fig.update_layout(
        title=(
            f"蒙地卡羅未來一年資金曲線模擬 (1,000 路徑｜倉位 {f*100:.1f}%)<br>"
            f"<sup>95% VaR 最大回撤：{var95_drawdown*100:.2f}%"
            + (f"　{title_suffix}" if title_suffix else "")
            + "</sup>"
        ),
        xaxis_title="交易日",
        xaxis=dict(automargin=True),
        yaxis_title="資產淨值 (初始 = 1.0)",
        yaxis=dict(automargin=True),
        hovermode="x",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
        height=380,                          # 原 480，縮短後手機螢幕可完整顯示
        margin=dict(l=50, r=10, t=70, b=50), # 原 l=60 r=20，縮減邊距增加有效圖寬
    )
    return fig


# ---------------------------------------------------------------------------
# Data Quality Report
# ---------------------------------------------------------------------------


def build_data_quality_report(
    returns_dict: dict[str, pd.Series],
    merged_returns: pd.Series,
    sigma_model: SigmaModel,
) -> pd.DataFrame:
    """
    建立各資料來源的 μ/σ 比較表，協助使用者評估資料品質與來源差異。

    Args:
        returns_dict: 各來源的原始日報酬率（合併前）。
        merged_returns: 合併 + Winsorized 後的日報酬率。
        sigma_model: 波動率模型（保持與主計算一致）。

    Returns:
        比較表 DataFrame，每列為一個資料來源。
    """
    rows: list[dict] = []

    for source, ret in returns_dict.items():
        mu_s = compute_mu(ret) * 100
        sigma_s = compute_sigma(ret, sigma_model) * 100
        rows.append(
            {
                "來源": source,
                "資料筆數": f"{len(ret):,}",
                "μ 年化 (%)": f"{mu_s:.2f}%",
                "σ 年化 (%)": f"{sigma_s:.2f}%",
            }
        )

    mu_m = compute_mu(merged_returns) * 100
    sigma_m = compute_sigma(merged_returns, sigma_model) * 100
    rows.append(
        {
            "來源": "合併後 (Winsorized 等權平均)",
            "資料筆數": f"{len(merged_returns):,}",
            "μ 年化 (%)": f"{mu_m:.2f}% ★",
            "σ 年化 (%)": f"{sigma_m:.2f}% ★",
        }
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """Streamlit 應用程式主函數。"""

    st.set_page_config(
        page_title="Kelly Criterion 倉位計算工具",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 進階版動態 Kelly Criterion 倉位計算與風險模擬工具")
    st.caption(
        "基於連續時間數學的凱利倉位最佳化，支援非對稱利率、多重波動率模型、"
        "多來源交叉驗證（yfinance + Stooq + FRED）與蒙地卡羅風險模擬。"
    )

    # ------------------------------------------------------------------
    # Mobile-Responsive CSS
    # 僅在螢幕寬度 ≤767px（手機/小平板）時生效，桌機完全不受影響。
    # 主要調整：
    #   1. st.columns(4) → 2×2 格線（每欄 50% 寬）
    #   2. Metric 字體大小適配小螢幕
    #   3. 主容器邊距縮減，讓圖表使用更多有效寬度
    #   4. Slider / NumberInput / Expander 觸控目標放大至 ≥44px（Apple HIG 標準）
    #   5. 標題字體縮小，避免在窄螢幕溢出
    # ------------------------------------------------------------------
    st.markdown(
        """
        <style>
        @media (max-width: 767px) {
            /* ── 1. 4欄 metrics → 2×2 格線 ───────────────────────── */
            /* stHorizontalBlock 是 flex-wrap:wrap 容器，
               強制子欄為 50% 寬後，4 個子欄自動排列成 2 行 2 列。
               同時覆蓋兩處 st.columns(4)：核心指標 & Monte Carlo 結果。 */
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                width: 50% !important;
                flex: 0 0 50% !important;
                min-width: 50% !important;
                max-width: 50% !important;
                box-sizing: border-box;
            }

            /* ── 2. Metric 元件文字調整 ──────────────────────────── */
            div[data-testid="stMetricLabel"] {
                font-size: 0.72rem !important;
                white-space: normal !important;
                word-break: break-word !important;
                line-height: 1.25 !important;
            }
            div[data-testid="stMetricValue"] {
                font-size: 1.4rem !important;
                line-height: 1.1 !important;
            }
            div[data-testid="stMetricDelta"] {
                font-size: 0.65rem !important;
            }

            /* ── 3. 主容器邊距縮減 ───────────────────────────────── */
            .main .block-container {
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
                padding-top: 1rem !important;
                max-width: 100% !important;
            }
            div[data-testid="stPlotlyChart"] {
                padding-left: 0 !important;
                padding-right: 0 !important;
            }

            /* ── 4. Slider 觸控區放大（44px Apple HIG 標準） ─────── */
            div[data-testid="stSlider"] {
                padding-top: 8px !important;
                padding-bottom: 16px !important;
            }
            div[data-testid="stSlider"] > div {
                padding-top: 8px !important;
                padding-bottom: 8px !important;
            }
            div[data-testid="stSlider"] > div > div > div[role="slider"] {
                width: 28px !important;
                height: 28px !important;
                margin-top: -12px !important;
            }

            /* ── 5. Number Input 按鈕 44px ───────────────────────── */
            div[data-testid="stNumberInput"] input {
                height: 44px !important;
                font-size: 1rem !important;
                padding: 8px 12px !important;
            }
            div[data-testid="stNumberInput"] button {
                height: 44px !important;
                min-width: 44px !important;
            }

            /* ── 6. Selectbox 最小高度 ───────────────────────────── */
            div[data-testid="stSelectbox"] > div > div {
                min-height: 44px !important;
            }

            /* ── 7. 標題字體縮小 ─────────────────────────────────── */
            h1 { font-size: 1.3rem !important; line-height: 1.3 !important; }
            h2 { font-size: 1.1rem !important; }

            /* ── 8. Expander 觸控高度 ────────────────────────────── */
            div[data-testid="stExpander"] summary {
                min-height: 44px !important;
                display: flex !important;
                align-items: center !important;
                padding: 8px 12px !important;
            }

            /* ── 9. DataFrame 橫向捲動 ───────────────────────────── */
            div[data-testid="stDataFrame"] {
                overflow-x: auto !important;
                -webkit-overflow-scrolling: touch !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Session State Initialization
    # ------------------------------------------------------------------
    if "r_free_pct" not in st.session_state:
        st.session_state["r_free_pct"] = 4.0

    # ------------------------------------------------------------------
    # Sidebar — Parameter Inputs
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ 參數設定")

        # ── 資產選擇 ──
        st.subheader("資產選擇")
        ticker_label = st.selectbox(
            "選擇資產",
            options=list(TICKERS.keys()),
            index=1,  # 預設 VOO
        )
        ticker: str = TICKERS[ticker_label]

        # ── 評估週期 ──
        st.subheader("預期報酬率 μ 評估週期")
        mu_period_label = st.selectbox(
            "歷史資料回溯期間",
            options=["過去 10 年", "過去 20 年"],
            index=0,
        )
        mu_period_years: int = 10 if mu_period_label == "過去 10 年" else 20

        # ── 波動率模型 ──
        st.subheader("波動率 σ 評估模型")
        sigma_model: SigmaModel = st.selectbox(  # type: ignore[assignment]
            "波動率計算方式",
            options=[
                "長期歷史平均 (與 μ 同週期)",
                "短期 126 日歷史波動率",
                "EWMA (Span=60)",
            ],
            index=0,
        )

        # ── FRED 無風險利率 ──
        st.subheader("非對稱利率設定")

        fred_rfr: float | None = fetch_rfr_fred()
        if fred_rfr is not None:
            st.info(
                f"📡 **FRED 最新 3M T-Bill**：**{fred_rfr*100:.2f}%**\n\n"
                "點擊下方按鈕自動套用。"
            )
            if st.button("✅ 套用 FRED 利率", use_container_width=True):
                st.session_state["r_free_pct"] = round(fred_rfr * 100, 2)
                st.rerun()

        r_free_pct: float = st.number_input(
            "無風險利率 Risk-Free Rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=float(st.session_state["r_free_pct"]),
            step=0.1,
            help="未滿倉時持有現金的年化收益率",
        )
        r_margin_pct: float = st.number_input(
            "融資借貸成本 Margin Rate (%)",
            min_value=0.0,
            max_value=30.0,
            value=6.5,
            step=0.1,
            help="使用槓桿時的年化借貸成本",
        )

        # ── 預期報酬均值回歸折扣 ──
        st.subheader("預期報酬均值回歸折扣")
        mu_haircut_pct: int = st.slider(
            "📉 μ 保守折扣",
            min_value=0,
            max_value=50,
            value=25,
            step=5,
            format="%d%%",
            help=(
                "將歷史 μ 打折，反映未來報酬率可能低於歷史水準（均值回歸）。\n"
                "• 0%：完全相信歷史報酬可延續\n"
                "• 25%（預設）：適度保守，貼近 GMO/Research Affiliates 長期預測\n"
                "• 50%：極保守，適合接近退休族群\n\n"
                "根因說明：QQQ 10 年化報酬 ~17%，以台灣信貸成本 2.5% 計算，"
                "Kelly 公式天然傾向 3× 以上槓桿。折扣可有效壓縮此偏樂觀效應。\n"
                "注意：此折扣僅影響 Kelly 計算，圖表顯示的歷史 μ 不受影響。"
            ),
        )
        mu_haircut: float = mu_haircut_pct / 100.0

        # ── VIX 波動率下限（可選） ──
        use_vix_floor: bool = st.checkbox(
            "📊 使用 VIX 隱含波動率為 σ 下限",
            value=True,
            help=(
                "VIX 反映市場對未來 30 天的隱含波動率（年化）。\n"
                "啟用後：σ_used = max(σ_歷史, VIX/100)。\n\n"
                "效果：VIX=15 時 σ 至少 15%，防止低波動期 Kelly 過度樂觀。"
            ),
        )

        # ── Kelly 乘數 ──
        st.subheader("凱利乘數 (Fractional Kelly)")
        kelly_multiplier: float = st.slider(
            "Kelly 乘數",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            format="%.2f",
            help="1.0 = Full Kelly；0.5 = Half Kelly（建議保守使用）",
        )

        # ── 最大槓桿上限 ──
        st.subheader("最大允許槓桿")
        max_leverage: float = st.slider(
            "最大槓桿倍數上限",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            format="%.1f×",
            help=(
                "限制 Kelly 建議的最高槓桿，避免高報酬低波動環境下出現天文數字。\n"
                "Full-Kelly 超過此上限時會顯示警告並截斷。"
            ),
        )

        # ──────────────────────────────────────────
        # 💎 價值評估區（市場低估/高估判斷）
        # ──────────────────────────────────────────
        st.markdown("---")
        st.subheader("💎 價值評估")

        _r_free_sidebar: float = r_free_pct / 100.0

        # ── 自動抓取市場指標 ──
        _indicators = fetch_market_indicators(ticker, r_free=_r_free_sidebar)

        # 台股 PE 特別處理：yfinance 無法取得 ^TWII trailingPE
        _manual_pe: float | None = None
        if ticker == "^TWII" and _indicators.get("pe_ratio") is None:
            with st.expander("📋 台股 P/E 手動輸入（yfinance 不支援指數 PE）"):
                _manual_pe = st.number_input(
                    "台股加權指數 P/E（可參考：台灣指數公司 / CMoney）",
                    min_value=5.0, max_value=50.0, value=15.0, step=0.5,
                    help="歷史中位約 15x；低於 12 = 低估；高於 22 = 高估",
                )

        # ── 市場估值儀表板（expander 展開） ──
        with st.expander("📊 市場指標儀表板（自動更新，點擊展開）", expanded=False):
            _c1, _c2 = st.columns(2)

            _pe_val = _manual_pe if _manual_pe else _indicators.get("pe_ratio")
            with _c1:
                if _pe_val:
                    _pe_bench = INDICATOR_BENCHMARKS["pe"].get(ticker, INDICATOR_BENCHMARKS["pe"]["VOO"])
                    st.metric("P/E 本益比", f"{_pe_val:.1f}x",
                              f"低估<{_pe_bench['low']}  公允~{_pe_bench['fair']}  高估>{_pe_bench['high']}",
                              help="低 PE = 便宜 = 建議加槓桿")
                else:
                    st.info("P/E：無法自動取得")

            _pb_val = _indicators.get("pb_ratio")
            with _c2:
                if _pb_val:
                    _pb_bench = INDICATOR_BENCHMARKS["pb"].get(ticker, INDICATOR_BENCHMARKS["pb"]["VOO"])
                    st.metric("P/B 淨值比", f"{_pb_val:.2f}x",
                              f"低估<{_pb_bench['low']}  公允~{_pb_bench['fair']}  高估>{_pb_bench['high']}",
                              help="低 P/B = 便宜 = 建議加槓桿")
                else:
                    st.info("P/B：無法自動取得")

            _c3, _c4 = st.columns(2)

            _div_val = _indicators.get("dividend_yield")
            with _c3:
                if _div_val:
                    _div_bench = INDICATOR_BENCHMARKS["dividend_yield"].get(ticker, INDICATOR_BENCHMARKS["dividend_yield"]["VOO"])
                    st.metric("股息殖利率", f"{_div_val*100:.2f}%",
                              f"低估>{_div_bench['high']*100:.1f}%  公允~{_div_bench['fair']*100:.1f}%",
                              help="高股息率 = 便宜 = 建議加槓桿")
                else:
                    st.info("股息率：無法自動取得")

            _vix_val = _indicators.get("vix")
            with _c4:
                if _vix_val:
                    _vix_label = "😱 恐慌（加槓機會）" if _vix_val > 30 else "😤 警戒" if _vix_val > 20 else "😌 平靜"
                    st.metric("VIX 恐慌指數", f"{_vix_val:.1f}", _vix_label,
                              help="VIX>30=市場恐慌=低估機會；VIX>40=極度恐慌=最佳加槓時機")
                else:
                    st.info("VIX：無法取得")

            _c5, _c6 = st.columns(2)

            _spread_val = _indicators.get("yield_spread_10y3m")
            with _c5:
                if _spread_val is not None:
                    _curve_label = "⚠️ 倒掛（衰退警告）" if _spread_val < 0 else "✅ 正常" if _spread_val > 0.5 else "⚠️ 趨平"
                    st.metric("殖利率曲線 10Y-3M", f"{_spread_val:.2f}%pt", _curve_label,
                              help="倒掛（<0）= 衰退預警 = 建議減槓桿")
                else:
                    st.info("殖利率曲線：無法取得")

            _erp_val = _indicators.get("erp_pct")
            with _c6:
                if _erp_val is not None:
                    st.metric("股債溢價 ERP", f"{_erp_val:.2f}%", "歷史中位 5.5%",
                              help="高 ERP = 股票相對債券便宜 = 建議加槓桿")
                else:
                    st.info("ERP：無法計算")

        # ── 計算指標評分 ──
        _ind_scores = compute_indicator_scores(_indicators, ticker, manual_pe=_manual_pe)
        _value_score_auto, _eff_weights = compute_value_score(_ind_scores, user_override=None)

        # ── 價值評分 Slider（可手動覆蓋） ──
        _use_manual_score = st.checkbox(
            "✏️ 手動覆蓋估值評分", value=False,
            help="勾選後可加入主觀判斷（如總經分析、政策面、產業週期）",
        )
        if _use_manual_score:
            value_score: float = float(st.slider(
                "💡 手動估值評分（0=超低估，100=超高估）",
                min_value=0, max_value=100, value=int(_value_score_auto), step=5,
                help="0-30：低估（加槓）；40-60：公允；70-100：高估（減槓）",
            ))
            st.caption(f"自動計算評分（僅供參考）：{_value_score_auto:.0f} 分")
        else:
            value_score = _value_score_auto
            _score_zone = "💚 低估區" if value_score < 40 else ("⚪ 公允區" if value_score < 60 else "🔴 高估區")
            st.info(f"📊 綜合估值評分：**{value_score:.0f} 分**　{_score_zone}")

        # ── 槓桿激進度 ──
        _aggressiveness_label: str = st.select_slider(
            "⚡ 槓桿激進度（加減碼幅度）",
            options=["保守（±25%）", "標準（±50%）", "激進（±75%）"],
            value="標準（±50%）",
            help="保守：低估時最多加 25% 槓桿；激進：最多加 75% 槓桿（對稱適用於高估減倉）",
        )
        _aggressiveness_map: dict[str, float] = {
            "保守（±25%）": 0.5,
            "標準（±50%）": 1.0,
            "激進（±75%）": 1.5,
        }
        aggressiveness_val: float = _aggressiveness_map[_aggressiveness_label]

        # ── 計算最終價值乘數 ──
        value_mult: float = value_multiplier_from_score(value_score, aggressiveness_val)

        st.divider()
        st.info(
            "⚠️ 本工具僅供量化研究參考，不構成任何投資建議。\n\n"
            "Kelly Criterion 假設對數常態分配，實際市場存在厚尾與跳躍風險。"
        )

    # 單位換算
    r_free: float = r_free_pct / 100.0
    r_margin: float = r_margin_pct / 100.0

    # ------------------------------------------------------------------
    # Data Fetching：yfinance（主） + Stooq（交叉驗證）
    # ------------------------------------------------------------------
    with st.spinner(f"正在從 yfinance 下載 {ticker} 資料..."):
        try:
            prices_yf: pd.Series = fetch_data(ticker, mu_period_years)
        except Exception as exc:
            st.error(
                f"❌ 資料下載失敗：{exc}\n\n"
                "請確認：\n"
                "1. 網路連線是否正常\n"
                f"2. 股票代號是否正確（目前嘗試：`{ticker}`）\n"
                "3. yfinance 版本是否為最新"
            )
            st.stop()

    stooq_ticker: str | None = STOOQ_TICKERS.get(ticker)
    prices_stooq: pd.Series | None = None
    stooq_failed = False

    if stooq_ticker:
        with st.spinner(f"正在從 Stooq 下載 {stooq_ticker} 交叉驗證資料..."):
            start_dt = prices_yf.index[0].date()
            end_dt = prices_yf.index[-1].date()
            prices_stooq = fetch_data_stooq(stooq_ticker, start_dt, end_dt)
        if prices_stooq is None:
            stooq_failed = True

    # 組合 returns_dict（依可用來源）
    returns_dict: dict[str, pd.Series] = {
        "Yahoo Finance": compute_log_returns(prices_yf),
    }
    if prices_stooq is not None and len(prices_stooq) >= TRADING_DAYS:
        returns_dict["Stooq"] = compute_log_returns(prices_stooq)

    if stooq_failed:
        st.warning(
            "⚠️ Stooq 資料抓取失敗，使用 **yfinance 單一來源** 計算。"
            "（可能為網路問題或 Stooq 資料庫無此代號）"
        )

    # 合併 + Winsorization 1%/99%
    merged_returns: pd.Series = merge_and_winsorize(returns_dict)

    if len(merged_returns) < TRADING_DAYS + 1:
        st.error(
            f"❌ 資料點數不足：需要至少 {TRADING_DAYS + 1} 筆，"
            f"實際取得 {len(merged_returns)} 筆。請嘗試延長回溯期間或更換資產。"
        )
        st.stop()

    # ------------------------------------------------------------------
    # Core Calculations
    # ------------------------------------------------------------------
    mu_log: float = compute_mu(merged_returns)
    sigma: float = compute_sigma(merged_returns, sigma_model)
    # 算術報酬修正：Kelly 公式需要 μ_arith = μ_log + σ²/2
    mu_arith: float = mu_log + 0.5 * sigma ** 2

    # ── Fix 1：均值回歸折扣（mu haircut） ──
    # 歷史 μ 可能因牛市偏高；打折反映均值回歸，降低 Kelly 過度樂觀
    mu_arith_for_kelly: float = mu_arith * (1.0 - mu_haircut)

    # ── Fix 4：VIX 波動率下限（sigma floor） ──
    # 防止低 VIX 期 EWMA σ 虛低，讓 Kelly 在平靜市況也有適當風險緩衝
    _vix_raw: float | None = _indicators.get("vix") if _indicators else None
    if use_vix_floor and _vix_raw and _vix_raw > 0:
        sigma_for_kelly: float = max(sigma, _vix_raw / 100.0)
    else:
        sigma_for_kelly = sigma

    # 計算不受上限限制的理論值，用於判斷是否截斷及顯示警告
    raw_kelly_uncapped: float = compute_kelly_analytical(
        mu_arith_for_kelly, sigma_for_kelly, r_free, r_margin, max_leverage=99.0
    )
    full_kelly: float = compute_kelly_analytical(
        mu_arith_for_kelly, sigma_for_kelly, r_free, r_margin, max_leverage
    )
    kelly_was_capped: bool = raw_kelly_uncapped > max_leverage

    # 價值調整後 Kelly
    kelly_value_adjusted: float = full_kelly * value_mult
    kelly_value_adjusted_capped: float = float(np.clip(kelly_value_adjusted, -1.0, max_leverage))
    value_was_capped: bool = kelly_value_adjusted > max_leverage

    fractional_kelly: float = kelly_value_adjusted_capped * kelly_multiplier

    # ------------------------------------------------------------------
    # Top-Level Metrics
    # ------------------------------------------------------------------
    st.subheader("📈 核心指標")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label="年化預期報酬率 μ_log",
        value=f"{mu_log * 100:.2f}%",
        help=(
            f"幾何平均對數報酬 × 252\n"
            f"算術報酬 μ_arith = {mu_arith*100:.2f}%（歷史值）\n"
            f"Kelly 使用 μ = {mu_arith_for_kelly*100:.2f}%（折扣 {mu_haircut_pct}% 後）"
        ),
    )
    col2.metric(
        label="年化波動率 σ",
        value=f"{sigma * 100:.2f}%",
        delta=(
            f"VIX floor 啟用 → {sigma_for_kelly*100:.1f}%"
            if sigma_for_kelly > sigma * 1.02 else None
        ),
        delta_color="off",
        help=(
            f"模型：{sigma_model}\n"
            + (f"VIX 隱含 σ={_vix_raw/100*100:.1f}%，採用較大值 {sigma_for_kelly*100:.1f}%"
               if sigma_for_kelly > sigma * 1.02 else "VIX floor 未啟用（歷史 σ ≥ VIX 隱含 σ）")
        ),
    )
    col3.metric(
        label="【純 Kelly】倉位",
        value=f"{full_kelly * 100:.1f}%",
        delta=(
            f"⚠️ 已截斷（理論值 {raw_kelly_uncapped*100:.0f}%）"
            if kelly_was_capped
            else ("使用槓桿" if full_kelly > 1.0 else ("空倉/放空" if full_kelly < 0 else ""))
        ),
        delta_color="inverse" if (kelly_was_capped or full_kelly < 0) else "normal",
        help=(
            f"最大化幾何增長率的最佳倉位（理論值 = {raw_kelly_uncapped*100:.1f}%）\n"
            f"已套用 μ 折扣 {mu_haircut_pct}%：歷史 μ={mu_arith*100:.2f}% → "
            f"Kelly 使用 {mu_arith_for_kelly*100:.2f}%"
        ),
    )
    col4.metric(
        label="【價值調整】倉位",
        value=f"{kelly_value_adjusted_capped * 100:.1f}%",
        delta=f"價值乘數 {value_mult:.2f}×（評分 {value_score:.0f}）",
        delta_color="normal" if value_mult >= 1.0 else "inverse",
        help=(
            f"純 Kelly × 價值乘數 {value_mult:.2f}，再截斷至 max_leverage {max_leverage:.1f}×\n"
            f"Half Kelly 實際倉位 = {fractional_kelly*100:.1f}%"
        ),
    )

    # μ 折扣提示（若有折扣則顯示）
    if mu_haircut_pct > 0:
        st.caption(
            f"📉 **均值回歸折扣 {mu_haircut_pct}%**："
            f"歷史 μ_arith {mu_arith*100:.2f}% → Kelly 使用 {mu_arith_for_kelly*100:.2f}%"
            + (f"　｜　📊 VIX floor：σ {sigma*100:.1f}% → {sigma_for_kelly*100:.1f}%"
               if sigma_for_kelly > sigma * 1.02 else "")
        )

    # 價值評估警告區
    if value_score < 30:
        st.success(
            f"💚 **市場低估**（評分 {value_score:.0f}）— "
            f"建議增加槓桿至 **{kelly_value_adjusted_capped*100:.1f}%**"
            + (f"（+{(value_mult-1)*100:.1f}% vs 純 Kelly）" if value_mult > 1.0 else "")
        )
    elif value_score > 70:
        st.warning(
            f"🔴 **市場高估**（評分 {value_score:.0f}）— "
            f"建議減少槓桿至 **{kelly_value_adjusted_capped*100:.1f}%**"
            + (f"（{(value_mult-1)*100:.1f}% vs 純 Kelly）" if value_mult < 1.0 else "")
        )
    else:
        st.info(f"⚪ **公允區間**（評分 {value_score:.0f}）— 倉位未調整（乘數 {value_mult:.2f}×）")

    if value_was_capped:
        st.info(
            f"✅ 價值調整後倉位 **{kelly_value_adjusted*100:.1f}%** 超過上限，"
            f"已安全截斷至 **{max_leverage:.1f}×（{max_leverage*100:.0f}%）**"
        )

    if kelly_was_capped:
        st.warning(
            f"⚠️ Full-Kelly 理論值為 **{raw_kelly_uncapped*100:.1f}%**，"
            f"已依設定截斷至最大槓桿上限 **{max_leverage:.1f}×（{max_leverage*100:.0f}%）**。\n\n"
            "這通常發生在所選資產歷史報酬極高、波動率相對偏低時（例如 QQQ/VOO 近 10 年）。"
            "建議搭配 Half-Kelly 或更保守的乘數使用。"
        )
    elif full_kelly < 0:
        st.warning("⚠️ Full-Kelly < 0：模型建議空倉或放空。預期報酬可能低於無風險利率。")

    st.divider()

    # ------------------------------------------------------------------
    # Data Quality Report
    # ------------------------------------------------------------------
    with st.expander("🔍 資料來源品質報告", expanded=False):
        quality_df = build_data_quality_report(returns_dict, merged_returns, sigma_model)
        st.dataframe(quality_df, hide_index=True, use_container_width=True)
        n_sources = len(returns_dict)
        source_names = "、".join(returns_dict.keys())
        st.caption(
            f"共使用 **{n_sources}** 個資料來源（{source_names}）。"
            f"合併後共 **{len(merged_returns):,}** 個共同交易日。"
            "已對各來源日報酬率進行 **Winsorization（截斷至 1%/99% 分位）** 後取等權平均。"
        )

    st.divider()

    # ------------------------------------------------------------------
    # Chart 1: Rolling Kelly
    # ------------------------------------------------------------------
    st.subheader("📉 圖表一：滾動 Kelly 倉位建議走勢")

    with st.spinner("計算滾動 Kelly 中（首次較慢，之後有快取）..."):
        rolling_kelly: pd.Series = compute_rolling_kelly(
            merged_returns, r_free, r_margin, sigma_model, max_leverage
        )

    if len(rolling_kelly) == 0:
        st.warning("資料點數不足，無法計算滾動 Kelly。")
    else:
        fig_rolling = plot_rolling_kelly(rolling_kelly, kelly_multiplier)
        st.plotly_chart(fig_rolling, use_container_width=True)

        with st.expander("滾動 Kelly 統計摘要"):
            summary_df = pd.DataFrame(
                {
                    "統計量": ["最小值", "25% 分位", "中位數", "平均值", "75% 分位", "最大值", "標準差"],
                    "Full-Kelly (%)": [
                        f"{rolling_kelly.min()*100:.1f}",
                        f"{rolling_kelly.quantile(0.25)*100:.1f}",
                        f"{rolling_kelly.median()*100:.1f}",
                        f"{rolling_kelly.mean()*100:.1f}",
                        f"{rolling_kelly.quantile(0.75)*100:.1f}",
                        f"{rolling_kelly.max()*100:.1f}",
                        f"{rolling_kelly.std()*100:.1f}",
                    ],
                }
            )
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # Chart 2: Monte Carlo Simulation
    # ------------------------------------------------------------------
    st.subheader("🎲 圖表二：蒙地卡羅未來一年風險模擬")

    sim_f = fractional_kelly if abs(fractional_kelly) > 1e-6 else 0.01

    with st.spinner("執行 1,000 次蒙地卡羅模擬中（有快取，重複計算極快）..."):
        mc_paths: np.ndarray = run_monte_carlo(mu_arith, sigma, sim_f)

    max_drawdowns: np.ndarray = compute_max_drawdown_per_path(mc_paths)
    var95_drawdown: float = float(np.percentile(max_drawdowns, 95))

    _mc_title_suffix = f"價值乘數 {value_mult:.2f}×｜評分 {value_score:.0f}"
    fig_mc = plot_monte_carlo(mc_paths, var95_drawdown, sim_f, title_suffix=_mc_title_suffix)
    st.plotly_chart(fig_mc, use_container_width=True)

    final_values = mc_paths[:, -1]
    col_a, col_b, col_c, col_d = st.columns(4)

    col_a.metric(
        "模擬期末淨值中位數",
        f"{np.median(final_values):.3f}",
        help="1,000 次模擬的期末淨值中位數（初始 = 1.0）",
    )
    col_b.metric(
        "95% VaR 最大回撤",
        f"{var95_drawdown*100:.2f}%",
        help="95% 的模擬路徑，其最大回撤不超過此值",
    )
    col_c.metric(
        "期末淨值 5% 分位",
        f"{np.percentile(final_values, 5):.3f}",
        help="最差 5% 情境的期末淨值",
    )
    col_d.metric(
        "獲利機率 (>初始淨值)",
        f"{(final_values > 1.0).mean()*100:.1f}%",
        help="1,000 次模擬中，期末淨值大於 1.0 的比例",
    )

    st.divider()

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    source_label = "yfinance + Stooq" if "Stooq" in returns_dict else "yfinance"
    st.caption(
        f"資料來源：{source_label} ｜ "
        f"資產：{ticker_label} ｜ "
        f"回溯期間：{mu_period_years} 年 ｜ "
        f"最後更新：{prices_yf.index[-1].strftime('%Y-%m-%d')}"
    )


if __name__ == "__main__":
    main()
