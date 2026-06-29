import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

# ======================================================
# 0. 中文字体设置
# ======================================================

font_candidates = [
    "PingFang SC",
    "Heiti SC",
    "Songti SC",
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS"
]

available_fonts = {f.name for f in font_manager.fontManager.ttflist}

chosen_font = None
for font in font_candidates:
    if font in available_fonts:
        chosen_font = font
        break

if chosen_font is not None:
    mpl.rcParams["font.sans-serif"] = [chosen_font]
    print("当前使用中文字体：", chosen_font)
else:
    print("没有找到常见中文字体，如果图中中文乱码，需要安装中文字体。")

mpl.rcParams["axes.unicode_minus"] = False


# ======================================================
# 1. 参数设置
# ======================================================

file_path = r"UNo.11SB蜡烛图与特殊价差.xlsx"
sheet_name = "Data"

# 全样本背景区间：2004-2025，共22年，不包含2026
HIST_START_YEAR = 2004
HIST_END_YEAR = 2025
HIST_YEARS = HIST_END_YEAR - HIST_START_YEAR + 1

# 周频聚合方法：
# mean = 每周日频CV均值
# last = 每周最后一个交易日CV
WEEK_VALUE_METHOD = "mean"

# 是否删除 ISO Week 53
DROP_WEEK_53 = True

# AR预测未来几周
N_FORECAST_WEEKS = 4

# AR最大滞后阶数
AR_MAX_LAG = 8

# 牛熊年份定义
bull_periods = [
    (2004, 2005),
    (2008, 2010),
    (2016, 2016),
    (2020, 2023)
]

bear_periods = [
    (2006, 2007),
    (2011, 2015),
    (2017, 2019),
    (2024, 2026)
]

# 当前t年是否从牛熊分位数样本中剔除
# True：当前年份已经单独画红线，不再进入牛熊分位数区间
EXCLUDE_CURRENT_YEAR_FROM_REGIME_BAND = True

print(f"全样本历史区间：{HIST_START_YEAR}-{HIST_END_YEAR}，共 {HIST_YEARS} 年，不包含2026")
print("周频聚合方法：", WEEK_VALUE_METHOD)
print("AR预测未来周数：", N_FORECAST_WEEKS)
print("AR最大候选滞后阶数：", AR_MAX_LAG)


# ======================================================
# 2. 工具函数
# ======================================================

def expand_periods(periods):
    years = []
    for start, end in periods:
        years.extend(list(range(start, end + 1)))
    return years


def find_cv_col(raw, n_header_rows=10):
    header_part = raw.iloc[:n_header_rows, :].astype(str)

    icesugar_cols = []
    for col in raw.columns:
        if header_part[col].str.contains("ICESugar", case=False, na=False).any():
            icesugar_cols.append(col)

    if len(icesugar_cols) > 0:
        return icesugar_cols[0]

    cv_candidates = []
    for col in raw.columns:
        if header_part[col].str.contains("成交指数", case=False, na=False).any():
            cv_candidates.append(col)

    if len(cv_candidates) > 0:
        return cv_candidates[0]

    raise ValueError("没有找到 CV / 成交指数 / ICESugar 列，请检查Excel表头。")


def load_cv_data(file_path, sheet_name):
    raw = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl"
    )

    date_col = 0
    cv_col = find_cv_col(raw)

    df = pd.DataFrame()
    df["Date"] = pd.to_datetime(raw.iloc[:, date_col], errors="coerce")
    df["CV"] = pd.to_numeric(raw.iloc[:, cv_col], errors="coerce")

    df = df.dropna(subset=["Date", "CV"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df["calendar_year"] = df["Date"].dt.year

    iso = df["Date"].dt.isocalendar()
    df["iso_year"] = iso["year"].astype(int)
    df["iso_week"] = iso["week"].astype(int)

    return df, cv_col


def make_weekly_df(df, value_method="mean", drop_week_53=True):
    weekly = (
        df
        .groupby(["iso_year", "iso_week"], as_index=False)
        .agg(
            Date=("Date", "max"),
            CV=("CV", value_method)
        )
    )

    if drop_week_53:
        weekly = weekly[weekly["iso_week"] <= 52].copy()

    weekly = weekly.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)

    return weekly


def make_season_band(weekly_df, years):
    sample = weekly_df[weekly_df["iso_year"].isin(years)].copy()

    season = (
        sample
        .groupby("iso_week")["CV"]
        .agg(
            q20=lambda x: x.quantile(0.20),
            q80=lambda x: x.quantile(0.80),
            median="median",
            count="count"
        )
        .reset_index()
    )

    return season, sample


def fit_ar_ols(y, p):
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n <= p + 5:
        return None

    target = y[p:]

    lag_cols = []
    for i in range(1, p + 1):
        lag_cols.append(y[p - i:n - i])

    X = np.column_stack([np.ones(len(target))] + lag_cols)

    beta = np.linalg.lstsq(X, target, rcond=None)[0]
    fitted = X @ beta
    resid = target - fitted

    rss = np.sum(resid ** 2)
    obs = len(target)
    k = p + 1

    sigma2 = rss / obs
    sigma2 = max(sigma2, 1e-12)

    aic = obs * np.log(sigma2) + 2 * k

    return beta, aic


def ar_recursive_forecast(y, n_forecast=4, max_lag=8):
    y = pd.Series(y).dropna().astype(float).values

    if len(y) < 20:
        raise ValueError("周频数据太少，AR模型不稳定，至少建议20个周频点以上。")

    max_lag = min(max_lag, len(y) // 4)
    max_lag = max(1, max_lag)

    best_p = None
    best_beta = None
    best_aic = np.inf

    for p in range(1, max_lag + 1):
        result = fit_ar_ols(y, p)

        if result is None:
            continue

        beta, aic = result

        if aic < best_aic:
            best_aic = aic
            best_p = p
            best_beta = beta

    if best_beta is None:
        raise ValueError("AR模型拟合失败，请检查数据。")

    history = list(y)
    forecasts = []

    for step in range(n_forecast):
        lags = [history[-i] for i in range(1, best_p + 1)]
        pred = best_beta[0] + np.dot(best_beta[1:], lags)

        forecasts.append(pred)
        history.append(pred)

    info = {
        "best_p": best_p,
        "best_aic": best_aic,
        "intercept": best_beta[0],
        "params": best_beta[1:]
    }

    return np.array(forecasts), info


def plot_weekly_season_chart(
    season_data,
    df_t1,
    df_t,
    forecast_x,
    forecast_y,
    title,
    band_label,
    band_color,
    median_label,
    current_year,
    prev_year,
    ar_p,
    show_forecast=True,
    figsize=(14, 7)
):
    plt.figure(figsize=figsize)

    # 20%-80% 分位区间
    plt.fill_between(
        season_data["iso_week"],
        season_data["q20"],
        season_data["q80"],
        color=band_color,
        alpha=0.18,
        label=band_label,
        zorder=1
    )

    # 中位数
    plt.plot(
        season_data["iso_week"],
        season_data["median"],
        color="gray",
        linestyle="--",
        linewidth=2.2,
        alpha=0.95,
        label=median_label,
        zorder=3
    )

    # t-1 年
    plt.plot(
        df_t1["iso_week"],
        df_t1["CV"],
        color="black",
        linewidth=2.5,
        label=f"{prev_year}年",
        zorder=5
    )

    # t 年
    plt.plot(
        df_t["iso_week"],
        df_t["CV"],
        color="red",
        linewidth=2.8,
        label=f"{current_year}年",
        zorder=6
    )

    # 预测线：只有全样本图打开
    if show_forecast:
        plt.plot(
            forecast_x,
            forecast_y,
            color="orange",
            linestyle="--",
            linewidth=3.0,
            alpha=0.95,
            label=f"AR({ar_p}) 未来{N_FORECAST_WEEKS}周预测",
            zorder=9
        )

        max_x = max(52, int(np.nanmax(forecast_x)))
    else:
        max_x = 52

    plt.title(title, fontsize=16)
    plt.xlabel("ISO周")
    plt.ylabel("原糖指数")

    plt.xticks(np.arange(1, max_x + 1, 4))
    plt.xlim(1, max_x)

    plt.grid(alpha=0.15)
    plt.legend(loc="upper left", ncol=2)

    plt.tight_layout()
    plt.show()


# ======================================================
# 3. 读取并处理数据
# ======================================================

df, cv_col = load_cv_data(file_path, sheet_name)

weekly_df = make_weekly_df(
    df,
    value_method=WEEK_VALUE_METHOD,
    drop_week_53=DROP_WEEK_53
)

print("数据区间：", df["Date"].min(), "到", df["Date"].max())
print("CV列位置：", cv_col)
print("最新日频数据：")
print(df[["Date", "CV", "calendar_year", "iso_year", "iso_week"]].tail(10))

print("最新周频数据：")
print(weekly_df.tail(10))


# ======================================================
# 4. 识别 t 年和 t-1 年
# ======================================================

current_year = int(weekly_df.dropna(subset=["CV"])["iso_year"].max())
prev_year = current_year - 1

df_t_w_valid = (
    weekly_df[weekly_df["iso_year"] == current_year]
    .dropna(subset=["CV"])
    .sort_values("Date")
)

df_t1_w_valid = (
    weekly_df[weekly_df["iso_year"] == prev_year]
    .dropna(subset=["CV"])
    .sort_values("Date")
)

print("当前t年：", current_year)
print("t-1年：", prev_year)
print(f"{current_year}周频点数：", len(df_t_w_valid))
print(f"{prev_year}周频点数：", len(df_t1_w_valid))


# ======================================================
# 5. 生成三套 season band
# ======================================================

# 图1：全样本 2004-2025
hist_years = list(range(HIST_START_YEAR, HIST_END_YEAR + 1))
all_season, all_sample = make_season_band(weekly_df, hist_years)

# 图2：牛市条件
bull_years = expand_periods(bull_periods)

if EXCLUDE_CURRENT_YEAR_FROM_REGIME_BAND:
    bull_band_years = [y for y in bull_years if y != current_year]
else:
    bull_band_years = bull_years

bull_season, bull_sample = make_season_band(weekly_df, bull_band_years)

# 图3：熊市条件
bear_years = expand_periods(bear_periods)

if EXCLUDE_CURRENT_YEAR_FROM_REGIME_BAND:
    bear_band_years = [y for y in bear_years if y != current_year]
else:
    bear_band_years = bear_years

bear_season, bear_sample = make_season_band(weekly_df, bear_band_years)

print("全样本分位数年份：", hist_years)
print("牛市分位数年份：", bull_band_years)
print("熊市分位数年份：", bear_band_years)
print("全样本周数：", len(all_sample))
print("牛市样本周数：", len(bull_sample))
print("熊市样本周数：", len(bear_sample))


# ======================================================
# 6. AR预测
# ======================================================

train_weekly = weekly_df.dropna(subset=["CV"]).sort_values("Date").copy()

ar_fc, ar_info = ar_recursive_forecast(
    train_weekly["CV"],
    n_forecast=N_FORECAST_WEEKS,
    max_lag=AR_MAX_LAG
)

if len(df_t_w_valid) == 0:
    raise ValueError(f"没有 {current_year} 年周频数据，无法接预测线。")

last_week = int(df_t_w_valid["iso_week"].iloc[-1])
last_cv = float(df_t_w_valid["CV"].iloc[-1])
last_date = pd.to_datetime(df_t_w_valid["Date"].iloc[-1])

forecast_dates = [
    last_date + pd.Timedelta(days=7 * i)
    for i in range(1, N_FORECAST_WEEKS + 1)
]

forecast_table = pd.DataFrame({
    "预测日期": forecast_dates,
    "AR预测原糖指数": ar_fc
})

forecast_iso = forecast_table["预测日期"].dt.isocalendar()
forecast_table["预测ISO年份"] = forecast_iso["year"].astype(int)
forecast_table["预测ISO周"] = forecast_iso["week"].astype(int)

forecast_x = np.arange(last_week, last_week + N_FORECAST_WEEKS + 1)
forecast_y = np.r_[last_cv, ar_fc]

print("AR模型信息：")
print("best_p =", ar_info["best_p"])
print("best_aic =", ar_info["best_aic"])
print("intercept =", ar_info["intercept"])
print("params =", ar_info["params"])

print("AR未来4周预测：")
print(forecast_table)


# ======================================================
# 7. 输出三张图
# ======================================================

# 图1：全样本 2004-2025，保留 AR 预测
plot_weekly_season_chart(
    season_data=all_season,
    df_t1=df_t1_w_valid,
    df_t=df_t_w_valid,
    forecast_x=forecast_x,
    forecast_y=forecast_y,
    title=f"ICE原糖指数周度季节性结构（全样本：{HIST_START_YEAR}-{HIST_END_YEAR}）",
    band_label=f"{HIST_YEARS}年历史区间（{HIST_START_YEAR}-{HIST_END_YEAR}，20%-80%分位）",
    band_color="royalblue",
    median_label=f"{HIST_YEARS}年历史中位数",
    current_year=current_year,
    prev_year=prev_year,
    ar_p=ar_info["best_p"],
    show_forecast=True
)

# 图2：牛市条件，不画 AR 预测
plot_weekly_season_chart(
    season_data=bull_season,
    df_t1=df_t1_w_valid,
    df_t=df_t_w_valid,
    forecast_x=forecast_x,
    forecast_y=forecast_y,
    title="ICE原糖指数周度季节性结构（牛市条件）",
    band_label="牛市年份区间（20%-80%分位）",
    band_color="lightcoral",
    median_label="牛市年份中位数",
    current_year=current_year,
    prev_year=prev_year,
    ar_p=ar_info["best_p"],
    show_forecast=False
)

# 图3：熊市条件，不画 AR 预测
plot_weekly_season_chart(
    season_data=bear_season,
    df_t1=df_t1_w_valid,
    df_t=df_t_w_valid,
    forecast_x=forecast_x,
    forecast_y=forecast_y,
    title="ICE原糖指数周度季节性结构（熊市条件）",
    band_label="熊市年份区间（20%-80%分位）",
    band_color="mediumseagreen",
    median_label="熊市年份中位数",
    current_year=current_year,
    prev_year=prev_year,
    ar_p=ar_info["best_p"],
    show_forecast=False
)
