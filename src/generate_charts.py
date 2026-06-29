from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "UNo.11SB蜡烛图与特殊价差.xlsx"
ASSET_DIR = ROOT / "assets"

SHEET_NAME = "Data"
HIST_START_YEAR = 2004
HIST_END_YEAR = 2025
HIST_YEARS = HIST_END_YEAR - HIST_START_YEAR + 1
WEEK_VALUE_METHOD = "mean"
DROP_WEEK_53 = True
N_FORECAST_WEEKS = 4
AR_MAX_LAG = 8
EXCLUDE_CURRENT_YEAR_FROM_REGIME_BAND = True

BULL_PERIODS = [
    (2004, 2005),
    (2008, 2010),
    (2016, 2016),
    (2020, 2023),
]

BEAR_PERIODS = [
    (2006, 2007),
    (2011, 2015),
    (2017, 2019),
    (2024, 2026),
]

FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]


def expand_periods(periods: list[tuple[int, int]]) -> list[int]:
    years: list[int] = []
    for start, end in periods:
        years.extend(range(start, end + 1))
    return years


def find_cv_col(raw: pd.DataFrame, n_header_rows: int = 10) -> int:
    header_part = raw.iloc[:n_header_rows, :].astype(str)

    for col in raw.columns:
        if header_part[col].str.contains("ICESugar", case=False, na=False).any():
            return int(col)

    for col in raw.columns:
        if header_part[col].str.contains("成交指数", case=False, na=False).any():
            return int(col)

    raise ValueError("没有找到 CV / 成交指数 / ICESugar 列，请检查 Excel 表头。")


def load_cv_data(file_path: Path, sheet_name: str) -> tuple[pd.DataFrame, int]:
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None, engine="openpyxl")
    cv_col = find_cv_col(raw)

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw.iloc[:, 0], errors="coerce"),
            "CV": pd.to_numeric(raw.iloc[:, cv_col], errors="coerce"),
        }
    )
    df = df.dropna(subset=["Date", "CV"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df["calendar_year"] = df["Date"].dt.year

    iso = df["Date"].dt.isocalendar()
    df["iso_year"] = iso["year"].astype(int)
    df["iso_week"] = iso["week"].astype(int)
    return df, cv_col


def make_weekly_df(
    df: pd.DataFrame,
    value_method: str = WEEK_VALUE_METHOD,
    drop_week_53: bool = DROP_WEEK_53,
) -> pd.DataFrame:
    weekly = (
        df.groupby(["iso_year", "iso_week"], as_index=False)
        .agg(Date=("Date", "max"), CV=("CV", value_method))
        .sort_values(["iso_year", "iso_week"])
        .reset_index(drop=True)
    )

    if drop_week_53:
        weekly = weekly[weekly["iso_week"] <= 52].copy()

    return weekly.reset_index(drop=True)


def make_season_band(
    weekly_df: pd.DataFrame,
    years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = weekly_df[weekly_df["iso_year"].isin(years)].copy()
    season = (
        sample.groupby("iso_week")["CV"]
        .agg(
            q20=lambda x: x.quantile(0.20),
            q80=lambda x: x.quantile(0.80),
            median="median",
            count="count",
        )
        .reset_index()
    )
    return season, sample


def fit_ar_ols(y: np.ndarray, p: int) -> tuple[np.ndarray, float] | None:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n <= p + 5:
        return None

    target = y[p:]
    lag_cols = [y[p - i : n - i] for i in range(1, p + 1)]
    x = np.column_stack([np.ones(len(target))] + lag_cols)

    beta = np.linalg.lstsq(x, target, rcond=None)[0]
    fitted = x @ beta
    resid = target - fitted

    rss = float(np.sum(resid**2))
    obs = len(target)
    sigma2 = max(rss / obs, 1e-12)
    aic = obs * math.log(sigma2) + 2 * (p + 1)
    return beta, aic


def ar_recursive_forecast(
    y: pd.Series,
    n_forecast: int = N_FORECAST_WEEKS,
    max_lag: int = AR_MAX_LAG,
) -> tuple[np.ndarray, dict[str, object]]:
    values = pd.Series(y).dropna().astype(float).values
    if len(values) < 20:
        raise ValueError("周频数据太少，AR 模型不稳定，至少建议 20 个周频点以上。")

    max_lag = min(max_lag, len(values) // 4)
    max_lag = max(1, max_lag)

    best_p: int | None = None
    best_beta: np.ndarray | None = None
    best_aic = np.inf

    for p in range(1, max_lag + 1):
        result = fit_ar_ols(values, p)
        if result is None:
            continue
        beta, aic = result
        if aic < best_aic:
            best_p = p
            best_beta = beta
            best_aic = aic

    if best_beta is None or best_p is None:
        raise ValueError("AR 模型拟合失败，请检查数据。")

    history = list(values)
    forecasts: list[float] = []
    for _ in range(n_forecast):
        lags = [history[-i] for i in range(1, best_p + 1)]
        pred = float(best_beta[0] + np.dot(best_beta[1:], lags))
        forecasts.append(pred)
        history.append(pred)

    return np.array(forecasts), {
        "best_p": best_p,
        "best_aic": float(best_aic),
        "intercept": float(best_beta[0]),
        "params": [float(x) for x in best_beta[1:]],
    }


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES:
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    bbox = draw.textbbox((0, 0), text, font=font)
    return float(bbox[2] - bbox[0])


def nice_ticks(ymin: float, ymax: float, count: int = 6) -> list[float]:
    if ymin == ymax:
        return [ymin]

    raw_step = (ymax - ymin) / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    residual = raw_step / magnitude
    if residual <= 1:
        nice_step = magnitude
    elif residual <= 2:
        nice_step = 2 * magnitude
    elif residual <= 5:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude

    start = math.floor(ymin / nice_step) * nice_step
    end = math.ceil(ymax / nice_step) * nice_step
    ticks = []
    value = start
    while value <= end + nice_step * 0.5:
        ticks.append(round(value, 10))
        value += nice_step
    return ticks


def format_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def rgba(color: tuple[int, int, int], alpha: int = 255) -> tuple[int, int, int, int]:
    return color[0], color[1], color[2], alpha


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    fill: tuple[int, int, int, int] | tuple[int, int, int],
    width: int,
    dash: float = 16,
    gap: float = 10,
) -> None:
    for start, end in zip(points, points[1:]):
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        distance = math.hypot(dx, dy)
        if distance == 0:
            continue
        ux = dx / distance
        uy = dy / distance
        position = 0.0
        while position < distance:
            segment_end = min(position + dash, distance)
            draw.line(
                [
                    (x1 + ux * position, y1 + uy * position),
                    (x1 + ux * segment_end, y1 + uy * segment_end),
                ],
                fill=fill,
                width=width,
            )
            position += dash + gap


def make_points(
    frame: tuple[int, int, int, int],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    weeks: pd.Series,
    values: pd.Series,
) -> list[tuple[float, float]]:
    left, top, right, bottom = frame

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    points: list[tuple[float, float]] = []
    for week, value in zip(weeks, values):
        if pd.notna(value):
            points.append((sx(float(week)), sy(float(value))))
    return points


def draw_chart(
    season_data: pd.DataFrame,
    df_t1: pd.DataFrame,
    df_t: pd.DataFrame,
    forecast_x: np.ndarray,
    forecast_y: np.ndarray,
    title: str,
    band_label: str,
    band_color: tuple[int, int, int],
    median_label: str,
    current_year: int,
    prev_year: int,
    ar_p: int,
    output_path: Path,
    show_forecast: bool,
) -> None:
    width, height = 1600, 900
    left, top, right, bottom = 115, 120, 1545, 760
    frame = left, top, right, bottom
    x_min = 1
    x_max = max(52, int(np.nanmax(forecast_x))) if show_forecast else 52

    y_values = [
        season_data["q20"],
        season_data["q80"],
        season_data["median"],
        df_t1["CV"],
        df_t["CV"],
    ]
    if show_forecast:
        y_values.append(pd.Series(forecast_y))
    combined = pd.concat(y_values).dropna().astype(float)
    y_min = float(combined.min())
    y_max = float(combined.max())
    padding = (y_max - y_min) * 0.08 if y_max > y_min else 1
    y_min -= padding
    y_max += padding
    y_ticks = nice_ticks(y_min, y_max)
    y_min = min(y_ticks)
    y_max = max(y_ticks)

    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    title_font = load_font(34)
    label_font = load_font(23)
    small_font = load_font(20)
    tick_font = load_font(18)

    draw.text((width / 2, 38), title, fill=(28, 31, 36), font=title_font, anchor="ma")
    draw.text((width / 2, 817), "ISO周", fill=(52, 58, 64), font=label_font, anchor="ma")
    draw.text((32, top + (bottom - top) / 2), "CV指数", fill=(52, 58, 64), font=label_font, anchor="lm")

    plot_bg = (248, 250, 252)
    draw.rectangle([left, top, right, bottom], fill=plot_bg, outline=(198, 205, 214), width=2)

    for tick in y_ticks:
        y = make_points(frame, x_min, x_max, y_min, y_max, pd.Series([1]), pd.Series([tick]))[0][1]
        draw.line([(left, y), (right, y)], fill=(226, 232, 240), width=1)
        draw.text((left - 14, y), format_tick(tick), fill=(82, 91, 101), font=tick_font, anchor="rm")

    for week in range(1, x_max + 1, 4):
        x = make_points(frame, x_min, x_max, y_min, y_max, pd.Series([week]), pd.Series([y_min]))[0][0]
        draw.line([(x, top), (x, bottom)], fill=(235, 239, 245), width=1)
        draw.text((x, bottom + 18), str(week), fill=(82, 91, 101), font=tick_font, anchor="ma")

    band = season_data.dropna(subset=["q20", "q80"]).sort_values("iso_week")
    upper = make_points(frame, x_min, x_max, y_min, y_max, band["iso_week"], band["q80"])
    lower = make_points(frame, x_min, x_max, y_min, y_max, band["iso_week"], band["q20"])
    if upper and lower:
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.polygon(upper + list(reversed(lower)), fill=rgba(band_color, 46))
        image.alpha_composite(overlay)
        draw = ImageDraw.Draw(image)

    median_points = make_points(
        frame,
        x_min,
        x_max,
        y_min,
        y_max,
        season_data["iso_week"],
        season_data["median"],
    )
    draw_dashed_line(draw, median_points, fill=(96, 103, 112), width=4, dash=18, gap=10)

    prev_points = make_points(frame, x_min, x_max, y_min, y_max, df_t1["iso_week"], df_t1["CV"])
    current_points = make_points(frame, x_min, x_max, y_min, y_max, df_t["iso_week"], df_t["CV"])
    if len(prev_points) > 1:
        draw.line(prev_points, fill=(24, 28, 33), width=5, joint="curve")
    if len(current_points) > 1:
        draw.line(current_points, fill=(214, 32, 39), width=6, joint="curve")

    if show_forecast:
        forecast_points = make_points(
            frame,
            x_min,
            x_max,
            y_min,
            y_max,
            pd.Series(forecast_x),
            pd.Series(forecast_y),
        )
        draw_dashed_line(draw, forecast_points, fill=(238, 132, 30), width=6, dash=19, gap=11)

    legend_items: list[tuple[str, tuple[int, int, int], str]] = [
        (band_label, band_color, "band"),
        (median_label, (96, 103, 112), "dash"),
        (f"{prev_year}年", (24, 28, 33), "line"),
        (f"{current_year}年", (214, 32, 39), "line"),
    ]
    if show_forecast:
        legend_items.append((f"AR({ar_p}) 未来{N_FORECAST_WEEKS}周预测", (238, 132, 30), "dash"))

    legend_x, legend_y = left + 18, top + 18
    row_height = 36
    column_width = 650
    rows = math.ceil(len(legend_items) / 2)
    legend_w = column_width * 2 - 24
    legend_h = rows * row_height + 22
    draw.rounded_rectangle(
        [legend_x - 12, legend_y - 10, legend_x - 12 + legend_w, legend_y - 10 + legend_h],
        radius=6,
        fill=(255, 255, 255, 232),
        outline=(222, 229, 236),
        width=1,
    )
    for i, (label, color, style) in enumerate(legend_items):
        col = i % 2
        row = i // 2
        x = legend_x + col * column_width
        y = legend_y + row * row_height
        if style == "band":
            draw.rectangle([x, y + 7, x + 34, y + 23], fill=rgba(color, 90), outline=color)
        elif style == "dash":
            draw_dashed_line(draw, [(x, y + 15), (x + 38, y + 15)], fill=color, width=4, dash=9, gap=5)
        else:
            draw.line([(x, y + 15), (x + 38, y + 15)], fill=color, width=5)
        draw.text((x + 48, y + 16), label, fill=(38, 43, 49), font=small_font, anchor="lm")

    latest = df_t.sort_values("Date").iloc[-1]
    note = (
        f"最新周频数据：{pd.to_datetime(latest['Date']).date()}，"
        f"ISO第{int(latest['iso_week'])}周，CV={float(latest['CV']):.2f}"
    )
    if show_forecast:
        note += f"；AR({ar_p})预测线从第{int(forecast_x[0])}周延伸至第{int(forecast_x[-1])}周"
    draw.text((left, 858), note, fill=(80, 89, 100), font=small_font, anchor="la")

    image.convert("RGB").save(output_path, quality=95, optimize=True)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    df, cv_col = load_cv_data(DATA_PATH, SHEET_NAME)
    weekly_df = make_weekly_df(df)

    current_year = int(weekly_df.dropna(subset=["CV"])["iso_year"].max())
    prev_year = current_year - 1

    df_t = (
        weekly_df[weekly_df["iso_year"] == current_year]
        .dropna(subset=["CV"])
        .sort_values("Date")
    )
    df_t1 = (
        weekly_df[weekly_df["iso_year"] == prev_year]
        .dropna(subset=["CV"])
        .sort_values("Date")
    )

    hist_years = list(range(HIST_START_YEAR, HIST_END_YEAR + 1))
    all_season, all_sample = make_season_band(weekly_df, hist_years)

    bull_years = expand_periods(BULL_PERIODS)
    bull_band_years = [y for y in bull_years if y != current_year] if EXCLUDE_CURRENT_YEAR_FROM_REGIME_BAND else bull_years
    bull_season, bull_sample = make_season_band(weekly_df, bull_band_years)

    bear_years = expand_periods(BEAR_PERIODS)
    bear_band_years = [y for y in bear_years if y != current_year] if EXCLUDE_CURRENT_YEAR_FROM_REGIME_BAND else bear_years
    bear_season, bear_sample = make_season_band(weekly_df, bear_band_years)

    train_weekly = weekly_df.dropna(subset=["CV"]).sort_values("Date").copy()
    ar_fc, ar_info = ar_recursive_forecast(train_weekly["CV"])

    if df_t.empty:
        raise ValueError(f"没有 {current_year} 年周频数据，无法接预测线。")

    last_week = int(df_t["iso_week"].iloc[-1])
    last_cv = float(df_t["CV"].iloc[-1])
    last_date = pd.to_datetime(df_t["Date"].iloc[-1])
    forecast_dates = [last_date + pd.Timedelta(days=7 * i) for i in range(1, N_FORECAST_WEEKS + 1)]
    forecast_table = pd.DataFrame({"预测日期": forecast_dates, "AR预测CV": ar_fc})
    forecast_iso = forecast_table["预测日期"].dt.isocalendar()
    forecast_table["预测ISO年份"] = forecast_iso["year"].astype(int)
    forecast_table["预测ISO周"] = forecast_iso["week"].astype(int)

    forecast_x = np.arange(last_week, last_week + N_FORECAST_WEEKS + 1)
    forecast_y = np.r_[last_cv, ar_fc]
    ar_p = int(ar_info["best_p"])

    charts = [
        (
            all_season,
            f"ICE原糖CV指数周度季节性结构（全样本：{HIST_START_YEAR}-{HIST_END_YEAR}）",
            f"{HIST_YEARS}年历史区间（{HIST_START_YEAR}-{HIST_END_YEAR}，20%-80%分位）",
            (65, 105, 225),
            f"{HIST_YEARS}年历史中位数",
            "ice_sugar_cv_all_sample.png",
            True,
        ),
        (
            bull_season,
            "ICE原糖CV指数周度季节性结构（牛市条件）",
            "牛市年份区间（20%-80%分位）",
            (235, 111, 111),
            "牛市年份中位数",
            "ice_sugar_cv_bull_market.png",
            False,
        ),
        (
            bear_season,
            "ICE原糖CV指数周度季节性结构（熊市条件）",
            "熊市年份区间（20%-80%分位）",
            (60, 179, 113),
            "熊市年份中位数",
            "ice_sugar_cv_bear_market.png",
            False,
        ),
    ]

    for season, title, band_label, band_color, median_label, filename, show_forecast in charts:
        draw_chart(
            season_data=season,
            df_t1=df_t1,
            df_t=df_t,
            forecast_x=forecast_x,
            forecast_y=forecast_y,
            title=title,
            band_label=band_label,
            band_color=band_color,
            median_label=median_label,
            current_year=current_year,
            prev_year=prev_year,
            ar_p=ar_p,
            output_path=ASSET_DIR / filename,
            show_forecast=show_forecast,
        )

    forecast_csv = ASSET_DIR / "forecast.csv"
    forecast_table.to_csv(forecast_csv, index=False, encoding="utf-8-sig")

    metadata = {
        "data_file": DATA_PATH.name,
        "sheet_name": SHEET_NAME,
        "cv_column_index": cv_col,
        "data_start": str(df["Date"].min().date()),
        "data_end": str(df["Date"].max().date()),
        "weekly_start": str(weekly_df["Date"].min().date()),
        "weekly_end": str(weekly_df["Date"].max().date()),
        "current_year": current_year,
        "previous_year": prev_year,
        "latest_week": int(df_t["iso_week"].iloc[-1]),
        "latest_cv": float(df_t["CV"].iloc[-1]),
        "history_years": hist_years,
        "bull_band_years": bull_band_years,
        "bear_band_years": bear_band_years,
        "all_sample_weeks": int(len(all_sample)),
        "bull_sample_weeks": int(len(bull_sample)),
        "bear_sample_weeks": int(len(bear_sample)),
        "ar_model": ar_info,
        "forecast": [
            {
                "date": str(row["预测日期"].date()),
                "iso_year": int(row["预测ISO年份"]),
                "iso_week": int(row["预测ISO周"]),
                "cv": float(row["AR预测CV"]),
            }
            for _, row in forecast_table.iterrows()
        ],
        "charts": [item[5] for item in charts],
    }
    (ASSET_DIR / "chart_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Generated:")
    for filename in metadata["charts"]:
        print(f"  assets/{filename}")
    print("  assets/forecast.csv")
    print("  assets/chart_metadata.json")


if __name__ == "__main__":
    main()
