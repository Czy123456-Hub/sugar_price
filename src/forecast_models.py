from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "ar": "AR自回归",
    "ridge": "Ridge线性回归",
    "knn": "KNN非线性回归",
    "holt": "Holt趋势",
}

MODEL_KEYS = list(MODEL_LABELS.keys())
DISPLAY_LABELS = MODEL_LABELS.copy()
FACTOR_DESCRIPTIONS = [
    {
        "name": "近8周滞后价格",
        "detail": "使用当前周至前7周的原糖指数水平值，捕捉短期惯性和均值回归。",
    },
    {
        "name": "1/4/8周动量",
        "detail": "当前原糖指数分别减去1周、4周、8周前水平，衡量短线、中线价格变化方向。",
    },
    {
        "name": "4/8/13周滚动均值",
        "detail": "用一个月、两个月、一个季度左右的平均水平描述近期中枢。",
    },
    {
        "name": "4/8/13周滚动波动率",
        "detail": "使用对应窗口内的标准差，衡量近期价格波动状态。",
    },
    {
        "name": "相对滚动均值偏离",
        "detail": "当前价格减去4/8/13周滚动均值，描述价格相对近期中枢的高低。",
    },
    {
        "name": "当前ISO周季节项",
        "detail": "用sin/cos编码当前ISO周，避免把第52周和第1周误判为距离很远。",
    },
    {
        "name": "目标ISO周季节项",
        "detail": "用sin/cos编码未来预测周，给直接预测模型加入未来周度季节位置。",
    },
]


def _clean_weekly(weekly_df: pd.DataFrame) -> pd.DataFrame:
    return (
        weekly_df.dropna(subset=["Date", "CV", "iso_year", "iso_week"])
        .sort_values("Date")
        .reset_index(drop=True)
        .copy()
    )


def _iso_weeks(dates: Iterable[pd.Timestamp]) -> np.ndarray:
    return np.array([int(pd.Timestamp(date).isocalendar().week) for date in dates], dtype=int)


def _safe_array(values: Iterable[float]) -> np.ndarray:
    values_arr = np.asarray(list(values), dtype=float)
    return values_arr[np.isfinite(values_arr)]


def _fallback_forecast(y: np.ndarray, horizon: int) -> np.ndarray:
    if len(y) == 0:
        return np.full(horizon, np.nan)
    return np.full(horizon, float(y[-1]))


def forecast_naive(train: pd.DataFrame, horizon: int) -> np.ndarray:
    values = _safe_array(train["CV"])
    return _fallback_forecast(values, horizon)


def forecast_drift(train: pd.DataFrame, horizon: int, window: int = 8) -> np.ndarray:
    values = _safe_array(train["CV"])
    if len(values) < 3:
        return _fallback_forecast(values, horizon)

    tail = values[-min(len(values), window + 1) :]
    diffs = np.diff(tail)
    drift = float(np.nanmedian(diffs)) if len(diffs) else 0.0
    return values[-1] + drift * np.arange(1, horizon + 1, dtype=float)


def _seasonal_profile(train: pd.DataFrame, years: list[int]) -> pd.Series:
    sample = train[train["iso_year"].isin(years)].copy()
    if sample.empty:
        sample = train.copy()

    profile = sample.groupby("iso_week")["CV"].median().sort_index()
    fallback = train.groupby("iso_week")["CV"].median().sort_index()
    return fallback.combine_first(profile) if profile.empty else profile.combine_first(fallback)


def forecast_seasonal_adjusted(
    train: pd.DataFrame,
    future_weeks: np.ndarray,
    years: list[int],
    offset_decay: float = 0.88,
) -> np.ndarray:
    values = _safe_array(train["CV"])
    if len(values) == 0:
        return np.full(len(future_weeks), np.nan)

    profile = _seasonal_profile(train, years)
    latest = train.iloc[-1]
    latest_week = int(latest["iso_week"])
    latest_value = float(latest["CV"])
    base_latest = float(profile.get(latest_week, profile.median()))
    offset = latest_value - base_latest

    preds: list[float] = []
    for h, week in enumerate(future_weeks, start=1):
        base = float(profile.get(int(week), profile.median()))
        preds.append(base + offset * (offset_decay**h))
    return np.array(preds, dtype=float)


def _fit_ar_ols(y: np.ndarray, p: int) -> tuple[np.ndarray, float] | None:
    n = len(y)
    if n <= p + 5:
        return None

    target = y[p:]
    lag_cols = [y[p - i : n - i] for i in range(1, p + 1)]
    x = np.column_stack([np.ones(len(target))] + lag_cols)
    beta = np.linalg.lstsq(x, target, rcond=None)[0]
    resid = target - x @ beta
    rss = float(np.sum(resid**2))
    obs = len(target)
    sigma2 = max(rss / max(obs, 1), 1e-12)
    aic = obs * math.log(sigma2) + 2 * (p + 1)
    return beta, aic


def forecast_ar(
    train: pd.DataFrame,
    horizon: int,
    max_lag: int = 8,
) -> tuple[np.ndarray, dict[str, float | int | list[float]]]:
    values = _safe_array(train["CV"])
    if len(values) < 20:
        return _fallback_forecast(values, horizon), {"best_p": 0, "best_aic": np.nan, "params": []}

    max_lag = max(1, min(max_lag, len(values) // 4))
    best_p: int | None = None
    best_beta: np.ndarray | None = None
    best_aic = np.inf

    for p in range(1, max_lag + 1):
        result = _fit_ar_ols(values, p)
        if result is None:
            continue
        beta, aic = result
        if aic < best_aic:
            best_p = p
            best_beta = beta
            best_aic = aic

    if best_beta is None or best_p is None:
        return _fallback_forecast(values, horizon), {"best_p": 0, "best_aic": np.nan, "params": []}

    history = list(values)
    preds: list[float] = []
    for _ in range(horizon):
        lags = [history[-i] for i in range(1, best_p + 1)]
        pred = float(best_beta[0] + np.dot(best_beta[1:], lags))
        preds.append(pred)
        history.append(pred)

    return np.array(preds, dtype=float), {
        "best_p": int(best_p),
        "best_aic": float(best_aic),
        "intercept": float(best_beta[0]),
        "params": [float(x) for x in best_beta[1:]],
    }


def forecast_holt_damped(train: pd.DataFrame, horizon: int, max_points: int = 260) -> np.ndarray:
    values = _safe_array(train["CV"])
    if len(values) < 8:
        return _fallback_forecast(values, horizon)

    y = values[-min(len(values), max_points) :]
    alphas = (0.2, 0.35, 0.5, 0.65, 0.8)
    betas = (0.03, 0.08, 0.15, 0.25)
    phis = (0.85, 0.93, 0.98, 1.0)
    best_state: tuple[float, float, float] | None = None
    best_mse = np.inf

    for alpha in alphas:
        for beta in betas:
            for phi in phis:
                level = float(y[0])
                trend = float(y[1] - y[0])
                errors: list[float] = []
                for value in y[1:]:
                    pred = level + phi * trend
                    errors.append(float(value - pred))
                    previous_level = level
                    level = alpha * float(value) + (1 - alpha) * (level + phi * trend)
                    trend = beta * (level - previous_level) + (1 - beta) * phi * trend
                mse = float(np.mean(np.square(errors))) if errors else np.inf
                if mse < best_mse:
                    best_mse = mse
                    best_state = (level, trend, phi)

    if best_state is None:
        return _fallback_forecast(values, horizon)

    level, trend, phi = best_state
    preds = []
    cumulative = 0.0
    for h in range(1, horizon + 1):
        cumulative += phi**h
        preds.append(level + cumulative * trend)
    return np.array(preds, dtype=float)


def _cyclical_week_features(week: int) -> list[float]:
    angle = 2 * math.pi * float(week) / 52.0
    return [math.sin(angle), math.cos(angle)]


def _ridge_features(values: np.ndarray, weeks: np.ndarray, anchor: int, target_week: int, max_lag: int) -> list[float]:
    feats: list[float] = []
    for lag in range(max_lag):
        feats.append(float(values[anchor - lag]))

    for step in (1, 4, 8):
        if anchor - step >= 0:
            feats.append(float(values[anchor] - values[anchor - step]))
        else:
            feats.append(0.0)

    for window in (4, 8, 13):
        block = values[max(0, anchor - window + 1) : anchor + 1]
        feats.append(float(np.mean(block)))
        feats.append(float(np.std(block)))
        feats.append(float(values[anchor] - np.mean(block)))

    feats.extend(_cyclical_week_features(int(weeks[anchor])))
    feats.extend(_cyclical_week_features(int(target_week)))
    return feats


def _ridge_predict_for_horizon(
    values: np.ndarray,
    weeks: np.ndarray,
    future_week: int,
    horizon: int,
    max_lag: int = 8,
    alpha: float = 25.0,
    max_points: int = 520,
) -> float:
    if len(values) > max_points:
        values = values[-max_points:]
        weeks = weeks[-max_points:]

    if len(values) < max_lag + horizon + 20:
        return float(values[-1])

    rows: list[list[float]] = []
    targets: list[float] = []
    for anchor in range(max_lag - 1, len(values) - horizon):
        target_idx = anchor + horizon
        rows.append(_ridge_features(values, weeks, anchor, int(weeks[target_idx]), max_lag))
        targets.append(float(values[target_idx]))

    x = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    x_new = np.asarray([_ridge_features(values, weeks, len(values) - 1, future_week, max_lag)], dtype=float)

    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma < 1e-9] = 1.0
    xs = (x - mu) / sigma
    x_new_s = (x_new - mu) / sigma
    design = np.column_stack([np.ones(len(xs)), xs])
    design_new = np.column_stack([np.ones(len(x_new_s)), x_new_s])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0

    try:
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return float((design_new @ beta)[0])


def forecast_ridge_direct(train: pd.DataFrame, future_weeks: np.ndarray, max_lag: int = 8) -> np.ndarray:
    values = _safe_array(train["CV"])
    weeks = np.asarray(train["iso_week"], dtype=int)
    if len(values) != len(weeks) or len(values) < max_lag + 24:
        return _fallback_forecast(values, len(future_weeks))

    preds = [
        _ridge_predict_for_horizon(values, weeks, int(future_week), horizon=horizon, max_lag=max_lag)
        for horizon, future_week in enumerate(future_weeks, start=1)
    ]
    return np.array(preds, dtype=float)


def _knn_predict_for_horizon(
    values: np.ndarray,
    weeks: np.ndarray,
    future_week: int,
    horizon: int,
    max_lag: int = 8,
    max_points: int = 520,
    neighbors: int = 18,
) -> float:
    if len(values) > max_points:
        values = values[-max_points:]
        weeks = weeks[-max_points:]

    if len(values) < max_lag + horizon + neighbors:
        return float(values[-1])

    rows: list[list[float]] = []
    targets: list[float] = []
    for anchor in range(max_lag - 1, len(values) - horizon):
        target_idx = anchor + horizon
        rows.append(_ridge_features(values, weeks, anchor, int(weeks[target_idx]), max_lag))
        targets.append(float(values[target_idx] - values[anchor]))

    x = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    x_new = np.asarray([_ridge_features(values, weeks, len(values) - 1, future_week, max_lag)], dtype=float)
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma[sigma < 1e-9] = 1.0
    xs = (x - mu) / sigma
    x_new_s = (x_new - mu) / sigma
    distances = np.linalg.norm(xs - x_new_s, axis=1)
    k = min(neighbors, len(distances))
    nearest_idx = np.argpartition(distances, k - 1)[:k]
    nearest_dist = distances[nearest_idx]
    scale = float(np.median(nearest_dist[nearest_dist > 0])) if np.any(nearest_dist > 0) else 1.0
    weights = np.exp(-nearest_dist / max(scale, 1e-6))
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
        weights = np.ones_like(nearest_dist)
    predicted_delta = float(np.dot(weights, y[nearest_idx]) / weights.sum())
    return float(values[-1] + predicted_delta)


def forecast_knn_direct(train: pd.DataFrame, future_weeks: np.ndarray, max_lag: int = 8) -> np.ndarray:
    values = _safe_array(train["CV"])
    weeks = np.asarray(train["iso_week"], dtype=int)
    if len(values) != len(weeks) or len(values) < max_lag + 32:
        return _fallback_forecast(values, len(future_weeks))

    preds = [
        _knn_predict_for_horizon(values, weeks, int(future_week), horizon=horizon, max_lag=max_lag)
        for horizon, future_week in enumerate(future_weeks, start=1)
    ]
    return np.array(preds, dtype=float)


def choose_regime_years(
    anchor_year: int,
    bull_years: list[int],
    bear_years: list[int],
    hist_years: list[int],
) -> list[int]:
    if anchor_year in bull_years:
        return [year for year in bull_years if year != anchor_year]
    if anchor_year in bear_years:
        return [year for year in bear_years if year != anchor_year]
    return hist_years


def predict_models(
    train: pd.DataFrame,
    future_dates: list[pd.Timestamp],
    hist_years: list[int],
    regime_years: list[int],
    horizon: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    future_weeks = _iso_weeks(future_dates)
    ar_preds, ar_info = forecast_ar(train, horizon)
    predictions = {
        "ar": ar_preds,
        "ridge": forecast_ridge_direct(train, future_weeks),
        "knn": forecast_knn_direct(train, future_weeks),
        "holt": forecast_holt_damped(train, horizon),
    }
    return predictions, {"ar": ar_info}


def rolling_backtest(
    weekly_df: pd.DataFrame,
    hist_years: list[int],
    bull_years: list[int],
    bear_years: list[int],
    horizon: int,
    min_train_weeks: int = 156,
    validation_weeks: int = 260,
    validation_stride: int = 4,
) -> pd.DataFrame:
    weekly = _clean_weekly(weekly_df)
    if len(weekly) <= min_train_weeks + horizon:
        return pd.DataFrame()

    start = max(min_train_weeks, len(weekly) - validation_weeks - horizon)
    rows: list[dict[str, object]] = []
    for anchor in range(start, len(weekly) - horizon, validation_stride):
        train = weekly.iloc[: anchor + 1].copy()
        actual_rows = weekly.iloc[anchor + 1 : anchor + horizon + 1].copy()
        future_dates = [pd.Timestamp(date) for date in actual_rows["Date"]]
        anchor_year = int(train["iso_year"].iloc[-1])
        regime_years = choose_regime_years(anchor_year, bull_years, bear_years, hist_years)
        preds, _model_info = predict_models(train, future_dates, hist_years, regime_years, horizon)
        current_value = float(train["CV"].iloc[-1])

        for h, (_, actual) in enumerate(actual_rows.iterrows(), start=1):
            record: dict[str, object] = {
                "anchor_date": str(pd.Timestamp(train["Date"].iloc[-1]).date()),
                "target_date": str(pd.Timestamp(actual["Date"]).date()),
                "horizon": h,
                "current": current_value,
                "actual": float(actual["CV"]),
            }
            for key in MODEL_KEYS:
                record[key] = float(preds[key][h - 1])
            rows.append(record)

    return pd.DataFrame(rows)


def _metric_summary(backtest: pd.DataFrame, model_keys: list[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if backtest.empty:
        return records

    for horizon, group in backtest.groupby("horizon"):
        actual = group["actual"].astype(float).to_numpy()
        current = group["current"].astype(float).to_numpy()
        actual_direction = np.sign(actual - current)
        for key in model_keys:
            pred = group[key].astype(float).to_numpy()
            mask = np.isfinite(pred) & np.isfinite(actual)
            if not mask.any():
                continue
            errors = pred[mask] - actual[mask]
            pred_direction = np.sign(pred[mask] - current[mask])
            comparable = actual_direction[mask] != 0
            if comparable.any():
                direction_accuracy = float(np.mean(pred_direction[comparable] == actual_direction[mask][comparable]))
            else:
                direction_accuracy = np.nan
            records.append(
                {
                    "horizon": int(horizon),
                    "model": key,
                    "model_label": DISPLAY_LABELS.get(key, key),
                    "mae": float(np.mean(np.abs(errors))),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "direction_accuracy": direction_accuracy,
                    "observations": int(mask.sum()),
                }
            )
    return records


def _weights_from_metrics(metrics: list[dict[str, object]]) -> dict[int, dict[str, float]]:
    weights: dict[int, dict[str, float]] = {}
    metric_df = pd.DataFrame(metrics)
    if metric_df.empty:
        return weights

    for horizon, group in metric_df.groupby("horizon"):
        raw: dict[str, float] = {}
        for _, row in group.iterrows():
            mae = float(row["mae"])
            if np.isfinite(mae) and mae > 0:
                raw[str(row["model"])] = 1.0 / (mae + 1e-6)
        total = sum(raw.values())
        if total <= 0:
            continue
        weights[int(horizon)] = {key: value / total for key, value in raw.items()}
    return weights


def _apply_ensemble(
    predictions: dict[str, np.ndarray],
    weights_by_horizon: dict[int, dict[str, float]],
    horizon: int,
) -> np.ndarray:
    values: list[float] = []
    for h in range(1, horizon + 1):
        weights = weights_by_horizon.get(h)
        if not weights:
            weights = {key: 1.0 / len(MODEL_KEYS) for key in MODEL_KEYS}
        numerator = 0.0
        denominator = 0.0
        for key, weight in weights.items():
            pred = float(predictions[key][h - 1])
            if np.isfinite(pred):
                numerator += weight * pred
                denominator += weight
        values.append(numerator / denominator if denominator else np.nan)
    return np.array(values, dtype=float)


def _add_ensemble_to_backtest(
    backtest: pd.DataFrame,
    weights_by_horizon: dict[int, dict[str, float]],
) -> pd.DataFrame:
    if backtest.empty:
        return backtest

    result = backtest.copy()
    ensemble_values: list[float] = []
    for _, row in result.iterrows():
        horizon = int(row["horizon"])
        weights = weights_by_horizon.get(horizon, {})
        if not weights:
            weights = {key: 1.0 / len(MODEL_KEYS) for key in MODEL_KEYS}
        numerator = 0.0
        denominator = 0.0
        for key, weight in weights.items():
            pred = float(row[key])
            if np.isfinite(pred):
                numerator += weight * pred
                denominator += weight
        ensemble_values.append(numerator / denominator if denominator else np.nan)
    result["ensemble"] = ensemble_values
    return result


def _interval_deltas(backtest: pd.DataFrame, horizon: int, model_key: str) -> tuple[float, float]:
    if backtest.empty or model_key not in backtest.columns:
        return -np.nan, np.nan
    group = backtest[backtest["horizon"] == horizon].copy()
    if group.empty:
        return -np.nan, np.nan
    errors = group["actual"].astype(float) - group[model_key].astype(float)
    errors = errors[np.isfinite(errors)]
    if errors.empty:
        return -np.nan, np.nan
    return float(errors.quantile(0.10)), float(errors.quantile(0.90))


def _display_labels_with_fit(model_info: dict[str, object]) -> dict[str, str]:
    labels = DISPLAY_LABELS.copy()
    ar_info = model_info.get("ar", {})
    if isinstance(ar_info, dict):
        best_p = int(ar_info.get("best_p") or 0)
        if best_p > 0:
            labels["ar"] = f"AR({best_p})自回归"
    return labels


def _apply_metric_labels(
    metrics: list[dict[str, object]],
    labels: dict[str, str],
) -> list[dict[str, object]]:
    relabeled: list[dict[str, object]] = []
    for record in metrics:
        item = dict(record)
        item["model_label"] = labels.get(str(item["model"]), str(item["model"]))
        relabeled.append(item)
    return relabeled


def _rank_models(
    metrics: list[dict[str, object]],
    labels: dict[str, str],
) -> list[dict[str, object]]:
    metric_df = pd.DataFrame(metrics)
    if metric_df.empty:
        return []

    rows: list[dict[str, object]] = []
    for model, group in metric_df.groupby("model"):
        mae = group["mae"].astype(float)
        rmse = group["rmse"].astype(float)
        direction = group["direction_accuracy"].astype(float)
        rows.append(
            {
                "model": str(model),
                "model_label": labels.get(str(model), str(model)),
                "mean_mae": float(mae.mean()),
                "mean_rmse": float(rmse.mean()),
                "mean_direction_accuracy": float(direction.mean()),
                "horizons": int(group["horizon"].nunique()),
            }
        )

    rows.sort(key=lambda item: (float(item["mean_mae"]), float(item["mean_rmse"])))
    for rank, item in enumerate(rows, start=1):
        item["rank"] = rank
    return rows


def _forecast_direction_filter(
    predictions: dict[str, np.ndarray],
    labels: dict[str, str],
    current_value: float,
    horizon: int,
    tolerance: float = 1e-9,
) -> tuple[list[str], dict[str, object]]:
    records: list[dict[str, object]] = []

    for key in MODEL_KEYS:
        values = np.asarray(predictions.get(key, []), dtype=float)
        if len(values) < horizon:
            final_value = np.nan
        else:
            final_value = float(values[horizon - 1])

        if np.isfinite(final_value):
            final_change = float(final_value - current_value)
        else:
            final_change = np.nan

        included = bool(np.isfinite(final_change) and final_change >= -tolerance)
        if not np.isfinite(final_change):
            direction = "missing"
        elif abs(final_change) <= tolerance:
            direction = "flat"
        elif final_change > 0:
            direction = "up"
        else:
            direction = "down"

        records.append(
            {
                "model": key,
                "model_label": labels.get(key, key),
                "horizon": horizon,
                "final_forecast": final_value,
                "final_change": final_change,
                "direction": direction,
                "included": included,
            }
        )

    included_keys = [str(record["model"]) for record in records if record["included"]]
    return included_keys, {
        "baseline_value": current_value,
        "rule": f"保留第{horizon}周预测值不低于最新周频原糖指数的模型",
        "included_models": included_keys,
        "excluded_models": [str(record["model"]) for record in records if not record["included"]],
        "models": records,
    }


def build_forecast_suite(
    weekly_df: pd.DataFrame,
    hist_years: list[int],
    bull_years: list[int],
    bear_years: list[int],
    horizon: int = 4,
) -> dict[str, object]:
    weekly = _clean_weekly(weekly_df)
    if weekly.empty:
        raise ValueError("周频数据为空，无法建立预测模型。")

    latest = weekly.iloc[-1]
    latest_date = pd.Timestamp(latest["Date"])
    latest_value = float(latest["CV"])
    future_dates = [latest_date + pd.Timedelta(days=7 * step) for step in range(1, horizon + 1)]
    current_year = int(latest["iso_year"])
    regime_years = choose_regime_years(current_year, bull_years, bear_years, hist_years)

    backtest = rolling_backtest(weekly, hist_years, bull_years, bear_years, horizon)
    all_metrics = _metric_summary(backtest, MODEL_KEYS)

    predictions, model_info = predict_models(weekly, future_dates, hist_years, regime_years, horizon)
    display_labels = _display_labels_with_fit(model_info)
    all_metrics = _apply_metric_labels(all_metrics, display_labels)
    eligible_model_keys, direction_filter = _forecast_direction_filter(
        predictions,
        display_labels,
        latest_value,
        horizon,
    )
    if not eligible_model_keys:
        predictions["flat"] = np.full(horizon, latest_value, dtype=float)
        display_labels["flat"] = "持平基准"
        eligible_model_keys = ["flat"]
        direction_filter["included_models"] = eligible_model_keys
        direction_filter["models"].append(
            {
                "model": "flat",
                "model_label": "持平基准",
                "horizon": horizon,
                "final_forecast": latest_value,
                "final_change": 0.0,
                "direction": "flat",
                "included": True,
            }
        )

    filtered_metrics = [record for record in all_metrics if str(record["model"]) in eligible_model_keys]
    filtered_labels = {key: display_labels.get(key, key) for key in eligible_model_keys}
    model_rank = _rank_models(filtered_metrics, display_labels)
    if not model_rank:
        model_rank = [
            {
                "model": key,
                "model_label": filtered_labels.get(key, key),
                "mean_mae": np.nan,
                "mean_rmse": np.nan,
                "mean_direction_accuracy": np.nan,
                "horizons": horizon,
                "rank": rank,
            }
            for rank, key in enumerate(eligible_model_keys, start=1)
        ]
    primary_model = str(model_rank[0]["model"])

    forecast_rows: list[dict[str, object]] = []
    future_iso = pd.Series(future_dates).dt.isocalendar()
    for idx, date in enumerate(future_dates):
        horizon_num = idx + 1
        interval_map: dict[str, dict[str, float]] = {}
        for key in eligible_model_keys:
            low_delta, high_delta = _interval_deltas(backtest, horizon_num, key)
            model_value = float(predictions[key][idx])
            interval_map[key] = {
                "low": model_value + low_delta if np.isfinite(low_delta) else np.nan,
                "high": model_value + high_delta if np.isfinite(high_delta) else np.nan,
            }
        record: dict[str, object] = {
            "date": pd.Timestamp(date).date().isoformat(),
            "horizon": horizon_num,
            "iso_year": int(future_iso["year"].iloc[idx]),
            "iso_week": int(future_iso["week"].iloc[idx]),
            "interval_low": interval_map[primary_model]["low"],
            "interval_high": interval_map[primary_model]["high"],
            "intervals": interval_map,
        }
        for key in eligible_model_keys:
            record[key] = float(predictions[key][idx])
        forecast_rows.append(record)

    forecast_table = pd.DataFrame(forecast_rows)
    return {
        "forecast_table": forecast_table,
        "backtest": backtest,
        "metrics": filtered_metrics,
        "model_labels": filtered_labels,
        "model_keys": eligible_model_keys,
        "model_rank": model_rank,
        "primary_model": primary_model,
        "factor_descriptions": FACTOR_DESCRIPTIONS,
        "model_info": model_info,
        "direction_filter": direction_filter,
        "validation_start": str(backtest["anchor_date"].min()) if not backtest.empty else None,
        "validation_end": str(backtest["anchor_date"].max()) if not backtest.empty else None,
    }
