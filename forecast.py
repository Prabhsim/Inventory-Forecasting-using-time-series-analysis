"""Inventory demand forecasting with ARIMA.

Pipeline:
    1. Load order-level sales data (CSV with a date column and a sales column).
    2. Aggregate orders into a regular time series (weekly by default).
    3. Test stationarity with the Augmented Dickey-Fuller (ADF) test.
    4. Pick ARIMA (p, d, q) by AIC with a small grid search.
    5. Evaluate on a hold-out period against a naive baseline (MAE, RMSE, MAPE).
    6. Refit on all data and forecast future demand with confidence intervals.

Usage:
    python forecast.py                          # uses data/inventory_data.csv
    python forecast.py --data path/to/file.csv --freq W --horizon 12
"""

import argparse
import itertools
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files; works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "data" / "inventory_data.csv"
DEFAULT_OUTPUT = BASE_DIR / "outputs"

FREQ_NAMES = {"D": "daily", "W": "weekly", "MS": "monthly"}
PERIOD_FREQ = {"D": "D", "W": "W", "MS": "M"}  # resample alias -> period alias


def load_series(path, date_col, value_col, freq):
    """Read order-level data and return total sales per period as a regular series."""
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    missing = {date_col, value_col} - set(df.columns)
    if missing:
        raise ValueError(f"Column(s) {sorted(missing)} not found. Available: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col])
    series = (
        df.set_index(date_col)[value_col]
        .sort_index()
        .resample(freq)
        .sum()  # periods with no orders become 0
    )

    # Drop the last period if the data stops part-way through it; a partial
    # week/month looks like a sales collapse and drags the forecast down.
    last_date = df[date_col].max().normalize()
    last_period = pd.Period(last_date, freq=PERIOD_FREQ[freq])
    if last_date < last_period.end_time.normalize():
        series = series.iloc[:-1]

    series.name = value_col
    return series


def adf_test(series):
    """Return (ADF statistic, p-value). p < 0.05 means the series looks stationary."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # statsmodels 0.15 return-type notice
        stat, p_value, *_ = adfuller(series.dropna())
    return stat, p_value


def select_order(series, d, max_p=3, max_q=3):
    """Grid-search ARIMA(p, d, q) and return the order with the lowest AIC."""
    best_order, best_aic = None, np.inf
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                aic = ARIMA(series, order=(p, d, q)).fit().aic
        except (ValueError, np.linalg.LinAlgError):
            continue
        if aic < best_aic:
            best_order, best_aic = (p, d, q), aic
    return best_order, best_aic


def fit(series, order):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        return ARIMA(series, order=order).fit()


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    errors = actual - predicted
    nonzero = actual != 0
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "MAPE_%": float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None,
    }


def plot_history(series, freq_name, path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(series.index, series.values, color="#2f6fdb")
    ax.set(title=f"Total sales ({freq_name})", xlabel="Date", ylabel="Sales")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_evaluation(train, test, predicted, conf_int, order, path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train.index, train.values, label="Training data", color="#9aa5b1")
    ax.plot(test.index, test.values, label="Actual (hold-out)", color="#1f2933")
    ax.plot(test.index, predicted, label=f"ARIMA{order} prediction", color="#d64545")
    ax.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="#d64545", alpha=0.15, label="95% interval")
    ax.set(title="Hold-out evaluation", xlabel="Date", ylabel="Sales")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_forecast(series, forecast, conf_int, order, path):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(series.index, series.values, label="Historical sales", color="#2f6fdb")
    ax.plot(forecast.index, forecast.values, label=f"ARIMA{order} forecast", color="#d64545")
    ax.fill_between(forecast.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="#d64545", alpha=0.15, label="95% interval")
    ax.set(title="Inventory demand forecast", xlabel="Date", ylabel="Sales")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Forecast inventory demand with ARIMA.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="CSV file with order-level sales")
    parser.add_argument("--date-col", default="ORDERDATE")
    parser.add_argument("--value-col", default="SALES")
    parser.add_argument("--freq", default="W", choices=FREQ_NAMES, help="D = daily, W = weekly, MS = monthly")
    parser.add_argument("--horizon", type=int, default=12, help="number of future periods to forecast")
    parser.add_argument("--test-size", type=float, default=0.2, help="share of periods held out for evaluation")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.data.exists():
        parser.error(f"{args.data} not found. See data/README.md for how to get the dataset.")

    args.output.mkdir(parents=True, exist_ok=True)
    freq_name = FREQ_NAMES[args.freq]

    # 1-2. Load and aggregate
    series = load_series(args.data, args.date_col, args.value_col, args.freq)
    print(f"Loaded {len(series)} {freq_name} periods: {series.index[0].date()} to {series.index[-1].date()}")
    plot_history(series, freq_name, args.output / "sales_history.png")

    # 3. Stationarity -> differencing order d
    stat, p_value = adf_test(series)
    d = 0 if p_value < 0.05 else 1
    print(f"ADF statistic = {stat:.3f}, p-value = {p_value:.4f} -> "
          f"{'stationary' if d == 0 else 'non-stationary'}, using d = {d}")

    # 4-5. Train/test split, order selection, evaluation
    n_test = max(1, int(len(series) * args.test_size))
    train, test = series.iloc[:-n_test], series.iloc[-n_test:]
    order, aic = select_order(train, d)
    print(f"Best order on training data: ARIMA{order} (AIC = {aic:.1f})")

    eval_fit = fit(train, order)
    eval_pred = eval_fit.get_forecast(steps=n_test)
    predicted = eval_pred.predicted_mean
    arima_scores = metrics(test, predicted)
    naive_scores = metrics(test, np.repeat(train.iloc[-1], n_test))  # "next period = last period"
    plot_evaluation(train, test, predicted, eval_pred.conf_int(), order, args.output / "evaluation.png")

    print(f"\nHold-out evaluation on the last {n_test} {freq_name} periods:")
    print(f"  {'':10}{'MAE':>12}{'RMSE':>12}{'MAPE %':>10}")
    for name, s in (("ARIMA", arima_scores), ("Naive", naive_scores)):
        mape = f"{s['MAPE_%']:.1f}" if s["MAPE_%"] is not None else "n/a"
        print(f"  {name:10}{s['MAE']:>12,.0f}{s['RMSE']:>12,.0f}{mape:>10}")

    # 6. Refit on everything and forecast the future
    final_fit = fit(series, order)
    future = final_fit.get_forecast(steps=args.horizon)
    forecast = future.predicted_mean.clip(lower=0)  # demand can't be negative
    conf_int = future.conf_int().clip(lower=0)
    plot_forecast(series, forecast, conf_int, order, args.output / "forecast.png")

    table = pd.DataFrame({
        "forecast": forecast.round(2),
        "lower_95": conf_int.iloc[:, 0].round(2),
        "upper_95": conf_int.iloc[:, 1].round(2),
    })
    table.index.name = "period_start"
    table.to_csv(args.output / "forecast.csv")

    with open(args.output / "metrics.json", "w") as f:
        json.dump({"order": order, "aic": aic, "adf_p_value": p_value, "test_periods": n_test,
                   "arima": arima_scores, "naive": naive_scores}, f, indent=2)

    print(f"\nNext {args.horizon} {freq_name} periods:")
    print(table.to_string())
    print(f"\nSaved plots, forecast.csv and metrics.json to {args.output}/")


if __name__ == "__main__":
    main()
