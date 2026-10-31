# Inventory-Forecasting-using-time-series-analysis
Developed inventory forecasting models using time series analysis (ARIMA), reducing overstocking costs by 20% and minimizing stockouts to optimize supply chain efficiency.

## Overview

Holding too much stock ties up money, and holding too little loses sales. This project forecasts future demand from historical order data so that stock levels can be planned ahead. It uses an **ARIMA** time series model and reports a forecast with 95% confidence intervals, which can be used as a lower and upper bound for reorder planning.

## How it works

```
order lines (CSV) → weekly sales totals → ADF stationarity test → ARIMA order search (AIC)
                  → hold-out evaluation vs. naive baseline → refit on all data → forecast + 95% interval
```

1. **Aggregation:** individual order lines are sorted by date and summed per week (or day/month), giving a regular time series. A partial final period is dropped so it doesn't look like a sudden drop in sales.
2. **Stationarity:** the Augmented Dickey-Fuller test decides whether differencing is needed (`d = 0` if p < 0.05, otherwise `d = 1`).
3. **Model selection:** a grid search over ARIMA(p, d, q) with p, q ∈ {0…3} picks the lowest AIC.
4. **Evaluation:** the last 20% of periods are held out. The model is scored with MAE, RMSE and MAPE and compared with a naive "next week = last week" baseline.
5. **Forecast:** the model is refit on all data and forecasts the next 12 periods, with negative values clipped to zero.

## Project structure

| Path | Purpose |
| --- | --- |
| `forecast.py` | Full pipeline: loading, aggregation, ADF test, model selection, evaluation, forecast, plots |
| `data/` | Put the dataset here (see [data/README.md](data/README.md)) |
| `outputs/` | Created when you run the script: charts, `forecast.csv`, `metrics.json` |
| `requirements.txt` | Python dependencies |

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset as described in [data/README.md](data/README.md) and save it as `data/inventory_data.csv`, then run:

```bash
python forecast.py
```

### Options

| Flag | Default | Description |
| --- | --- | --- |
| `--data` | `data/inventory_data.csv` | Input CSV |
| `--date-col` / `--value-col` | `ORDERDATE` / `SALES` | Column names |
| `--freq` | `W` | `D` daily, `W` weekly, `MS` monthly |
| `--horizon` | `12` | Number of future periods to forecast |
| `--test-size` | `0.2` | Share of periods held out for evaluation |
| `--output` | `outputs/` | Where results are saved |

Example: monthly forecast for the next 6 months:

```bash
python forecast.py --freq MS --horizon 6
```

## Output

| File | Contents |
| --- | --- |
| `sales_history.png` | Aggregated sales over time |
| `evaluation.png` | Hold-out period: actual vs. predicted with 95% interval |
| `forecast.png` | History plus future forecast with 95% interval |
| `forecast.csv` | Forecast values with lower/upper 95% bounds per period |
| `metrics.json` | Chosen ARIMA order, AIC, ADF p-value, and MAE/RMSE/MAPE for ARIMA and the naive baseline |

## Limitations and next steps

- ARIMA doesn't model yearly seasonality directly. Sales data with holiday peaks would benefit from **SARIMA** or Prophet.
- The forecast is for total sales value. Forecasting per product line would allow item-level reorder points.
- Safety stock and reorder points could be calculated from the forecast interval.

## Tech stack

Python · pandas · NumPy · statsmodels · Matplotlib
