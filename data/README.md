# Dataset

The project uses the public **Sample Sales Data** dataset from Kaggle:
https://www.kaggle.com/datasets/kyanyoga/sample-sales-data

1. Download `sales_data_sample.csv` (a free Kaggle account is required).
2. Save it in this folder as `inventory_data.csv`.

The script only needs two columns:

| Column | Example | Meaning |
| --- | --- | --- |
| `ORDERDATE` | `2/24/2003 0:00` | date of the order line |
| `SALES` | `2871.00` | sales value of the order line |

To use your own data, pass `--data`, `--date-col` and `--value-col` to `forecast.py`.
