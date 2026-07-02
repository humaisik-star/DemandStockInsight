# DemandStockInsight

A local demand and stock analysis project focused on retail inventory data. This repo explores demand behavior, stock impact, promotions, and sales patterns without relying on cloud-specific tooling.

> **Status:** ✅ End-to-end. A tuned XGBoost forecasts daily demand (**R² = 0.95**
> on held-out dates), and a forecast-driven inventory policy cuts stock **~29%**
> while holding a 95% service level.

---

## 📌 Overview

This project predicts future product demand from historical retail sales data and uses those forecasts to optimize inventory (stock) levels. The goal is to avoid two common and costly problems:

- **Overstocking** → money tied up, storage cost, waste.
- **Understocking** → stock-outs, lost sales, unhappy customers.

By forecasting demand and computing the right stock level, the project keeps inventory at the optimal point — neither too much nor too little.

---

## 🎯 Objectives

1. Build a machine learning model that forecasts weekly product demand.
2. Measure forecast accuracy with a clear error metric.
3. Translate forecasts into recommended stock levels (inventory optimization).
4. Compare the optimized strategy against a baseline to quantify the improvement.

---

## 🗂️ Dataset

- **Source:** Walmart Sales Forecast dataset (Kaggle)
- **Granularity:** Weekly sales per store and department
- **Key fields:** Date, Store, Department, Weekly Sales, Holiday flag, and external factors (temperature, fuel price, CPI, unemployment, markdowns)

*(Data is used for educational/academic purposes.)*

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|------|---------|
| Python 3.13 + pandas / numpy | Data preparation and feature engineering |
| scikit-learn | Preprocessing pipeline, baseline models, metrics |
| XGBoost | Gradient-boosted forecasting model (best performer) |
| Optuna | Hyperparameter optimization |
| matplotlib / seaborn | Exploratory analysis and evaluation plots |
| GitHub | Version control and project documentation |

Everything runs **locally and offline** — no cloud account required.

---

## 🧭 Methodology

1. **Data preparation** — load the daily panel (5 stores × 20 products × 760 days).
2. **Feature engineering** — calendar features (month, day-of-week, cyclical
   encodings) plus **lag and rolling-window demand** per store-product series.
   Outcome/identifier columns (`Units Sold`, `Units Ordered`) are dropped to
   avoid target leakage.
3. **Time-based split** — train on the past, test on the most recent 20% of
   dates so evaluation reflects real forecasting, not random interpolation.
4. **Modeling** — tune XGBoost with Optuna, then compare against Ridge,
   RandomForest, and HistGradientBoosting on identical features.
5. **Evaluation** — report RMSE, MAE, and R² on the held-out dates.
6. **Inventory optimization** — turn forecasts into stock policy. A model-based
   reorder point buffers only the *forecast error*, while a forecast-free
   ("naive") policy must buffer the full demand variability:
   `Reorder Point = forecast over lead time + z · σ · √(lead time)`.

---

## 📊 Results

Evaluated on the most recent **20% of dates** (held out from training), lower
RMSE/MAE is better and higher R² is better:

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| **XGBoost (tuned)** ⭐ | **9.54** | **6.61** | **0.953** |
| HistGradientBoosting | 23.15 | 17.33 | 0.725 |
| RandomForest | 27.52 | 20.05 | 0.612 |
| Ridge (linear baseline) | 32.69 | 25.36 | 0.452 |

The tuned XGBoost explains **95%** of the variance in daily demand. The biggest
drivers are promotions, epidemic periods, product category, weather/seasonality,
and the recent 7-day demand trend — see the plots below.

**Evaluation plots** (in [`results/`](results/)):
- `07_feature_importance.png` — what the model relies on
- `08_pred_vs_actual.png` — predicted vs. actual demand on the test set

> Reproduce with `python train_model.py --trials 40`. Full metrics are written to
> [`results/model_metrics.csv`](results/model_metrics.csv).

### 📦 Inventory optimization

Applying the forecast to stock policy at a **95% target service level** with a
**7-day lead time** (100 store-product series):

| Policy | Total avg inventory | Achieved service level |
|--------|--------------------:|:----------------------:|
| Naive (no forecast) | 55,135 units | 87.8% |
| **Model-based** ⭐ | **39,076 units** | **94.8%** |

The forecast-driven policy holds **~29% less inventory** *and* hits the target
service level, because it only needs to buffer the model's small forecast error
(σ ≈ 6) instead of the full demand swings (σ ≈ 40). Per-product recommendations
are in [`results/stock_recommendations.csv`](results/stock_recommendations.csv);
see `results/09_stock_optimization.png`.

---

## 👥 Team

| Name | Role |
|------|------|
| Hüma Işık (Industrial Engineering) | Problem definition, data preparation, inventory analysis, reporting |
| Oğuz Temelli (Computer Engineering) | Model training, Azure setup, technical implementation |

---

## 📁 Repository Structure

```
.
├── data/              # Datasets (demand_forecasting.csv, inventory_...csv)
├── src/
│   └── features.py    # Feature engineering (calendar + lag/rolling)
├── notebooks/         # Exploratory analysis
├── results/           # Charts, metrics, predictions, dashboards
├── models/            # Trained model + best hyperparameters
├── analyze_data.py    # Exploratory data analysis / charts
├── train_model.py     # Full training pipeline (Optuna + model comparison)
├── predict.py         # Inference on new data
├── stock.py           # Inventory optimization from forecasts
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## 🚀 How to Reproduce

```bash
# 1. Set up the environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # macOS + XGBoost also needs: brew install libomp

# 2. Train (feature engineering + Optuna tuning + model comparison)
python train_model.py --trials 40

# 3. Predict on new data (same schema as the training CSV)
python predict.py --input data/demand_forecasting.csv

# 4. Turn forecasts into stock recommendations
python stock.py --service-level 0.95 --lead-time 7
```

Outputs land in `results/` (metrics, plots, predictions) and `models/`
(the trained model and its tuned hyperparameters).

---

## 📄 License

For academic and educational use.
