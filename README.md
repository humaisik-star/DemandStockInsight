# DemandStockInsight

A local demand and stock analysis project focused on retail inventory data. This repo explores demand behavior, stock impact, promotions, and sales patterns without relying on cloud-specific tooling.

> **Status:** ✅ Data is loaded and initial analysis is ready. The notebook and script support further exploration.

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
| Microsoft Azure Machine Learning | Training and deploying the forecasting model |
| Azure AutoML | Automated model selection and training |
| Power BI | Visualizing forecasts and inventory results |
| Excel | Data preparation and inventory calculations |
| GitHub | Version control and project documentation |

---

## 🧭 Methodology

1. **Data preparation** — clean and structure the historical sales data.
2. **Forecasting** — train a demand forecasting model on Azure ML.
3. **Evaluation** — measure accuracy (e.g., WMAE / MAPE).
4. **Inventory optimization** — convert forecasts into stock recommendations:
   `Recommended Stock = Forecasted Demand + Safety Stock`
5. **Comparison** — benchmark against the current/baseline approach.

---

## 📊 Results

*To be completed after model training.*

| Metric | Baseline | Our Model |
|--------|----------|-----------|
| Forecast accuracy (e.g., MAPE) | — | — |
| Average stock level | — | — |
| Stock reduction (%) | — | — |
| Stock-out rate | — | — |

> Example target outcome: *Reduced average inventory by X% while keeping the stock-out rate stable.*

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
├── data/            # Datasets (or links to them)
├── notebooks/       # Model training and analysis
├── results/         # Charts, metrics, dashboards
├── docs/            # Report and presentation
└── README.md        # This file
```

---

## 🚀 How to Reproduce

1. Download the Walmart Sales Forecast dataset from Kaggle.
2. Open the Azure Machine Learning workspace.
3. Run the AutoML forecasting experiment on the prepared data.
4. Review forecasts and apply the inventory optimization step.

---

## 📄 License

For academic and educational use.
