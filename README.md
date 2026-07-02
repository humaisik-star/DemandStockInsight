# Demand Forecasting & Inventory Optimization

A retail analytics project that turns historical sales into smarter inventory decisions: better forecasts, leaner stock, and stronger business outcomes.

> **Status:** 🚀 Active — core model logic is in place and results are ready to be documented.

---

## 🌟 Why this matters

Retail inventory is a high-stakes balancing act:

- **Too much stock** ties up cash, increases storage cost, and risks waste.
- **Too little stock** causes stock-outs, lost revenue, and unhappy customers.

This project solves that by forecasting demand and turning those forecasts into clear stock recommendations.

---

## 🎯 What this project delivers

- Accurate weekly demand forecasts from historical retail sales
- Inventory optimization recommendations based on forecast + safety stock
- Visual comparisons against a baseline stocking strategy
- Metrics that show business value, not just model accuracy

---

## 📦 Project summary

This repository demonstrates how to:

1. Prepare retail sales data for forecasting.
2. Build and evaluate a demand forecasting model.
3. Calculate recommended inventory levels.
4. Compare optimized inventory against business-as-usual.

The goal is simple: **reduce excess stock while maintaining service levels**.

---

## 📊 Key results to highlight

Once complete, these are the most important numbers readers should see:

- Forecast accuracy (e.g. MAPE, WMAE)
- Average inventory before vs after optimization
- Stock reduction percentage
- Stock-out or service-level change

A strong story is:

> Better forecasts → smarter stock levels → lower inventory cost with the same or better availability.

---

## 🧠 Data & approach

**Dataset:** Walmart Sales Forecast dataset from Kaggle

**What it includes:**

- Weekly sales per store and department
- Date, store, department IDs
- Holiday indicators
- External signals: temperature, fuel price, CPI, unemployment, markdowns

**Approach:**

- Clean and prepare the data
- Train a forecasting model (Azure ML / AutoML)
- Measure accuracy with business-friendly metrics
- Compute recommended stock using forecast + safety stock
- Compare results with a baseline inventory strategy

---

## 🛠️ Tools used

| Tool | Purpose |
|------|---------|
| Azure Machine Learning | Train and manage forecasting experiments |
| Azure AutoML | Automated model selection and tuning |
| Power BI | Visualize forecasts and inventory outcomes |
| Excel | Analyze safety stock and business impact |
| GitHub | Version control and documentation |

---

## 📈 Recommended visuals

To make the story easy to understand, include charts such as:

- Forecast vs actual sales over time
- Baseline inventory vs optimized inventory
- Forecast error distribution
- Inventory reduction impact

If possible, add illustrations or screenshot files under `results/` and reference them here.

---

## 📁 Repository structure

```text
.
├── data/            # raw and processed datasets
├── notebooks/       # model training and analysis
├── results/         # charts, dashboards, and summary tables
├── docs/            # reports and presentation materials
├── PROJECT_PLAN.md  # one-month local RAG project plan
└── README.md        # project overview
```

---

## 🚀 How to run

1. Download the Walmart Sales Forecast dataset from Kaggle.
2. Place the raw files in `data/`.
3. Open the notebook or Azure ML experiment.
4. Run data preparation.
5. Train the demand forecasting model.
6. Generate metrics and inventory recommendations.

---

## 📘 Project plan

For a full one-month program plan, see `PROJECT_PLAN.md`. It includes weekly objectives, hands-on exercises, and milestones for building a local offline Q&A assistant with Foundry Local and RAG.

---

## ✨ Useful additions

If helpful, add any of these to make the project even clearer:

- A short **project topic summary** for your chosen ML domain
- Example **results and metrics** once the model is trained
- Sample **screenshots or charts** under `results/`
- A short **demo script** or `run` section for the app

---

## 👥 Contributors

- **OguzBABA** (Industrial Engineering) — demand definition, data preparation, inventory analysis, business insights
- **Oğuz** (Computer Engineering) — model training, Azure implementation, technical pipeline

---

## 💡 Notes

- Designed for academic and educational use.
- The README is written so any reader can understand the full project story quickly.
- Add charts, performance metrics, and business conclusions once model results are finalized.
- The plan is adaptable to other ML topics, so the curriculum can be reused for a different domain.
