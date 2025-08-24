# 💳 Transaction Categorization & Spending Insights

A lightweight **Streamlit application** and **CLI tool** that categorize banking transactions and surface spending insights.

Upload a CSV and get:

- ✅ Predicted category per transaction
- 📊 Summary KPIs and spend over time
- 🛍 Category and merchant breakdowns
- 🔁 Likely recurring charges (heuristic)
- ⚠️ Unusually high purchases (z-score by category)
- 📉 Budget vs Actual tracking with a Clear all budgets button

---

## 🎯 Objectives

- Provide an easy-to-use **UI (Streamlit)** and CLI for running transaction categorization.
- Deliver **_intuitive insights_** (KPIs, breakdowns, anomalies, budgets) from raw CSVs.
- Ensure reproducibility and robustness with flexible schema mapping.
- Keep the repo lightweight with sample data, trained pipelines, and clear outputs.

---

## 📂 Repository Structure

<details>
<summary>Click to expand</summary>
```bash
data/
┣ for_testing.csv
┗ README.md

models/
┣ pipeline_v2_300k.joblib
┣ pipeline.joblib
┗ README.md

notebooks/
┣ transaction_categorization.ipynb
┗ README.md

outputs/
┣ confusion_matrix_v2_300k.png
┣ confusion_matrix.png
┣ metrics_v2_300k.json
┣ metrics.json
┣ predictions_sample.csv
┣ predictions(for_testing).csv
┗ README.md

app.py
credit_card_transactions.csv.zip
README.md
requirements.txt
run_check.py

</details>```

---

## 📥 Input Schema

- **Required fields:**

  - `timestamp` → transaction datetime (variants auto-detected: trans_date_trans_time, datetime, date, etc.)
  - `merchant` → merchant/vendor/description field
  - `amount` → numeric amount (variants: amt, amnt, transaction_amount, value)

- **Optional fields:**

  - city
  - state
  - category (used only for agreement checks, not for scoring)

- ⚠️ Extra columns (ZIP, lat/long, etc.) are safely ignored.
- 📌 CSV format: UTF-8 (Comma delimited) recommended.

---

## ✨ Features

- 🔎 **Flexible schema mapping** with auto-detected header variants.

- 🗂 **Robust pipeline** aligned with training preprocessing.

- 📈 **Insights dashboard:**

  - Summary KPIs (total spend, count, avg ticket, date range)

  - Weekly spend trends (stacked by category)

  - Category and merchant breakdowns

  - Recurring charge detection (heuristic: cadence + variance)

  - Unusually high purchases (category-level z-scores)

  - Budget vs Actual view with reset button

- 🔒 **Inference only** → no local file writes; predictions downloadable directly.

---

## 🚀 Getting Started

1️⃣ Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
pip install -r requirements.txt

```

2️⃣ Run the App

```bash
streamlit run app.py
```

3️⃣ CLI Sanity Check

```bash
python run_check.py path/to/your.csv

```

---

## 📊 What You’ll See in the App

- **`KPIs`** → total spend, transaction count, avg ticket, date range

- **`Spend over time`** → weekly totals + stacked category chart

- **`Category breakdown`** → spend and counts, top categories table

- **`Top merchants`** → spend and counts (with optional filters)

- **`Recurring charges`** → monthly cadence, low-variance merchants

- **`Unusual purchases`** → flagged using z-scores within categories

- **`Budget vs Actual`** → select month, set category budgets, see variance, reset with Clear all

---

## 📦 Requirements

The pinned dependencies are listed in `requirements.txt`.

streamlit==1.37.1

pandas==2.2.2

numpy==1.26.4

scikit-learn==1.6.1

joblib==1.4.2

python-dateutil==2.9.0.post0

pytz==2024.1

---

## 📤 Outputs

- `📊 Metrics` → JSON summaries (e.g., metrics_v2_300k.json)

- `🖼 Diagnostics` → confusion matrices (.png)

- `📑 Predictions` → CSVs (sample + test set outputs)

---

## 📌 Dataset Credit

This project uses the **Credit Card Transactions Dataset** from Kaggle:  
🔗 [Credit Card Transactions Dataset — Priyam Choksi](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

## 📊 Dataset License

The dataset used in this project is the **[Credit Card Transactions Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)** by _Priyam Choksi_ on Kaggle.

⚠️ **Note:** The dataset is provided under its own license terms on Kaggle.  
This repository does **not** redistribute the dataset — only small anonymized samples (`for_testing.csv`) are included for demo purposes.  
To access the full dataset, please download it directly from Kaggle.
