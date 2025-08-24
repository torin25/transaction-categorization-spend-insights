import io
import re
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from datetime import timedelta

# ---------------- Config ----------------
MODEL_PATH = "models/pipeline_v2_300k.joblib"


AMOUNT_BINS = None  # keep None to use per-batch qcut fallback

# Canonical columns expected downstream
REQUIRED_CANON = ["timestamp", "merchant", "amount"]
OPTIONAL_CANON  = ["city", "state", "category"]

# Alias map (normalized: lowercase, remove non-alphanum). Extend as needed.
ALIASES = {
    "timestamp": {
        "transdatetranstime","transactiontime","transactiondatetime",
        "datetime","date","time","timestamp","transtime","transdatetime",
        "transdate","transtimestamp"
    },
    "merchant": {
        "merchant","merchantname","vendor","payee","description","narration",
        "merchantdesc","merchantdescription","merchanttext","merchant_details",
        "merchant_name"
    },
    "amount": {
        "amount","amt","amnt","transactionamount","txnamount","value",
        "debit","credit","amountinr","amtinr","totalamount","amount_rs",
        "amountinrs","txn_amt","amountrs"
    },
    "city": {"city","merchantcity","billingcity","txncity"},
    "state": {"state","merchantstate","region","province","txnstate"},
    "category": {"category","label","class","txncategory","merchantcategory"},
}

# ---------------- Utilities: schema detection ----------------
def _normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())

def guess_schema_columns(df: pd.DataFrame):
    """
    Return mapping canonical -> original column name in df (or None if not found),
    plus a list of missing required canonicals.
    """
    norm_map = {_normalize_col(c): c for c in df.columns}
    found = {}

    # Exact/alias matching
    for canon, alias_set in ALIASES.items():
        match = None
        for norm, orig in norm_map.items():
            if norm == canon or norm in alias_set:
                match = orig
                break
        found[canon] = match

    # Heuristic fallbacks
    # amount: try any column containing 'amount'
    if found["amount"] is None:
        for norm, orig in norm_map.items():
            if "amount" in norm:
                found["amount"] = orig
                break

    # timestamp: any date/time-like column
    if found["timestamp"] is None:
        for norm, orig in norm_map.items():
            if "date" in norm or "time" in norm or norm.endswith("ts") or "timestamp" in norm:
                found["timestamp"] = orig
                break

    # merchant: description-like field
    if found["merchant"] is None:
        for norm, orig in norm_map.items():
            if "desc" in norm or "narrat" in norm or "vendor" in norm or "payee" in norm or "merchant" in norm:
                found["merchant"] = orig
                break

    missing_req = [c for c in REQUIRED_CANON if found.get(c) is None]
    return found, missing_req

def align_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects and renames columns to canonical names expected by the pipeline.
    Raises a clear error if required columns are missing.
    """
    mapping, missing = guess_schema_columns(df)
    if missing:
        raise ValueError(
            "Missing required columns after auto-detection: "
            f"{missing}. Rename headers or provide recognizable aliases."
        )

    rename_map = {
        mapping["timestamp"]: "timestamp",
        mapping["merchant"]: "merchant",
        mapping["amount"]: "amount",
    }
    for opt in OPTIONAL_CANON:
        if mapping.get(opt):
            rename_map[mapping[opt]] = opt

    df_out = df.rename(columns=rename_map)
    keep = [c for c in ["timestamp","merchant","amount","city","state","category"] if c in df_out.columns]
    return df_out[keep] if keep else df_out

# ---------------- Feature engineering (match training) ----------------
def build_features(df: pd.DataFrame):
    df = df.copy()

    # Parse & clean
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["merchant"] = (
        df["merchant"].fillna("")
        .str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    before = len(df)
    df = df.dropna(subset=["timestamp","amount"]).reset_index(drop=True)
    dropped = before - len(df)

    # Features
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour
    df["log_amount"] = np.log1p(df["amount"])

    # amount_bin: fixed edges if provided, else per-batch qcut
    if AMOUNT_BINS:
        df["amount_bin"] = pd.cut(df["amount"], bins=AMOUNT_BINS, labels=False, include_lowest=True)
    else:
        try:
            df["amount_bin"] = pd.qcut(df["amount"], q=4, labels=False, duplicates="drop")
        except Exception:
            df["amount_bin"] = pd.cut(
                df["amount"],
                bins=[-np.inf, 10, 50, 200, np.inf],
                labels=False,
                include_lowest=True
            )

    return df, dropped

# ---------------- Insights helpers ----------------
def make_facts(out_df: pd.DataFrame) -> pd.DataFrame:
    facts = out_df.copy()
    facts["timestamp"] = pd.to_datetime(facts["timestamp"], errors="coerce")
    facts = facts.dropna(subset=["timestamp","amount"]).reset_index(drop=True)
    facts["date"] = facts["timestamp"].dt.date
    facts["week"] = facts["timestamp"].dt.to_period("W").astype(str)
    facts["month"] = facts["timestamp"].dt.to_period("M").astype(str)
    return facts

def recurring_detection(facts: pd.DataFrame):
    """
    Simple recurring detection heuristic:
    - At least 3 transactions for a merchant
    - Median interval ~ monthly (26-33 days)
    - Low amount variation (CV < 0.2)
    """
    if facts.empty:
        return pd.DataFrame(columns=["merchant","count","median_interval_days","amount_cv","last_seen","next_estimated"])

    df = facts.sort_values("timestamp").copy()
    g = df.groupby("merchant", as_index=False)

    # intervals between consecutive tx per merchant
    intervals = (
        df.groupby("merchant")["timestamp"]
        .apply(lambda s: s.sort_values().diff().dropna().dt.days)
    )

    if intervals.empty:
        return pd.DataFrame(columns=["merchant","count","median_interval_days","amount_cv","last_seen","next_estimated"])

    med_interval = intervals.groupby(level=0).median().rename("median_interval_days")

    # coefficient of variation for amounts per merchant
    amt_stats = g["amount"].agg(["mean","std","count"]).rename(columns={"mean":"mu","std":"sigma"})
    amt_stats["amount_cv"] = amt_stats.apply(lambda r: (r["sigma"] / r["mu"]) if r["mu"] and r["sigma"] is not None else np.nan, axis=1)

    last_seen = g["timestamp"].max().rename(columns={"timestamp":"last_seen"})

    rec = amt_stats.join(med_interval, how="left").merge(last_seen, on="merchant", how="left")
    rec = rec.rename(columns={"count":"count"})
    rec["median_interval_days"] = rec["median_interval_days"].fillna(np.inf)

    # filters
    mask = (
        (rec["count"] >= 3) &
        (rec["median_interval_days"].between(26, 33)) &
        (rec["amount_cv"].fillna(1.0) < 0.2)
    )
    rec = rec.loc[mask, ["merchant","count","median_interval_days","amount_cv","last_seen"]].copy()
    if not rec.empty:
        rec["next_estimated"] = rec["last_seen"] + rec["median_interval_days"].apply(lambda d: pd.Timedelta(days=float(d)))
    return rec.sort_values(["median_interval_days","merchant"]).reset_index(drop=True)

def anomaly_high_spend(facts: pd.DataFrame, top_n=5):
    """
    Highlight unusually large purchases within each predicted category using z-score.
    """
    if facts.empty:
        return facts

    by_cat = facts.groupby("pred_category")["amount"]
    mu = by_cat.transform("mean")
    sigma = by_cat.transform("std").replace(0, np.nan)
    facts = facts.copy()
    facts["z"] = (facts["amount"] - mu) / sigma
    out = facts.dropna(subset=["z"]).sort_values("z", ascending=False)
    return out[["timestamp","merchant","amount","pred_category","z"]].head(top_n)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Transaction Categorization & Spending Insights", layout="wide")
st.title("Transaction Categorization & Spending Insights")

st.markdown(
    "- Upload a CSV with flexible headers. The app will auto-detect columns for timestamp, merchant, and amount.\n"
    "- Optional columns: city, state, category (category is used only to compute agreement vs. predictions)."
)

# Load model
model_status = st.empty()
try:
    pipe = joblib.load(MODEL_PATH)
    model_status.success(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    model_status.error(f"Failed to load model: {e}")
    st.stop()

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# Read CSV
try:
    data = uploaded.read()
    df_in = pd.read_csv(io.BytesIO(data))
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.subheader("Preview of uploaded data")
st.dataframe(df_in.head(10), use_container_width=True)

# Auto-align schema
try:
    df_aligned = align_to_canonical(df_in)
except Exception as e:
    st.error(str(e))
    st.stop()

with st.expander("Detected column mapping"):
    mapping, _ = guess_schema_columns(df_in)
    st.write(mapping)

# Build features
try:
    df_feat, dropped = build_features(df_aligned)
    if dropped > 0:
        st.info(f"Dropped {dropped} rows with missing timestamp/amount after cleaning")
except Exception as e:
    st.error(f"Error during feature engineering: {e}")
    st.stop()

# Predict
try:
    preds = pipe.predict(df_feat)
except Exception as e:
    st.error(f"Error during prediction: {e}")
    st.stop()

out = df_aligned.copy()
out["pred_category"] = preds

# Predictions Preview
st.subheader("Predictions (first 50 rows)")
show_cols = [c for c in ["timestamp","merchant","amount","city","state","pred_category","category"] if c in out.columns]
st.dataframe(out[show_cols].head(50), use_container_width=True)

# Agreement vs provided labels (optional)
if "category" in out.columns:
    try:
        agree = (out["pred_category"] == out["category"]).mean()
        st.metric("Agreement with provided 'category' (uploaded rows)", f"{agree:.2%}")
    except Exception:
        pass

# ---------------- Insights ----------------
facts = make_facts(out)

# KPIs
st.markdown("### Summary KPIs")
if not facts.empty:
    total_spend = float(facts["amount"].sum())
    tx_count = int(len(facts))
    avg_ticket = float(facts["amount"].mean())
    dmin, dmax = facts["timestamp"].min(), facts["timestamp"].max()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Spend", f"{total_spend:,.2f}")
    col2.metric("Transactions", f"{tx_count}")
    col3.metric("Avg Ticket", f"{avg_ticket:,.2f}")
    col4.metric("Date Range", f"{dmin:%Y-%m-%d} → {dmax:%Y-%m-%d}")
else:
    st.info("No rows available after cleaning to compute KPIs.")

# Spend over time
st.markdown("### Spend Over Time")
if not facts.empty:
    ts = facts.set_index("timestamp")["amount"].resample("W").sum().rename("weekly_spend").to_frame()
    st.line_chart(ts)
    # Stacked by category (weekly)
    pivot_w = (
        facts.pivot_table(
            index=pd.Grouper(key="timestamp", freq="W"),
            columns="pred_category",
            values="amount",
            aggfunc="sum",
        ).fillna(0.0)
    )
    with st.expander("Weekly Spend by Category (stacked)"):
        st.area_chart(pivot_w)
else:
    st.info("Not enough data to render time-series charts.")

# Category breakdown
st.markdown("### Category Breakdown")
if not facts.empty:
    by_cat_amt = facts.groupby("pred_category")["amount"].sum().sort_values(ascending=False)
    by_cat_cnt = facts["pred_category"].value_counts()
    colA, colB = st.columns(2)
    with colA:
        st.bar_chart(by_cat_amt)
    with colB:
        st.bar_chart(by_cat_cnt)
    st.write("Top categories by spend:")
    st.dataframe(by_cat_amt.head(10).rename("total_spend").to_frame())
else:
    st.info("No categorized rows to summarize.")

# Top merchants
st.markdown("### Top Merchants")
if not facts.empty:
    cat_filter = st.selectbox("Filter by category (optional)", options=["(All)"] + sorted(facts["pred_category"].unique().tolist()))
    fdf = facts if cat_filter == "(All)" else facts[facts["pred_category"] == cat_filter]
    top_merchants = (
        fdf.groupby("merchant")["amount"].agg(["sum","count"])
        .rename(columns={"sum":"total_spend","count":"transactions"})
        .sort_values("total_spend", ascending=False)
        .head(15)
    )
    st.dataframe(top_merchants, use_container_width=True)
else:
    st.info("No merchant data to display.")

# Recurring detection
st.markdown("### Likely Recurring Charges (MVP)")
rec = recurring_detection(facts)
if rec.empty:
    st.write("No likely recurring merchants detected yet.")
else:
    rec_disp = rec.copy()
    rec_disp["amount_cv"] = rec_disp["amount_cv"].round(3)
    st.dataframe(rec_disp, use_container_width=True)

# Anomaly highlights
st.markdown("### Unusually High Purchases (by category)")
anom = anomaly_high_spend(facts, top_n=5)
if anom.empty:
    st.write("No anomalies detected (or insufficient variance).")
else:
    anom_disp = anom.copy()
    anom_disp["z"] = anom_disp["z"].round(2)
    st.dataframe(anom_disp, use_container_width=True)


# ---------------- Budget Check (explicit month + selectable categories) ----------------

st.markdown("### Budget Check")
if not facts.empty:
    # Build month list from data (YYYY-MM)
    months = sorted(facts["month"].unique().tolist())
    sel_month = st.selectbox(
        "Select month for budget comparison",
        options=months,
        index=len(months) - 1  # default to latest month
    )

    mdf = facts[facts["month"] == sel_month]
    if mdf.empty:
        st.info("No data available for the selected month.")
    else:
        # Let user pick up to 5 categories (default = top 5 by spend in selected month)
        all_cats = sorted(mdf["pred_category"].unique().tolist())
        default_cats = (
            mdf.groupby("pred_category")["amount"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .index
            .tolist()
        )

        chosen = st.multiselect(
            "Choose up to 5 categories",
            options=all_cats,
            default=default_cats,
            max_selections=5
        )

        if not chosen:
            st.caption("Select categories to define budgets.")
        else:
            # Use session_state per-input so we can clear all with one button
            cols = st.columns(len(chosen))
            for i, cat in enumerate(chosen):
                key = f"budget_{cat}"
                if key not in st.session_state:
                    st.session_state[key] = 0.0
                st.number_input(
                    f"Budget for {cat}",
                    min_value=0.0,
                    value=float(st.session_state[key]),
                    step=100.0,
                    key=key,
                )

            # Clear all budgets button
            def clear_budgets():
                for cat in chosen:
                    st.session_state[f"budget_{cat}"] = 0.0

            st.button("Clear all budgets", on_click=clear_budgets)

            # Compute Actuals for selected month and categories
            by_cat_curr = mdf.groupby("pred_category")["amount"].sum()
            rows = []
            for cat in chosen:
                budget = float(st.session_state[f"budget_{cat}"])
                actual = float(by_cat_curr.get(cat, 0.0))
                variance = budget - actual  # positive = under budget
                rows.append({
                    "category": cat,
                    "budget": budget,
                    "actual": actual,
                    "variance": variance
                })

            if rows:
                bdf = pd.DataFrame(rows)
                # Totals row
                bdf.loc[len(bdf)] = {
                    "category": "TOTAL",
                    "budget": bdf["budget"].sum(),
                    "actual": bdf["actual"].sum(),
                    "variance": bdf["variance"].sum(),
                }

                st.caption("Variance = budget − actual (positive means under budget)")

                # OPTION 1: Solid color (green for >=0, red for <0) – most legible
                def variance_style(v):
                    if pd.isna(v):
                        return ""
                    return (
                        "background-color: #e8f5e9; color: #1b5e20;"  # green
                        if v >= 0
                        else "background-color: #ffebee; color: #b71c1c;"  # red
                    )

                # styled = (
                #     bdf.style
                #     .map(variance_style, subset=["variance"])
                #     .format({"budget": "₹{:,.0f}", "actual": "₹{:,.0f}", "variance": "₹{:,.0f}"})
                # )

                # If you prefer your gradient instead of solid colors, replace `styled = ...` above with:
                styled = (
                    bdf.style
                    .background_gradient(subset=["variance"], cmap="RdYlGn")  # green positive, red negative
                    .format({"budget": "₹{:,.0f}", "actual": "₹{:,.0f}", "variance": "₹{:,.0f}"})
                )

                # IMPORTANT: Use st.table to preserve Styler colors on deployment
                st.table(styled)
else:
    st.info("No data to compute budgets.")




# Download predictions and insights (in-memory)
st.markdown("### Download Predictions")
csv_buf = io.StringIO()
out[show_cols].to_csv(csv_buf, index=False)
st.download_button(
    "Download predictions as CSV",
    csv_buf.getvalue(),
    file_name="predictions.csv",
    mime="text/csv"
)

st.caption("Tip: For exact parity with training, keep column names consistent or rely on auto-detection. "
           "For amount_bin reproducibility across datasets, load fixed bin edges saved during training.")
