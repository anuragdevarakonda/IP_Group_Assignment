import os
import re
import unicodedata
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# CONFIG / CONSTANTS
# ---------------------------------------------------------
# Repo-relative defaults (work locally + Streamlit Cloud)
INPUT_DIR = os.getenv("INPUT_DIR", "dirty_data/")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "clean_data/")

# Controlled vocab for store dimensions (still validated)
VALID_CITIES = ["Dubai", "Abu Dhabi", "Sharjah"]
VALID_CHANNELS = ["App", "Web", "Marketplace"]

# Categories are OPEN vocabulary. These are only "known" categories for optional governance logging.
KNOWN_CATEGORIES = ["Electronics", "Fashion", "Grocery", "Home", "Beauty"]

ISSUES_LOG = []

ISSUE_TYPE_MAP = {
    "Normalization": "DATA_NORMALIZATION",
    "New Category": "NEW_CATEGORY_OBSERVED",
    "Unknown Value": "UNKNOWN_VALUE",
    "Imputation": "MISSING_VALUE_IMPUTED",
    "Logic Error (Cost > Price)": "LOGIC_ERROR",
    "Parsing Error": "INVALID_FORMAT",
    "Date Range Outlier": "OUTLIER_DATE",
    "Duplicate ID": "DUPLICATE_ID",
    "Outlier (IQR)": "OUTLIER_VALUE",
    "Missing Value": "MISSING_VALUE",
    "Outlier (High)": "OUTLIER_VALUE",
    "Logic Error (Negative)": "INVALID_VALUE",
    "Outlier (Extreme)": "OUTLIER_VALUE",
    "Deduplication (Merged)": "DUPLICATE_ENTITY",
}

# Avoid spamming "new category observed" logs
_SEEN_NEW_CATEGORIES = set()


def log_issue(table, record_id_col, record_id_val, field, original, new, issue_type_raw):
    """Log an issue in the required format."""
    issue_type = ISSUE_TYPE_MAP.get(issue_type_raw, issue_type_raw.upper().replace(" ", "_"))

    # Construct Action Taken
    if new in ("DROPPED", "DROPPED (Unparsable)"):
        action = "Dropped Record"
    elif issue_type_raw == "Deduplication (Merged)":
        action = f"Merged into Store {new}"
    elif issue_type_raw == "Normalization":
        action = "Standardized"
    elif issue_type_raw == "New Category":
        action = "Observed New Category"
    elif pd.isna(original):
        action = f"Imputed with {new}"
    else:
        action = f"Corrected to {new}"

    ISSUES_LOG.append(
        {
            "record identifier": f"{table} | {record_id_col}: {record_id_val}",
            "issue_type": issue_type,
            "issue_detail": f"Field '{field}' had value '{original}'",
            "action_taken": action,
        }
    )


# ---------------------------------------------------------
# TEXT NORMALIZATION HELPERS
# ---------------------------------------------------------
_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")


def normalize_text(x) -> str:
    """Normalize text to remove hidden Unicode differences that can split categories in charts.

    - NFKC normalization
    - Replace NBSP with regular space
    - Remove zero-width characters
    - Collapse whitespace
    - Strip edges
    """
    if pd.isna(x):
        return ""
    s = str(x)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")  # NBSP
    s = _ZERO_WIDTH_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_free_text_category(x) -> str:
    """Open-vocabulary category normalization.

    Ensures ' sports', 'SPORTS', 'Sports', 'Sports\u00A0' all become 'Sports'.
    Also handles artifacts like 'sports/SPORTS/Sports' by using the first token.
    """
    s = normalize_text(x)
    if not s:
        return "Other"
    if "/" in s:
        s = normalize_text(s.split("/")[0])
    # Title-case for canonical display
    return s.title()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _boolish_to_int(series: pd.Series) -> pd.Series:
    """Convert boolean-like encodings to 0/1."""
    if series is None:
        return series

    # numeric fast-path
    num = pd.to_numeric(series, errors="coerce")
    if len(series) and float(num.notna().mean()) >= 0.85:
        return num.fillna(0).clip(0, 1)

    s = series.astype(str).map(normalize_text).str.lower()
    true_set = {"1", "true", "t", "yes", "y", "returned", "return", "r"}
    false_set = {"0", "false", "f", "no", "n", "not returned", "not_returned", "nr", "nan", "", "none"}
    out = s.map(lambda v: 1 if v in true_set else (0 if v in false_set else np.nan))
    return pd.to_numeric(out, errors="coerce").fillna(0).clip(0, 1)


# ---------------------------------------------------------
# 1. CLEAN STORES
# ---------------------------------------------------------
def clean_stores(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    print("--- Cleaning Stores ---")
    df_clean = df.copy()

    for col in ["store_id", "city", "channel"]:
        if col not in df_clean.columns:
            raise KeyError(f"stores table missing required column: {col}")

    city_patterns = {
        r"(?i)^du.*": "Dubai",
        r"(?i)^ab.*": "Abu Dhabi",
        r"(?i)^sh.*": "Sharjah",
    }
    channel_patterns = {
        r"(?i)^ap.*": "App",
        r"(?i)^we.*": "Web",
        r"(?i)^ma.*": "Marketplace",
    }

    # Clean City
    for idx, row in df_clean.iterrows():
        raw_city = row["city"]
        raw_city_str = str(raw_city) if not pd.isna(raw_city) else ""
        normalized_input = normalize_text(raw_city_str)

        new_city = normalized_input
        matched = False
        for pattern, valid_val in city_patterns.items():
            if re.match(pattern, normalized_input):
                new_city = valid_val
                matched = True
                break

        # Standardize title-case for display
        new_city = normalize_text(new_city).title()

        if normalize_text(raw_city_str) != normalize_text(new_city):
            log_issue("stores", "store_id", row["store_id"], "city", raw_city, new_city, "Normalization")
            df_clean.at[idx, "city"] = new_city

        if new_city not in VALID_CITIES:
            log_issue("stores", "store_id", row["store_id"], "city", raw_city, new_city, "Unknown Value")

    # Clean Channel
    for idx, row in df_clean.iterrows():
        raw_channel = row["channel"]
        raw_channel_str = str(raw_channel) if not pd.isna(raw_channel) else ""
        normalized_input = normalize_text(raw_channel_str)

        new_channel = normalized_input
        matched = False
        for pattern, valid_val in channel_patterns.items():
            if re.match(pattern, normalized_input):
                new_channel = valid_val
                matched = True
                break

        new_channel = normalize_text(new_channel).title()

        if normalize_text(raw_channel_str) != normalize_text(new_channel):
            log_issue("stores", "store_id", row["store_id"], "channel", raw_channel, new_channel, "Normalization")
            df_clean.at[idx, "channel"] = new_channel

        if new_channel not in VALID_CHANNELS:
            log_issue("stores", "store_id", row["store_id"], "channel", raw_channel, new_channel, "Unknown Value")

    # Deduplicate on (city, channel) and assign stable numeric store_id
    store_mapping: Dict = {}
    deduped_rows = []

    unique_groups = df_clean.groupby(["city", "channel"], dropna=False)

    next_store_id = 1
    for (city, channel), group in unique_groups:
        survivor_id = next_store_id
        next_store_id += 1

        old_ids = list(group["store_id"].unique())
        for old_id in old_ids:
            store_mapping[old_id] = survivor_id

        if len(old_ids) > 1:
            for old_id in old_ids[1:]:
                log_issue("stores", "store_id", old_id, "store_id", old_id, survivor_id, "Deduplication (Merged)")

        survivor_row = group.iloc[0].copy()
        survivor_row["store_id"] = survivor_id
        deduped_rows.append(survivor_row)

    df_deduped = pd.DataFrame(deduped_rows)
    df_deduped.sort_values("store_id", inplace=True)

    return df_deduped, store_mapping


# ---------------------------------------------------------
# 2. CLEAN PRODUCTS
# ---------------------------------------------------------
def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    print("--- Cleaning Products ---")
    df_clean = df.copy()

    for col in ["product_id", "category", "base_price_aed", "unit_cost_aed"]:
        if col not in df_clean.columns:
            raise KeyError(f"products table missing required column: {col}")

    df_clean["base_price_aed"] = pd.to_numeric(df_clean["base_price_aed"], errors="coerce")
    df_clean["unit_cost_aed"] = pd.to_numeric(df_clean["unit_cost_aed"], errors="coerce")

    # Known normalization patterns (not restrictive)
    category_patterns = {
        r"(?i)^elec.*": "Electronics",
        r"(?i)^fash.*": "Fashion",
        r"(?i)^groc.*": "Grocery",
        r"(?i)^home.*": "Home",
        r"(?i)^beauty.*": "Beauty",
    }

    for idx, row in df_clean.iterrows():
        raw_cat = row["category"]
        raw_cat_str = str(raw_cat) if not pd.isna(raw_cat) else ""

        # Use a normalized input for matching, but always compare/assign against the RAW string
        norm_in = normalize_text(raw_cat_str)

        if not norm_in:
            new_cat = "Other"
            log_issue("products", "product_id", row["product_id"], "category", raw_cat, new_cat, "Missing Value")
            df_clean.at[idx, "category"] = new_cat
            continue

        matched = False
        new_cat = None
        for pattern, valid_val in category_patterns.items():
            if re.match(pattern, norm_in):
                new_cat = valid_val
                matched = True
                break

        if not matched:
            new_cat = normalize_free_text_category(raw_cat_str)

            # Optional governance logging (once per new category)
            if new_cat not in KNOWN_CATEGORIES and new_cat not in _SEEN_NEW_CATEGORIES:
                _SEEN_NEW_CATEGORIES.add(new_cat)
                log_issue("products", "product_id", row["product_id"], "category", norm_in, new_cat, "New Category")

        # Always assign canonical category (this fixes leading/trailing/hidden whitespace issues)
        if normalize_text(raw_cat_str) != normalize_text(new_cat):
            log_issue("products", "product_id", row["product_id"], "category", raw_cat, new_cat, "Normalization")

        df_clean.at[idx, "category"] = new_cat

    # Handle missing/invalid base price
    missing_base = df_clean["base_price_aed"].isna() | (df_clean["base_price_aed"] <= 0)
    if missing_base.any():
        med_by_cat = df_clean.loc[~missing_base].groupby("category")["base_price_aed"].median().to_dict()
        global_median = float(df_clean.loc[~missing_base, "base_price_aed"].median()) if (~missing_base).any() else 100.0

        for idx in df_clean[missing_base].index:
            cat = df_clean.at[idx, "category"]
            new_base = float(med_by_cat.get(cat, global_median))
            log_issue("products", "product_id", df_clean.at[idx, "product_id"], "base_price_aed", df_clean.at[idx, "base_price_aed"], new_base, "Imputation")
            df_clean.at[idx, "base_price_aed"] = round(new_base, 2)

    # Missing unit cost -> category ratio, fallback 0.6
    valid_mask = df_clean["unit_cost_aed"].notna() & (df_clean["base_price_aed"] > 0)
    temp_df = df_clean[valid_mask].copy()
    temp_df["cost_ratio"] = temp_df["unit_cost_aed"] / temp_df["base_price_aed"]
    avg_ratios = temp_df.groupby("category")["cost_ratio"].mean().to_dict()

    missing_cost_mask = df_clean["unit_cost_aed"].isna()
    for idx in df_clean[missing_cost_mask].index:
        cat = df_clean.at[idx, "category"]
        base = float(df_clean.at[idx, "base_price_aed"])
        ratio = float(avg_ratios.get(cat, 0.6))
        new_cost = round(base * ratio, 2)
        log_issue("products", "product_id", df_clean.at[idx, "product_id"], "unit_cost_aed", np.nan, new_cost, "Imputation")
        df_clean.at[idx, "unit_cost_aed"] = new_cost

    # Validate cost <= price
    invalid_cost_mask = df_clean["unit_cost_aed"] > df_clean["base_price_aed"]
    for idx in df_clean[invalid_cost_mask].index:
        orig = float(df_clean.at[idx, "unit_cost_aed"])
        limit = float(df_clean.at[idx, "base_price_aed"])
        log_issue("products", "product_id", df_clean.at[idx, "product_id"], "unit_cost_aed", orig, limit, "Logic Error (Cost > Price)")
        df_clean.at[idx, "unit_cost_aed"] = limit

    return df_clean


# ---------------------------------------------------------
# 3. CLEAN SALES
# ---------------------------------------------------------
def clean_sales(df: pd.DataFrame, products_df: pd.DataFrame, store_mapping: Dict) -> pd.DataFrame:
    print("--- Cleaning Sales ---")
    df_clean = df.copy()

    for col in ["order_id", "order_time", "store_id", "product_id", "qty", "selling_price_aed", "discount_pct"]:
        if col not in df_clean.columns:
            raise KeyError(f"sales table missing required column: {col}")

    # Timestamp cleaning
    df_clean["order_time_clean"] = pd.to_datetime(df_clean["order_time"], errors="coerce")
    nat_mask = df_clean["order_time_clean"].isna()
    if nat_mask.any():
        for idx in df_clean[nat_mask].index:
            log_issue("sales", "order_id", df_clean.at[idx, "order_id"], "order_time", df_clean.at[idx, "order_time"], "DROPPED (Unparsable)", "Parsing Error")
        df_clean = df_clean.dropna(subset=["order_time_clean"])

    min_date = df_clean["order_time_clean"].quantile(0.01) if len(df_clean) else None
    out_of_range_mask = df_clean["order_time_clean"].dt.year < 2020
    for idx in df_clean[out_of_range_mask].index:
        orig = df_clean.at[idx, "order_time_clean"]
        log_issue("sales", "order_id", df_clean.at[idx, "order_id"], "order_time", orig, min_date, "Date Range Outlier")
        df_clean.at[idx, "order_time_clean"] = min_date

    df_clean["order_time"] = df_clean["order_time_clean"]
    df_clean.drop(columns=["order_time_clean"], inplace=True)

    # Store mapping
    orig_store_series = df_clean["store_id"].copy()
    df_clean["store_id"] = df_clean["store_id"].map(store_mapping)
    if df_clean["store_id"].isna().any():
        bad = df_clean["store_id"].isna()
        for idx in df_clean[bad].index:
            log_issue("sales", "order_id", df_clean.at[idx, "order_id"], "store_id", orig_store_series.loc[idx], "DROPPED", "Unknown Value")
        df_clean = df_clean.dropna(subset=["store_id"])
    df_clean["store_id"] = df_clean["store_id"].astype(int)

    # Deduplicate order_id
    dup_mask = df_clean.duplicated(subset=["order_id"], keep="first")
    if dup_mask.any():
        for idx in df_clean[dup_mask].index:
            log_issue("sales", "order_id", df_clean.at[idx, "order_id"], "order_id", df_clean.at[idx, "order_id"], "DROPPED", "Duplicate ID")
        df_clean = df_clean[~dup_mask]

    # qty numeric
    df_clean["qty"] = pd.to_numeric(df_clean["qty"], errors="coerce")
    bad_qty = df_clean["qty"].isna()
    if bad_qty.any():
        mean_qty = float(df_clean["qty"].dropna().mean()) if df_clean["qty"].dropna().any() else 1.0
        for idx in df_clean[bad_qty].index:
            log_issue("sales", "order_id", df_clean.at[idx, "order_id"], "qty", df_clean.at[idx, "qty"], mean_qty, "Parsing Error")
        df_clean.loc[bad_qty, "qty"] = mean_qty

    # selling_price numeric (strip currency)
    sp = df_clean["selling_price_aed"].astype(str).map(normalize_text).str.replace(r"[^0-9.\-]", "", regex=True)
    df_clean["selling_price_aed"] = pd.to_numeric(sp, errors="coerce")

    # Fill missing selling_price from base_price
    missing_price_mask = df_clean["selling_price_aed"].isna() | (df_clean["selling_price_aed"] == 0)
    if missing_price_mask.any():
        df_clean = df_clean.merge(products_df[["product_id", "base_price_aed"]], on="product_id", how="left")
        for idx in df_clean[missing_price_mask].index:
            base = df_clean.at[idx, "base_price_aed"]
            log_issue("sales", "order_id", df_clean.at[idx, "order_id"], "selling_price_aed", df_clean.at[idx, "selling_price_aed"], base, "Missing Value")
            df_clean.at[idx, "selling_price_aed"] = base
        df_clean.drop(columns=["base_price_aed"], inplace=True)

    # discount_pct numeric (strip %)
    disc = df_clean["discount_pct"].astype(str).map(normalize_text).str.replace("%", "", regex=False).str.strip()
    df_clean["discount_pct"] = pd.to_numeric(disc, errors="coerce")
    nan_disc_mask = df_clean["discount_pct"].isna()
    if nan_disc_mask.any():
        for idx in df_clean[nan_disc_mask].index:
            log_issue("sales", "order_id", df_clean.at[idx, "order_id"], "discount_pct", df_clean.at[idx, "discount_pct"], 0, "Missing Value")
        df_clean.loc[nan_disc_mask, "discount_pct"] = 0.0

    # return_flag -> 0/1 if present
    if "return_flag" in df_clean.columns:
        df_clean["return_flag"] = _boolish_to_int(df_clean["return_flag"])

    # IQR outliers (qty)
    Q1_q = df_clean["qty"].quantile(0.25)
    Q3_q = df_clean["qty"].quantile(0.75)
    IQR_q = Q3_q - Q1_q
    upper_q = Q3_q + 1.5 * IQR_q
    mean_qty_int = int(round(df_clean["qty"].mean())) if len(df_clean) else 1
    qty_outlier_mask = df_clean["qty"] > upper_q
    for idx in df_clean[qty_outlier_mask].index:
        orig = df_clean.at[idx, "qty"]
        log_issue("sales", "order_id", df_clean.at[idx, "order_id"], "qty", orig, mean_qty_int, "Outlier (IQR)")
        df_clean.at[idx, "qty"] = mean_qty_int

    # IQR outliers (price)
    Q1_p = df_clean["selling_price_aed"].quantile(0.25)
    Q3_p = df_clean["selling_price_aed"].quantile(0.75)
    IQR_p = Q3_p - Q1_p
    upper_p = Q3_p + 1.5 * IQR_p
    price_high_mask = df_clean["selling_price_aed"] > upper_p
    for idx in df_clean[price_high_mask].index:
        orig = df_clean.at[idx, "selling_price_aed"]
        log_issue("sales", "order_id", df_clean.at[idx, "order_id"], "selling_price_aed", orig, upper_p, "Outlier (High)")
        df_clean.at[idx, "selling_price_aed"] = round(float(upper_p), 2)

    df_clean.sort_values("order_time", inplace=True)
    return df_clean


# ---------------------------------------------------------
# 4. CLEAN INVENTORY
# ---------------------------------------------------------
def clean_inventory(df: pd.DataFrame, store_mapping: Dict) -> pd.DataFrame:
    print("--- Cleaning Inventory ---")
    df_clean = df.copy()

    for col in ["snapshot_date", "store_id", "product_id", "stock_on_hand", "reorder_point", "lead_time_days"]:
        if col not in df_clean.columns:
            raise KeyError(f"inventory table missing required column: {col}")

    df_clean["snapshot_date"] = pd.to_datetime(df_clean["snapshot_date"], errors="coerce")
    bad_dates = df_clean["snapshot_date"].isna()
    if bad_dates.any():
        for idx in df_clean[bad_dates].index:
            log_issue("inventory", "idx", idx, "snapshot_date", df_clean.at[idx, "snapshot_date"], "DROPPED (Unparsable)", "Parsing Error")
        df_clean = df_clean.dropna(subset=["snapshot_date"])

    # Store mapping
    df_clean["store_id"] = df_clean["store_id"].map(store_mapping)
    df_clean.dropna(subset=["store_id"], inplace=True)
    df_clean["store_id"] = df_clean["store_id"].astype(int)

    # Numeric coercion
    df_clean["stock_on_hand"] = pd.to_numeric(df_clean["stock_on_hand"], errors="coerce").fillna(0)
    df_clean["reorder_point"] = pd.to_numeric(df_clean["reorder_point"], errors="coerce").fillna(0)
    df_clean["lead_time_days"] = pd.to_numeric(df_clean["lead_time_days"], errors="coerce").fillna(0)

    # Aggregate after dedupe
    df_clean = df_clean.groupby(["snapshot_date", "store_id", "product_id"], as_index=False).agg(
        {"stock_on_hand": "sum", "reorder_point": "max", "lead_time_days": "max"}
    )

    # Negative stock -> 0
    neg_mask = df_clean["stock_on_hand"] < 0
    for idx in df_clean[neg_mask].index:
        orig = df_clean.at[idx, "stock_on_hand"]
        log_issue("inventory", "idx", idx, "stock_on_hand", orig, 0, "Logic Error (Negative)")
        df_clean.at[idx, "stock_on_hand"] = 0

    # Extreme high stock -> median
    median_stock = float(df_clean["stock_on_hand"].median()) if len(df_clean) else 0
    high_mask = df_clean["stock_on_hand"] > 5000
    for idx in df_clean[high_mask].index:
        orig = df_clean.at[idx, "stock_on_hand"]
        log_issue("inventory", "idx", idx, "stock_on_hand", orig, median_stock, "Outlier (Extreme)")
        df_clean.at[idx, "stock_on_hand"] = int(round(median_stock))

    return df_clean


# ---------------------------------------------------------
# MAIN EXECUTION (OPTIONAL CLI)
# ---------------------------------------------------------
def main():
    print("Loading datasets...")
    try:
        stores = pd.read_csv(os.path.join(INPUT_DIR, "stores.csv"))
        products = pd.read_csv(os.path.join(INPUT_DIR, "products.csv"))
        sales = pd.read_csv(os.path.join(INPUT_DIR, "sales_raw.csv"))
        inventory = pd.read_csv(os.path.join(INPUT_DIR, "inventory_snapshot.csv"))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    clean_stores_df, store_mapping = clean_stores(stores)
    clean_products_df = clean_products(products)
    clean_sales_df = clean_sales(sales, clean_products_df, store_mapping)
    clean_inventory_df = clean_inventory(inventory, store_mapping)

    _ensure_dir(OUTPUT_DIR)
    print("Saving cleaned datasets...")
    clean_stores_df.to_csv(os.path.join(OUTPUT_DIR, "stores_clean.csv"), index=False)
    clean_products_df.to_csv(os.path.join(OUTPUT_DIR, "products_clean.csv"), index=False)
    clean_sales_df.to_csv(os.path.join(OUTPUT_DIR, "sales_clean.csv"), index=False)
    clean_inventory_df.to_csv(os.path.join(OUTPUT_DIR, "inventory_clean.csv"), index=False)

    if ISSUES_LOG:
        issues_df = pd.DataFrame(ISSUES_LOG)
        issues_df.to_csv(os.path.join(OUTPUT_DIR, "issues.csv"), index=False)
        print(f"Issues Log created: issues.csv with {len(issues_df)} records.")
    else:
        print("No issues found (clean run).")

    print("Cleaning Complete.")


if __name__ == "__main__":
    main()
