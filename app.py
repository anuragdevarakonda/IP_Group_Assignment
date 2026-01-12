# app.py
# UAE Omnichannel Retail – Data Rescue + Promo Pulse (Streamlit)
# Expected repo layout (same folder as this file):
#   cleaner.py
#   simulator_promo.py
#   dirty_data/  (products.csv, stores.csv, sales_raw.csv, inventory_snapshot.csv, campaign_plan.csv)
#   clean_data/  (products_clean.csv, stores_clean.csv, sales_clean.csv, inventory_clean.csv, issues.csv)  [optional]
#
# Notes:
# - No ML/forecasting: simulator is rule-based (per simulator_promo.py).
# - KPIs are reported pre-tax; tax_rate is shown for informational context.

from __future__ import annotations

import io
import hashlib
import json
import zipfile
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Plot styling defaults (presentation-friendly)
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = px.colors.qualitative.Vivid

# Custom vibrant palette (presentation-friendly)
VIBRANT_COLORWAY = [
    "#00A3E0",  # blue
    "#FF6F61",  # coral
    "#6B5B95",  # purple
    "#88B04B",  # green
    "#F7CAC9",  # light pink
    "#92A8D1",  # periwinkle
    "#F4A261",  # orange
    "#2A9D8F",  # teal
    "#E76F51",  # red-orange
    "#E9C46A",  # yellow
    "#264653",  # dark teal
    "#D90368",  # magenta
]


import cleaner
import simulator_promo as sim


# -------------------------
# Page config + styling
# -------------------------
st.set_page_config(page_title="UAE Retail – Data Rescue & Promo Pulse", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
      body { font-size: 16px; }
      div[data-testid="metric-container"] {
        background: rgba(250,250,250,0.85);
        border: 1px solid rgba(0,0,0,0.06);
        padding: 12px 14px;
        border-radius: 14px;
      }
      div[data-testid="stMetricLabel"] { font-size: 0.95rem; }
      div[data-testid="stMetricValue"] { font-size: 1.65rem; }
      div[data-testid="stMetricDelta"] { font-size: 0.95rem; }
      .section-title { font-size: 1.25rem; font-weight: 750; margin: 0.2rem 0 0.6rem 0; }
      .subtle { color: rgba(0,0,0,0.65); font-size: 1.0rem; }
      .callout {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 14px;
        padding: 14px;
        background: rgba(245,247,250,0.92);
      }
      .badge-ok {
        display:inline-block; padding:2px 10px; border-radius:999px;
        background:#e9f7ef; border:1px solid #bfe6cf; color:#17663a; font-size:0.85rem;
      }
      .badge-bad {
        display:inline-block; padding:2px 10px; border-radius:999px;
        background:#fdecec; border:1px solid #f5c2c2; color:#8a1f1f; font-size:0.85rem;
      }
      .hr { height:1px; background: rgba(0,0,0,0.06); margin: 14px 0; }
      .small-note { font-size: 0.85rem; color: rgba(0,0,0,0.62); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("UAE Retail – Data Rescue & Promo Pulse")
st.caption("Validation-first toolkit + rule-based promo simulator. KPIs are pre-tax; tax_rate is displayed for context.")


# -------------------------
# Canonical schema (expected)
# -------------------------
CANONICAL_SCHEMA: Dict[str, List[str]] = {
    "products": ["product_id", "category", "brand", "base_price_aed", "unit_cost_aed", "tax_rate", "launch_flag"],
    "stores": ["store_id", "city", "channel", "fulfillment_type"],
    "sales_raw": [
        "order_id", "order_time", "product_id", "store_id", "qty",
        "selling_price_aed", "discount_pct", "payment_status", "return_flag",
    ],
    "inventory_snapshot": ["snapshot_date", "product_id", "store_id", "stock_on_hand", "reorder_point", "lead_time_days"],
    "campaign_plan": ["campaign_id", "start_date", "end_date", "city", "channel", "category", "discount_pct", "promo_budget_aed"],
}
REQUIRED_TABLES = ["products", "stores", "sales_raw", "inventory_snapshot", "campaign_plan"]
REQUIRED_COLS = {
    "products": ["product_id", "category", "brand", "base_price_aed", "unit_cost_aed"],
    "stores": ["store_id", "city", "channel"],
    "sales_raw": ["order_id", "order_time", "product_id", "store_id", "qty", "selling_price_aed", "payment_status"],
    "inventory_snapshot": ["snapshot_date", "product_id", "store_id", "stock_on_hand"],
    "campaign_plan": ["campaign_id", "start_date", "end_date", "discount_pct", "promo_budget_aed"],
}
SYNONYMS = {
    "product_id": ["product_id", "sku", "item_id", "prod_id"],
    "store_id": ["store_id", "store", "branch_id", "outlet_id"],
    "order_id": ["order_id", "transaction_id", "txn_id", "invoice_id"],
    "order_time": ["order_time", "timestamp", "datetime", "order_datetime", "order_date"],
    "qty": ["qty", "quantity", "units"],
    "selling_price_aed": ["selling_price_aed", "price", "unit_price", "sale_price"],
    "discount_pct": ["discount_pct", "discount", "disc_pct", "discount_percent"],
    "payment_status": ["payment_status", "status", "payment"],
    "return_flag": ["return_flag", "returned", "is_return"],
    "snapshot_date": ["snapshot_date", "inventory_date", "as_of_date", "date"],
    "stock_on_hand": ["stock_on_hand", "stock", "on_hand", "soh", "available_stock"],
    "reorder_point": ["reorder_point", "rop", "min_stock", "threshold"],
    "lead_time_days": ["lead_time_days", "lead_time", "restock_days"],
    "promo_budget_aed": ["promo_budget_aed", "promo_budget", "budget"],
    "start_date": ["start_date", "start"],
    "end_date": ["end_date", "end"],
    "campaign_id": ["campaign_id", "campaign", "promo_id"],
    "city": ["city", "emirate"],
    "channel": ["channel", "platform"],
    "category": ["category", "cat"],
    "brand": ["brand", "make", "label"],
    "base_price_aed": ["base_price_aed", "base_price", "list_price", "regular_price"],
    "unit_cost_aed": ["unit_cost_aed", "unit_cost", "cost"],
    "tax_rate": ["tax_rate", "vat", "vat_rate", "tax"],
    "fulfillment_type": ["fulfillment_type", "fulfillment", "delivery_type"],
    "launch_flag": ["launch_flag", "is_new", "new_launch"],
}


# -------------------------
# Formatting helpers
# -------------------------
def fmt_aed(x: float) -> str:
    try:
        return f"AED {x:,.0f}"
    except Exception:
        return "AED -"


def fmt_pct(x: float, d: int = 1) -> str:
    try:
        return f"{x:.{d}f}%"
    except Exception:
        return "-"


def fmt_multi_selection(sel, all_vals, max_items: int = 4) -> str:
    """Format multi-select filters for display (show 'All' or a short list)."""
    if sel is None:
        return "All"
    if isinstance(sel, (list, tuple, set)):
        sel_list = [str(x) for x in sel if x is not None and str(x).strip() != ""]
        if len(sel_list) == 0:
            return "All"
        try:
            if all_vals and set(sel_list) == set([str(x) for x in all_vals]):
                return "All"
        except Exception:
            pass
        if len(sel_list) > max_items:
            return ", ".join(sel_list[:max_items]) + f" (+{len(sel_list) - max_items})"
        return ", ".join(sel_list)
    # Back-compat: single-select style
    return "All" if str(sel) == "All" else str(sel)

def fmt_delta_aed(d: float) -> str:
    """Format delta for st.metric (leading sign enables arrows/color)."""
    try:
        sign = "+" if d >= 0 else "-"
        return f"{sign}AED {abs(d):,.0f}"
    except Exception:
        return "n/a"


def fmt_delta_pp(d: float, decimals: int = 1) -> str:
    """Format percentage-point delta (leading sign enables arrows/color)."""
    try:
        sign = "+" if d >= 0 else "-"
        return f"{sign}{abs(d):.{decimals}f} pp"
    except Exception:
        return "n/a"


def prior_date_range(date_range):
    """Return a prior period date range of equal length (inclusive), or None."""
    if not date_range or len(date_range) != 2:
        return None
    start_d, end_d = date_range
    if start_d is None or end_d is None:
        return None
    days = (end_d - start_d).days + 1
    if days <= 0:
        return None
    prev_end = start_d - timedelta(days=1)
    prev_start = prev_end - timedelta(days=days - 1)
    return (prev_start, prev_end)


def apply_sales_filters(
    df: pd.DataFrame,
    date_range,
    city,
    channel,
    category,
    brand,
    fulfillment,
) -> pd.DataFrame:
    """Apply dashboard filters.

    Supports single-select (str/"All") and multi-select (list/tuple/set) inputs.
    """
    out = df.copy()
    if "order_time" in out.columns:
        out["order_time"] = pd.to_datetime(out["order_time"], errors="coerce")

    # Date range
    if date_range and len(date_range) == 2 and "order_time" in out.columns and out["order_time"].notna().any():
        start_d, end_d = date_range
        out = out[(out["order_time"].dt.date >= start_d) & (out["order_time"].dt.date <= end_d)]

    def _as_list(v):
        if v is None:
            return []
        if isinstance(v, (list, tuple, set)):
            return [x for x in v if x is not None and str(x).strip() != ""]
        if isinstance(v, str):
            return [] if v == "All" else [v]
        return [v]

    # Multi/single select handling
    cities = _as_list(city)
    channels = _as_list(channel)
    categories = _as_list(category)
    brands = _as_list(brand)
    fulf = _as_list(fulfillment)

    if cities and "city" in out.columns:
        out = out[out["city"].isin(cities)]
    if channels and "channel" in out.columns:
        out = out[out["channel"].isin(channels)]
    if categories and "category" in out.columns:
        out = out[out["category"].isin(categories)]
    if brands and "brand" in out.columns:
        out = out[out["brand"].isin(brands)]
    if fulf and "fulfillment_type" in out.columns:
        out = out[out["fulfillment_type"].isin(fulf)]

    return out

def compute_return_rate_pct(df: pd.DataFrame) -> float:
    """Return rate among paid transactions (pre-tax)."""
    if "return_flag" not in df.columns:
        return 0.0
    use = df.copy()
    if "payment_status" in use.columns:
        use = use[use["payment_status"] == "Paid"]
    if len(use) == 0:
        return 0.0
    rf = pd.to_numeric(use["return_flag"], errors="coerce").fillna(0.0)
    return float(rf.mean() * 100.0)


def compute_payment_failure_pct(df: pd.DataFrame) -> float:
    """Payment failure rate among all records with a payment_status."""
    if "payment_status" not in df.columns:
        return 0.0
    ps = df["payment_status"]
    denom = ps.notna().sum()
    if denom == 0:
        return 0.0
    return float((ps.astype(str) == "Failed").sum() / denom * 100.0)


def style_fig(fig, height: int = 360):
    """Consistent, presentation-ready Plotly formatting."""
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=46, b=12),
        font=dict(size=14),
        legend=dict(font=dict(size=12), title_font=dict(size=12)),
        colorway=VIBRANT_COLORWAY,
        hovermode="x unified",
    )
    fig.update_xaxes(title_font=dict(size=13), tickfont=dict(size=12))
    fig.update_yaxes(title_font=dict(size=13), tickfont=dict(size=12))
    return fig





def apply_bar_value_labels(fig, value_kind: str = 'number', max_points: int = 15) -> None:
    """Add succinct value labels to bar-like visuals for readability (no hover required).

    value_kind: 'number' | 'aed' | 'pct'
    """
    try:
        for tr in fig.data:
            if getattr(tr, 'type', None) != 'bar':
                continue
            # Determine point count
            n = 0
            if getattr(tr, 'x', None) is not None:
                try:
                    n = len(tr.x)
                except Exception:
                    n = 0
            if n and n > max_points:
                continue
            y = getattr(tr, 'y', None)
            if y is None:
                continue
            if value_kind == 'pct':
                text = [f"{float(v):.1f}%" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "" for v in y]
            elif value_kind == 'aed':
                text = [f"{float(v):,.0f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "" for v in y]
            else:
                text = [f"{float(v):,.0f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "" for v in y]
            tr.text = text
            tr.textposition = 'outside'
            tr.cliponaxis = False
            # keep labels small but readable
            tr.textfont = dict(size=12)
    except Exception:
        return

def _normalize_id_series(s: pd.Series) -> pd.Series:
    """Normalize identifier series to integer IDs (robust to codes like 'S001', 'P-0007').

    This is required because the project cleaner expects store_id to be convertible to int in some paths.
    For external datasets, IDs are frequently alphanumeric; we normalize them consistently across tables.
    """
    if s is None or len(s) == 0:
        return s

    s_str = s.astype(str).str.strip()
    # Treat placeholders as missing
    s_str = s_str.replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # Attempt: extract digits and use them if they preserve uniqueness (e.g., S001 -> 1)
    digits = s_str.str.extract(r"(\d+)")[0]
    num = pd.to_numeric(digits, errors="coerce")

    coverage = float(num.notna().mean()) if len(num) else 0.0
    uniq_ok = False
    if coverage >= 0.95:
        try:
            uniq_ok = int(num.dropna().astype(int).nunique()) == int(s_str.dropna().nunique())
        except Exception:
            uniq_ok = False

    if coverage >= 0.95 and uniq_ok:
        return num.fillna(-1).astype(int)

    # Fallback: stable mapping from unique string IDs -> 1..N
    uniq = sorted(pd.Series(s_str.dropna().unique()).astype(str).tolist())
    mapper = {u: i + 1 for i, u in enumerate(uniq)}
    mapped = s_str.map(mapper)

    # Any truly unknown remains NaN (downstream cleaner may drop/log)
    return pd.to_numeric(mapped, errors="coerce")


def _build_id_mapping(ref: pd.Series) -> Dict[str, int]:
    """Build a stable mapping from external IDs (often alphanumeric) to integer IDs."""
    if ref is None:
        return {}

    s = ref.astype(str).str.strip()
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})

    nonnull = s.dropna()
    if len(nonnull) == 0:
        return {}

    digits = nonnull.str.extract(r"(\d+)")[0]
    num = pd.to_numeric(digits, errors="coerce")
    coverage = float(num.notna().mean()) if len(num) else 0.0

    if coverage >= 0.95:
        try:
            uniq_ok = int(num.dropna().astype(int).nunique()) == int(nonnull.nunique())
        except Exception:
            uniq_ok = False
        if uniq_ok:
            mapping = {}
            for orig, val in zip(nonnull.tolist(), num.astype(int).tolist()):
                mapping.setdefault(str(orig), int(val))
            return mapping

    uniq = sorted(pd.Series(nonnull.unique()).astype(str).tolist())
    return {u: i + 1 for i, u in enumerate(uniq)}


def _apply_id_mapping(series: pd.Series, mapping: Dict[str, int]) -> pd.Series:
    if series is None:
        return series
    s = series.astype(str).str.strip()
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    out = s.map(mapping)
    return pd.to_numeric(out, errors="coerce")


def normalize_ids_for_cleaner(raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Normalize store_id and product_id across tables consistently for external datasets."""
    out = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in raw.items()}

    store_map: Dict[str, int] = {}
    if "stores" in out and isinstance(out["stores"], pd.DataFrame) and "store_id" in out["stores"].columns:
        store_map = _build_id_mapping(out["stores"]["store_id"])
        out["stores"]["store_id"] = _apply_id_mapping(out["stores"]["store_id"], store_map)

    for tbl in ("sales_raw", "inventory_snapshot"):
        if tbl in out and isinstance(out[tbl], pd.DataFrame) and "store_id" in out[tbl].columns and store_map:
            out[tbl]["store_id"] = _apply_id_mapping(out[tbl]["store_id"], store_map)

    prod_map: Dict[str, int] = {}
    if "products" in out and isinstance(out["products"], pd.DataFrame) and "product_id" in out["products"].columns:
        prod_map = _build_id_mapping(out["products"]["product_id"])
        out["products"]["product_id"] = _apply_id_mapping(out["products"]["product_id"], prod_map)

    for tbl in ("sales_raw", "inventory_snapshot"):
        if tbl in out and isinstance(out[tbl], pd.DataFrame) and "product_id" in out[tbl].columns and prod_map:
            out[tbl]["product_id"] = _apply_id_mapping(out[tbl]["product_id"], prod_map)

    return out

def _boolish_to_int(series: pd.Series) -> pd.Series:
    """Convert common boolean-like encodings to 0/1 numeric values.

    Handles: 1/0, True/False, Y/N, Yes/No, T/F, Returned/Not Returned.
    Unknowns become NaN (caller decides fill behavior).
    """
    if series is None:
        return series

    # Numeric fast path
    num = pd.to_numeric(series, errors="coerce")
    if len(series) and float(num.notna().mean()) >= 0.85:
        return num

    s_str = series.astype(str).str.strip().str.lower()
    true_set = {"1", "true", "t", "yes", "y", "returned", "return", "r"}
    false_set = {"0", "false", "f", "no", "n", "not returned", "not_returned", "nr", "none", "nan", ""}
    out = s_str.map(lambda v: 1 if v in true_set else (0 if v in false_set else np.nan))
    return out


def normalize_sales_for_simulator(df_sales: pd.DataFrame) -> pd.DataFrame:
    """Ensure simulator-critical sales columns have safe, numeric types.

    External datasets often encode fields like return_flag as strings (e.g., 'Y'/'N').
    The promo simulator expects numeric arithmetic on these fields.
    """
    if df_sales is None or not isinstance(df_sales, pd.DataFrame) or df_sales.empty:
        return df_sales

    df = df_sales.copy()

    if "return_flag" in df.columns:
        rf = _boolish_to_int(df["return_flag"])
        df["return_flag"] = pd.to_numeric(rf, errors="coerce").fillna(0).clip(0, 1)

    if "qty" in df.columns:
        df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0)

    if "selling_price_aed" in df.columns:
        s = df["selling_price_aed"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
        df["selling_price_aed"] = pd.to_numeric(s, errors="coerce").fillna(0)

    if "discount_pct" in df.columns:
        d = df["discount_pct"].astype(str).str.replace("%", "", regex=False).str.strip()
        df["discount_pct"] = pd.to_numeric(d, errors="coerce").fillna(0).clip(0, 100)

    if "payment_status" in df.columns:
        ps = df["payment_status"].astype(str).str.strip().str.title()
        repl = {
            "Completed": "Paid",
            "Success": "Paid",
            "Successful": "Paid",
            "Refund": "Refunded",
            "Returned": "Refunded",
            "Fail": "Failed",
            "Declined": "Failed",
            "Cancelled": "Failed",
            "Canceled": "Failed",
        }
        df["payment_status"] = ps.map(lambda v: repl.get(v, v))

    if "order_time" in df.columns:
        df["order_time"] = pd.to_datetime(df["order_time"], errors="coerce")

    return df





def normalize_col(c: str) -> str:
    return (
        str(c).strip().lower()
        .replace("(", "_").replace(")", "")
        .replace("%", "pct")
        .replace("/", "_").replace("-", "_")
        .replace(" ", "_")
    )


def suggest_table(cols_norm: List[str]) -> Optional[str]:
    s = set(cols_norm)
    if {"order_id", "order_time"}.issubset(s):
        return "sales_raw"
    if {"snapshot_date", "stock_on_hand"}.issubset(s) or {"stock_on_hand", "reorder_point"}.issubset(s):
        return "inventory_snapshot"
    if {"base_price_aed", "unit_cost_aed"}.issubset(s) or {"product_id", "brand", "category"}.issubset(s):
        return "products"
    if {"store_id", "city", "channel"}.issubset(s):
        return "stores"
    if {"campaign_id", "promo_budget_aed"}.issubset(s) or {"start_date", "end_date"}.issubset(s):
        return "campaign_plan"
    return None


def auto_map_columns(table: str, uploaded_cols: List[str]) -> Dict[str, Optional[str]]:
    cols_norm = {normalize_col(c): c for c in uploaded_cols}
    out: Dict[str, Optional[str]] = {}
    for canonical in CANONICAL_SCHEMA[table]:
        if canonical in cols_norm:
            out[canonical] = cols_norm[canonical]
            continue
        hit = None
        for s in SYNONYMS.get(canonical, []):
            if normalize_col(s) in cols_norm:
                hit = cols_norm[normalize_col(s)]
                break
        out[canonical] = hit
    return out


def coerce_types(table: str, df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    def to_dt(col: str):
        if col in d.columns:
            d[col] = pd.to_datetime(d[col], errors="coerce")
    def to_num(col: str):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    if table == "sales_raw":
        to_dt("order_time"); to_num("qty"); to_num("selling_price_aed"); to_num("discount_pct")
    if table == "inventory_snapshot":
        to_dt("snapshot_date"); to_num("stock_on_hand"); to_num("reorder_point"); to_num("lead_time_days")
    if table == "products":
        to_num("base_price_aed"); to_num("unit_cost_aed"); to_num("tax_rate")
    if table == "campaign_plan":
        to_dt("start_date"); to_dt("end_date"); to_num("discount_pct"); to_num("promo_budget_aed")
    return d


# -------------------------
# Repo data paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DIRTY_DIR = BASE_DIR / "dirty_data"
CLEAN_DIR = BASE_DIR / "clean_data"


@st.cache_data(show_spinner=False)
def load_repo_dirty() -> Dict[str, pd.DataFrame]:
    return {
        "products": pd.read_csv(DIRTY_DIR / "products.csv"),
        "stores": pd.read_csv(DIRTY_DIR / "stores.csv"),
        "sales_raw": pd.read_csv(DIRTY_DIR / "sales_raw.csv"),
        "inventory_snapshot": pd.read_csv(DIRTY_DIR / "inventory_snapshot.csv"),
        "campaign_plan": pd.read_csv(DIRTY_DIR / "campaign_plan.csv"),
    }


@st.cache_data(show_spinner=False)
def load_repo_clean() -> Dict[str, pd.DataFrame]:
    out = {
        "products": pd.read_csv(CLEAN_DIR / "products_clean.csv"),
        "stores": pd.read_csv(CLEAN_DIR / "stores_clean.csv"),
        "sales_raw": pd.read_csv(CLEAN_DIR / "sales_clean.csv"),
        "inventory_snapshot": pd.read_csv(CLEAN_DIR / "inventory_clean.csv"),
        "campaign_plan": pd.read_csv(DIRTY_DIR / "campaign_plan.csv") if (DIRTY_DIR / "campaign_plan.csv").exists() else pd.DataFrame(columns=CANONICAL_SCHEMA["campaign_plan"]),
        "issues": pd.read_csv(CLEAN_DIR / "issues.csv") if (CLEAN_DIR / "issues.csv").exists() else pd.DataFrame(columns=["record identifier","issue_type","issue_detail","action_taken"]),
    }
    out["sales_raw"]["order_time"] = pd.to_datetime(out["sales_raw"].get("order_time"), errors="coerce")
    out["inventory_snapshot"]["snapshot_date"] = pd.to_datetime(out["inventory_snapshot"].get("snapshot_date"), errors="coerce")
    if "start_date" in out["campaign_plan"].columns: out["campaign_plan"]["start_date"] = pd.to_datetime(out["campaign_plan"]["start_date"], errors="coerce")
    if "end_date" in out["campaign_plan"].columns: out["campaign_plan"]["end_date"] = pd.to_datetime(out["campaign_plan"]["end_date"], errors="coerce")
    return out


def clean_pipeline(raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    raw = normalize_ids_for_cleaner(raw)
    cleaner.ISSUES_LOG.clear()
    stores_clean, store_mapping = cleaner.clean_stores(raw["stores"])
    products_clean = cleaner.clean_products(raw["products"])
    sales_clean = cleaner.clean_sales(raw["sales_raw"], products_clean, store_mapping)
    inv_clean = cleaner.clean_inventory(raw["inventory_snapshot"], store_mapping)
    issues = pd.DataFrame(cleaner.ISSUES_LOG) if cleaner.ISSUES_LOG else pd.DataFrame(columns=["record identifier","issue_type","issue_detail","action_taken"])
    return {
        "products": products_clean,
        "stores": stores_clean,
        "sales_raw": sales_clean,
        "inventory_snapshot": inv_clean,
        "campaign_plan": raw.get("campaign_plan", pd.DataFrame(columns=CANONICAL_SCHEMA["campaign_plan"])).copy(),
        "issues": issues,
    }


# -------------------------
# Upload parsing: CSV / XLSX / ZIP(CSVs)
# -------------------------
def parse_uploads(files) -> Tuple[List[Tuple[str, pd.DataFrame]], str]:
    """Parse uploads into (name, dataframe) assets and return a stable content signature.

    The signature lets us persist canonical/cleaned tables across Streamlit reruns (view/filter changes)
    without forcing users to rebuild the ingestion pipeline each time.
    """
    assets: List[Tuple[str, pd.DataFrame]] = []
    h = hashlib.sha1()

    for f in files:
        name = f.name
        # Use getvalue() (non-destructive) rather than read() to avoid empty reads on reruns.
        b = f.getvalue()

        # Include both metadata and contents in the signature
        h.update(name.encode("utf-8", errors="ignore"))
        h.update(len(b).to_bytes(8, "little", signed=False))
        h.update(b)

        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                for member in z.namelist():
                    if member.lower().endswith(".csv") and not member.endswith("/"):
                        with z.open(member) as fh:
                            df = pd.read_csv(fh)
                        assets.append((f"{name}::{Path(member).name}", df))
        elif name.lower().endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(io.BytesIO(b))
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)
                assets.append((f"{name}::{sheet}", df))
        else:
            df = pd.read_csv(io.BytesIO(b))
            assets.append((name, df))

    return assets, h.hexdigest()


def mapping_wizard(assets: List[Tuple[str, pd.DataFrame]]) -> Optional[Dict[str, pd.DataFrame]]:
    if not assets:
        return None

    st.markdown("<div class='section-title'>Schema Mapping Wizard</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Assign each file/sheet to a table, then map columns via dropdowns.</div>", unsafe_allow_html=True)

    asset_names = [a[0] for a in assets]
    asset_df = {a[0]: a[1] for a in assets}
    suggestions = {n: suggest_table([normalize_col(c) for c in asset_df[n].columns]) for n in asset_names}

    # Step 1: table assignment
    st.markdown("#### Step 1 — Table assignment")
    cols = st.columns(5)
    for i, t in enumerate(REQUIRED_TABLES):
        with cols[i]:
            default = next((n for n in asset_names if suggestions.get(n) == t), asset_names[0])
            st.selectbox(t, options=asset_names, index=asset_names.index(default), key=f"assign_{t}")

    with st.expander("Preview assigned tables"):
        for t in REQUIRED_TABLES:
            n = st.session_state.get(f"assign_{t}")
            st.write(f"**{t}** ← {n}")
            st.dataframe(asset_df[n].head(3), use_container_width=True, height=140)

    # Step 2: column mapping
    st.markdown("#### Step 2 — Column mapping (dropdowns)")
    tabs = st.tabs(REQUIRED_TABLES)

    for t, tab in zip(REQUIRED_TABLES, tabs):
        with tab:
            src_name = st.session_state.get(f"assign_{t}")
            src = asset_df[src_name]
            src_cols = list(src.columns)
            suggested = auto_map_columns(t, src_cols)

            st.caption("Fields marked * are required to proceed.")
            grid = st.columns(3)
            for idx, canonical in enumerate(CANONICAL_SCHEMA[t]):
                with grid[idx % 3]:
                    options = ["(None)"] + src_cols
                    pre = suggested.get(canonical)
                    st.selectbox(
                        canonical + (" *" if canonical in REQUIRED_COLS[t] else ""),
                        options=options,
                        index=(options.index(pre) if pre in options else 0),
                        key=f"map_{t}_{canonical}",
                    )

            missing = [c for c in REQUIRED_COLS[t] if st.session_state.get(f"map_{t}_{c}") in (None, "(None)")]
            if missing:
                st.warning(f"Missing required mappings: {', '.join(missing)}")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    if not st.button("Build canonical tables", type="primary"):
        return None

    can_raw: Dict[str, pd.DataFrame] = {}
    for t in REQUIRED_TABLES:
        src_name = st.session_state.get(f"assign_{t}")
        df = asset_df[src_name].copy()

        # validate required mappings
        missing = [c for c in REQUIRED_COLS[t] if st.session_state.get(f"map_{t}_{c}") in (None, "(None)")]
        if missing:
            st.error(f"Cannot build {t}: missing required mappings: {', '.join(missing)}")
            return None

        # rename uploaded -> canonical
        rename = {}
        for canonical in CANONICAL_SCHEMA[t]:
            uploaded = st.session_state.get(f"map_{t}_{canonical}")
            if uploaded not in (None, "(None)"):
                rename[uploaded] = canonical
        df = df.rename(columns=rename)

        # keep canonical cols first
        canonical_cols = [c for c in CANONICAL_SCHEMA[t] if c in df.columns]
        extra_cols = [c for c in df.columns if c not in canonical_cols]
        df = df[canonical_cols + extra_cols]

        can_raw[t] = coerce_types(t, df)

    st.success("Canonical tables built successfully.")
    return can_raw


# -------------------------
# Core analytics helpers
# -------------------------
def _coalesce(df: pd.DataFrame, target: str, candidates: List[str], default: str = "Unknown") -> None:
    """Create/overwrite `target` using the first non-null among candidate columns."""
    series = None
    for c in candidates:
        if c in df.columns:
            if series is None:
                series = df[c]
            else:
                series = series.combine_first(df[c])
    if series is None:
        df[target] = default
    else:
        df[target] = series.fillna(default)


def enrich_sales(sales: pd.DataFrame, products: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Join sales with product/store dimensions and compute revenue, COGS and margin (defensive)."""
    s = sales.copy()

    # Normalize types used across analytics
    s["order_time"] = pd.to_datetime(s.get("order_time"), errors="coerce")
    s["qty"] = pd.to_numeric(s.get("qty"), errors="coerce").fillna(0)
    s["selling_price_aed"] = pd.to_numeric(s.get("selling_price_aed"), errors="coerce").fillna(0)

    # Avoid merge suffix collisions by renaming any pre-existing dims in sales extract
    for col in ("city", "channel", "fulfillment_type", "category", "brand", "tax_rate", "unit_cost_aed"):
        if col in s.columns:
            s.rename(columns={col: f"{col}_src"}, inplace=True)

    # Defensive: choose available columns in dimension tables
    prod_cols = ["product_id"] + [c for c in ("category", "brand", "tax_rate", "unit_cost_aed") if c in products.columns]
    store_cols = ["store_id"] + [c for c in ("city", "channel", "fulfillment_type") if c in stores.columns]

    if "product_id" in s.columns and "product_id" in products.columns:
        s = s.merge(products[prod_cols], on="product_id", how="left")
    if "store_id" in s.columns and "store_id" in stores.columns:
        s = s.merge(stores[store_cols], on="store_id", how="left")

    # Coalesce dims to canonical names
    _coalesce(s, "city", ["city", "city_y", "city_x", "city_src"])
    _coalesce(s, "channel", ["channel", "channel_y", "channel_x", "channel_src"])
    _coalesce(s, "fulfillment_type", ["fulfillment_type", "fulfillment_type_y", "fulfillment_type_x", "fulfillment_type_src"])
    _coalesce(s, "category", ["category", "category_y", "category_x", "category_src"])
    _coalesce(s, "brand", ["brand", "brand_y", "brand_x", "brand_src"])

    if "unit_cost_aed" not in s.columns:
        _coalesce(s, "unit_cost_aed", ["unit_cost_aed", "unit_cost_aed_y", "unit_cost_aed_x", "unit_cost_aed_src"], default="0")
    if "tax_rate" not in s.columns:
        _coalesce(s, "tax_rate", ["tax_rate", "tax_rate_y", "tax_rate_x", "tax_rate_src"], default="0")

    s["unit_cost_aed"] = pd.to_numeric(s["unit_cost_aed"], errors="coerce").fillna(0)
    s["tax_rate"] = pd.to_numeric(s["tax_rate"], errors="coerce").fillna(0)

    # Compute pre-tax revenue and COGS
    s["revenue"] = s["selling_price_aed"] * s["qty"]
    s["cogs"] = s["unit_cost_aed"] * s["qty"]
    s["gross_margin_aed"] = s["revenue"] - s["cogs"]
    return s


def net_revenue_ts(sales_enriched: pd.DataFrame) -> pd.DataFrame:
    df = sales_enriched.dropna(subset=["order_time"]).copy()
    df["date"] = df["order_time"].dt.date
    paid = df[df["payment_status"] == "Paid"]
    ref = df[df["payment_status"] == "Refunded"]
    paid_d = paid.groupby("date", as_index=False)["revenue"].sum().rename(columns={"revenue": "gross_revenue"})
    ref_d = ref.groupby("date", as_index=False)["revenue"].sum().rename(columns={"revenue": "refund_amount"})
    ts = paid_d.merge(ref_d, on="date", how="left")
    ts["refund_amount"] = ts["refund_amount"].fillna(0.0)
    ts["net_revenue"] = ts["gross_revenue"] - ts["refund_amount"]
    return ts.sort_values("date")


def inventory_risk(inventory: pd.DataFrame, products: pd.DataFrame, stores: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    inv = inventory.copy()
    inv["snapshot_date"] = pd.to_datetime(inv.get("snapshot_date"), errors="coerce")
    inv = inv.merge(products[["product_id", "category", "brand"]], on="product_id", how="left")
    inv = inv.merge(stores[["store_id", "city", "channel", "fulfillment_type"]], on="store_id", how="left")

    for c in ["stock_on_hand", "reorder_point", "lead_time_days"]:
        if c in inv.columns:
            inv[c] = pd.to_numeric(inv[c], errors="coerce")

    if inv["snapshot_date"].notna().any():
        inv = inv.sort_values("snapshot_date").groupby(["product_id", "store_id"], as_index=False).tail(1)

    b = baseline[["product_id", "store_id", "baseline_daily_qty"]].copy()
    inv = inv.merge(b, on=["product_id", "store_id"], how="left")
    inv["baseline_daily_qty"] = inv["baseline_daily_qty"].fillna(0.1)
    inv["days_of_cover"] = inv["stock_on_hand"] / inv["baseline_daily_qty"].replace(0, 0.1)
    inv["risk_score"] = (inv["lead_time_days"] - inv["days_of_cover"]).fillna(0).clip(lower=0)
    inv["risk_flag"] = (inv["days_of_cover"] < inv["lead_time_days"]).astype(int)
    return inv


def render_constraints(constraints: Dict) -> None:
    """Readable constraint panel with PASS/FAIL, numbers, deltas, and next-step guidance."""
    budget_ok = bool(constraints.get('budget_ok', True))
    margin_ok = bool(constraints.get('margin_ok', True))
    stock_ok = bool(constraints.get('stock_ok', True))
    ok_all = budget_ok and margin_ok and stock_ok

    # Pull numeric fields (present in simulator_promo.py)
    budget_used = float(constraints.get('budget_used', 0.0) or 0.0)
    budget_limit = float(constraints.get('budget_limit', 0.0) or 0.0)
    budget_util = float(constraints.get('budget_util_pct', 0.0) or 0.0)

    margin_ach = float(constraints.get('margin_achieved', 0.0) or 0.0)
    margin_floor = float(constraints.get('margin_floor', 0.0) or 0.0)

    stockout_count = int(constraints.get('stockout_count', 0) or 0)
    stockout_risk_pct = float(constraints.get('stockout_risk_pct', 0.0) or 0.0)

    st.markdown(
        f"<div class='callout'><span class='{'badge-ok' if ok_all else 'badge-bad'}'>"
        f"{'All constraints satisfied' if ok_all else 'Constraint violations detected'}"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    def _tile(title: str, ok: bool, body_html: str, hint: str) -> None:
        badge = 'badge-ok' if ok else 'badge-bad'
        st.markdown(
            f"""
<div style='border:1px solid rgba(255,255,255,.10); border-radius:16px; padding:14px 14px; background:rgba(255,255,255,.04);'>
  <div style='display:flex; align-items:center; justify-content:space-between;'>
    <div style='font-weight:700; font-size:14px'>{title}</div>
    <span class='{badge}'>{'PASS' if ok else 'FAIL'}</span>
  </div>
  <div style='margin-top:8px; font-size:13px; line-height:1.35;'>
    {body_html}
  </div>
  <div class='small-note' style='margin-top:10px'>{hint}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with c1:
        if budget_limit <= 0:
            body = "Promo budget not set (0)."
            hint = "Set a positive budget to enable budget feasibility checks."
        else:
            over = budget_used - budget_limit
            body = (
                f"Spend: <b>AED {budget_used:,.0f}</b><br>"
                f"Budget: <b>AED {budget_limit:,.0f}</b><br>"
                f"Utilization: <b>{budget_util:.1f}%</b>"
            )
            hint = (
                "If FAIL: reduce discount, narrow scope (city/channel/category), or shorten window."
                if over > 0
                else "Healthy budget headroom."
            )
        _tile("Budget", budget_ok, body, hint)

    with c2:
        delta_pp = margin_ach - margin_floor
        body = (
            f"Achieved: <b>{margin_ach:.1f}%</b><br>"
            f"Floor: <b>{margin_floor:.1f}%</b><br>"
            f"Delta: <b>{delta_pp:+.1f}pp</b>"
        )
        hint = (
            "If FAIL: reduce discount, exclude low-margin categories, or focus on higher-margin SKUs."
            if delta_pp < 0
            else "Margin constraint satisfied."
        )
        _tile("Margin", margin_ok, body, hint)

    with c3:
        body = (
            f"Stockout risk rows: <b>{stockout_count:,}</b><br>"
            f"Stockout risk: <b>{stockout_risk_pct:.1f}%</b>"
        )
        hint = (
            "If FAIL: reduce discount, restrict to high-cover SKUs, or remove low-stock store–SKU pairs."
            if not stock_ok
            else "No stockout-risk detected under current scenario."
        )
        _tile("Stock", stock_ok, body, hint)

    viols = constraints.get('violations', []) or []
    if viols:
        with st.expander("Top contributors (why it failed)", expanded=False):
            for v in viols:
                vtype = v.get('type', 'UNKNOWN').replace('_', ' ').title()
                st.markdown(f"**{vtype}**")
                top = v.get('top_contributors', []) or []
                if top:
                    st.dataframe(pd.DataFrame(top), use_container_width=True, height=260)
                else:
                    st.info("No contributor detail provided for this violation.")

def recommendation(constraints: Dict, sim_kpis: Dict, sim_kpis_base: Dict, discount_pct: float, budget: float) -> str:
    """Rule-based recommendation driven by constraints + uplift vs baseline."""
    budget_ok = bool(constraints.get('budget_ok', True))
    margin_ok = bool(constraints.get('margin_ok', True))
    stock_ok = bool(constraints.get('stock_ok', True))
    ok_all = budget_ok and margin_ok and stock_ok

    base_profit = float(sim_kpis_base.get('profit_proxy', 0.0) or 0.0)
    sim_profit = float(sim_kpis.get('profit_proxy', 0.0) or 0.0)
    delta_profit = sim_profit - base_profit

    promo_spend = float(sim_kpis.get('promo_spend', 0.0) or constraints.get('budget_used', 0.0) or 0.0)
    util = (promo_spend / max(budget, 1e-9)) * 100 if budget > 0 else 0.0

    stockout_risk_pct = float(sim_kpis.get('stockout_risk_pct', constraints.get('stockout_risk_pct', 0.0)) or 0.0)
    margin_pct = float(sim_kpis.get('promo_margin_pct', constraints.get('margin_achieved', 0.0)) or 0.0)
    margin_floor = float(constraints.get('margin_floor', 0.0) or 0.0)

    lines_out: List[str] = []

    # Headline decision
    if ok_all and delta_profit >= 0:
        decision = 'PROCEED'
    elif ok_all and delta_profit < 0:
        decision = 'ADJUST (profits down)';
    else:
        # If exactly one constraint fails, suggest adjust; if multiple fail, stop
        fails = sum([not budget_ok, not margin_ok, not stock_ok])
        decision = 'ADJUST' if fails == 1 else 'DO NOT PROCEED'

    lines_out.append(f"Decision: {decision}")
    lines_out.append(f"Scenario: {discount_pct:.1f}% discount (scope = current filters)")
    lines_out.append(f"Profit Proxy vs baseline: {fmt_aed(delta_profit)}")
    if budget > 0:
        lines_out.append(f"Budget utilization: {util:.1f}% (spend ≈ {fmt_aed(promo_spend)})")
    lines_out.append(f"Promo margin: {margin_pct:.1f}% (floor {margin_floor:.1f}%)")
    lines_out.append(f"Stockout risk: {stockout_risk_pct:.1f}%")

    # Action guidance
    actions: List[str] = []
    if not budget_ok:
        actions.append("Reduce discount or narrow scope (city/channel/category) to fit the promo budget.")
    if not margin_ok:
        actions.append("Reduce discount and/or exclude low-margin categories/SKUs to meet the margin floor.")
    if not stock_ok or stockout_risk_pct > 0:
        actions.append("Limit promo to high-cover SKUs (raise min days-of-cover) or remove low-stock store–SKU pairs.")
    if ok_all and delta_profit < 0:
        actions.append("Constraints pass, but profitability declines; consider lowering discount or targeting fewer segments.")

    if actions:
        lines_out.append("")
        lines_out.append("Recommended adjustments:")
        for a in actions[:4]:
            lines_out.append(f"- {a}")

    return "\n".join(lines_out)

with st.sidebar:
    st.markdown("### Data Source")
    data_mode = st.radio(
        "Choose a data source",
        options=[
            "Repo sample (already cleaned)",
            "Repo sample (clean from dirty_data now)",
            "Upload external dataset (schema mapping wizard)",
        ],
        index=0,
    )

    st.markdown("### Dashboard View")
    view_mode = st.radio("View", options=["Executive View", "Manager View"], index=0)

    st.markdown("### Simulation Controls")
    sim_days = st.selectbox("Simulation window", options=[7, 14], index=1)
    discount_pct = st.slider("Discount (%)", 0.0, 30.0, 10.0, 0.5)
    promo_budget_aed = st.number_input("Promo Budget (AED)", min_value=0.0, value=250000.0, step=5000.0)
    margin_floor_pct = st.slider("Margin floor (%)", 0.0, 60.0, 20.0, 0.5)
    st.markdown("<div class='small-note'>Constraints enforced: budget, margin floor, stock.</div>", unsafe_allow_html=True)
    st.markdown("### Simulation Eligibility")
    min_baseline_daily_qty = st.number_input(
    "Min baseline daily demand (SKU-store)",
    min_value=0.0,
    value=0.0,
    step=0.1,
    help="Exclude low-velocity SKU–store combinations from promo eligibility (baseline demand is computed from recent Paid sales).",
    )
    exclude_bottom_demand_pct = st.slider(
    "Exclude bottom demand SKUs (%)",
    min_value=0,
    max_value=50,
    value=0,
    step=5,
    help="Exclude the bottom X% of SKU–store rows by baseline daily demand within the selected simulation scope.",
    )
    min_days_of_cover_elig = st.slider(
    "Min Days of Cover (eligibility)",
    min_value=0.0,
    max_value=30.0,
    value=0.0,
    step=0.5,
    help="Exclude SKU–store rows with insufficient inventory coverage (stock_on_hand / baseline_daily_qty).",
    )



# -------------------------
# Data acquisition
# -------------------------
data_clean: Optional[Dict[str, pd.DataFrame]] = None

if data_mode == "Repo sample (already cleaned)":
    repo = load_repo_clean()
    data_clean = {
        "products": repo["products"],
        "stores": repo["stores"],
        "sales_raw": repo["sales_raw"],
        "inventory_snapshot": repo["inventory_snapshot"],
        "campaign_plan": repo["campaign_plan"],
        "issues": repo["issues"],
    }

elif data_mode == "Repo sample (clean from dirty_data now)":

    # Optional: generate a fresh dirty sample dataset (for demos) and clean it.
    # This affects only the Repo sample mode. External uploads are not impacted.
    with st.expander("Optional: Generate a fresh dirty sample dataset (Repo sample only)", expanded=False):
        st.markdown(
            "<div class='small-note'>Impact guide: "
            "<ul>"
            "<li><b>Seed</b>: changes values only (no row/column change)</li>"
            "<li><b>Days of sales history</b>: increases sales rows (orders spread across a longer horizon)</li>"
            "<li><b>Products</b>: increases product rows; impacts linked joins (sales/inventory reference more products)</li>"
            "<li><b>Stores</b>: increases store rows; impacts linked joins (sales/inventory reference more stores)</li>"
            "<li><b>Orders</b>: directly increases <b>sales_raw</b> row count</li>"
            "<li><b>Inventory snapshot days</b>: increases <b>inventory_snapshot</b> row count</li>"
            "<li><b>Columns</b>: remain constant (canonical schema)</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            gen_seed = st.number_input("Seed", min_value=1, value=42, step=1, key="repo_gen_seed")
            gen_days_history = st.number_input("Days of sales history", min_value=30, value=120, step=10, key="repo_gen_days_history")
        with c2:
            gen_n_products = st.number_input("Products", min_value=50, value=300, step=50, key="repo_gen_products")
            gen_n_stores = st.number_input("Stores", min_value=6, value=18, step=2, key="repo_gen_stores")
        with c3:
            gen_n_orders = st.number_input("Orders", min_value=2000, value=30000, step=2000, key="repo_gen_orders")
            gen_inventory_days = st.number_input("Inventory snapshot days", min_value=7, value=30, step=1, key="repo_gen_inventory_days")

        gen_clicked = st.button("Generate & load (Repo sample)", type="primary", key="repo_gen_btn")

    if gen_clicked:
        with st.spinner("Generating dirty data (generator.py) and running cleaning pipeline..."):
            try:
                # Import only on click to avoid side effects during normal app runs.
                import generator
                if not hasattr(generator, "generate_dirty_data"):
                    raise RuntimeError("generator.py must expose generate_dirty_data(...) for in-app generation.")
                raw = generator.generate_dirty_data(
                    seed=int(gen_seed),
                    n_products=int(gen_n_products),
                    n_stores=int(gen_n_stores),
                    n_orders=int(gen_n_orders),
                    days_history=int(gen_days_history),
                    inventory_days=int(gen_inventory_days),
                    write_csv=False,
                    output_path=None,
                )
                data_clean = clean_pipeline(raw)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                data_clean = None

    if data_clean is None:
        raw = load_repo_dirty()
        with st.spinner("Running cleaning + validation on dirty_data..."):
            data_clean = clean_pipeline(raw)

else:
    st.markdown("<div class='section-title'>External Dataset Intake</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtle'>Upload 5 tables as CSVs, an Excel workbook (5 sheets), or a ZIP of CSVs. "
        "Once canonical tables are built, you can freely switch views and adjust filters without rebuilding.</div>",
        unsafe_allow_html=True,
    )

    # Persist external ingestion results across Streamlit reruns (view/filter changes).
    if "external_ready" not in st.session_state:
        st.session_state.external_ready = False
        st.session_state.external_sig = None
        st.session_state.external_data_clean = None

    if st.session_state.external_ready and st.session_state.external_data_clean is not None:
        st.success("Canonical tables are ready. The dashboard will refresh with your selections without re-running ingestion.")
        if st.button("Rebuild canonical tables / Upload new dataset", type="secondary"):
            st.session_state.external_ready = False
            st.session_state.external_sig = None
            st.session_state.external_data_clean = None
            st.rerun()

        data_clean = st.session_state.external_data_clean

    else:
        files = st.file_uploader(
            "Upload files",
            type=["csv", "xlsx", "xls", "zip"],
            accept_multiple_files=True,
            key="external_upload_files",
        )

        # Allow reuse even if the uploader is cleared by a rerun
        if files:
            assets, sig = parse_uploads(files)

            # If a different dataset is uploaded, reset cached results.
            if st.session_state.external_sig is not None and st.session_state.external_sig != sig:
                st.session_state.external_ready = False
                st.session_state.external_data_clean = None

            # If we already built for this signature earlier, reuse it.
            if st.session_state.external_sig == sig and st.session_state.external_data_clean is not None:
                st.session_state.external_ready = True
                data_clean = st.session_state.external_data_clean
            else:
                can_raw = mapping_wizard(assets)
                if can_raw is not None:
                    with st.spinner("Running cleaning + validation on uploaded dataset..."):
                        data_clean = clean_pipeline({
                            "products": can_raw["products"],
                            "stores": can_raw["stores"],
                            "sales_raw": can_raw["sales_raw"],
                            "inventory_snapshot": can_raw["inventory_snapshot"],
                            "campaign_plan": can_raw["campaign_plan"],
                        })
                    st.session_state.external_ready = True
                    st.session_state.external_sig = sig
                    st.session_state.external_data_clean = data_clean

if data_clean is None:
    st.info("Select a data source (and complete mapping if uploading) to load the dashboard.")
    st.stop()

products_df = data_clean["products"].copy()
stores_df = data_clean["stores"].copy()
sales_df = data_clean["sales_raw"].copy()
inventory_df = data_clean["inventory_snapshot"].copy()
campaign_df = data_clean.get("campaign_plan", pd.DataFrame(columns=CANONICAL_SCHEMA["campaign_plan"])).copy()
issues_df = data_clean.get("issues", pd.DataFrame(columns=["record identifier","issue_type","issue_detail","action_taken"])).copy()

sales_df["order_time"] = pd.to_datetime(sales_df.get("order_time"), errors="coerce")

# -------------------------
# Post-clean validation (app-level safeguards)
# -------------------------
# Quantity must be >= 1 for a valid order line.
if "qty" in sales_df.columns:
    qty_num = pd.to_numeric(sales_df["qty"], errors="coerce")
    bad = qty_num.isna() | (qty_num <= 0)
    if bad.any():
        bad_df = sales_df.loc[bad, ["order_id", "qty"]].copy() if "order_id" in sales_df.columns else sales_df.loc[bad, ["qty"]].copy()
        # Log up to 5000 corrections to keep the issues file readable.
        max_log = 5000
        records = []
        for k, row in bad_df.head(max_log).iterrows():
            oid = row.get("order_id", k)
            orig = row.get("qty", None)
            records.append({
                "record identifier": f"sales | order_id: {oid}",
                "issue_type": "INVALID_VALUE",
                "issue_detail": f"Field 'qty' had value '{orig}'",
                "action_taken": "Corrected to 1",
            })
        if records:
            issues_df = pd.concat([issues_df, pd.DataFrame(records)], ignore_index=True)
        # Apply fix
        qty_num = qty_num.fillna(1).clip(lower=1).round()
    sales_df["qty"] = qty_num.astype(int)

inventory_df["snapshot_date"] = pd.to_datetime(inventory_df.get("snapshot_date"), errors="coerce")
if "start_date" in campaign_df.columns: campaign_df["start_date"] = pd.to_datetime(campaign_df["start_date"], errors="coerce")
if "end_date" in campaign_df.columns: campaign_df["end_date"] = pd.to_datetime(campaign_df["end_date"], errors="coerce")


# -------------------------
# Sidebar filters (min 5)
# -------------------------
with st.sidebar:
    st.markdown("### Filters")

    min_dt = sales_df["order_time"].min()
    max_dt = sales_df["order_time"].max()
    date_range = None
    if pd.notna(min_dt) and pd.notna(max_dt):
        date_range = st.date_input(
            "Order date range",
            value=(min_dt.date(), max_dt.date()),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
        )
    else:
        st.warning("Sales dates missing/invalid after cleaning; date filter disabled.")

    city_vals = sorted(stores_df.get("city", pd.Series(dtype=str)).dropna().unique().tolist())
    channel_vals = sorted(stores_df.get("channel", pd.Series(dtype=str)).dropna().unique().tolist())
    cat_vals = sorted(products_df.get("category", pd.Series(dtype=str)).dropna().unique().tolist())
    brand_vals = sorted(products_df.get("brand", pd.Series(dtype=str)).dropna().unique().tolist())
    fulf_vals = sorted(stores_df.get("fulfillment_type", pd.Series(dtype=str)).dropna().unique().tolist())

    f_city = st.multiselect("City", options=city_vals, default=city_vals)
    f_channel = st.multiselect("Channel", options=channel_vals, default=channel_vals)
    f_category = st.multiselect("Category", options=cat_vals, default=cat_vals)
    f_brand = st.multiselect("Brand", options=brand_vals, default=brand_vals)
    f_fulfillment = st.multiselect("Fulfillment Type", options=fulf_vals, default=fulf_vals)


# -------------------------
# Working slices
# -------------------------
sales_enriched = enrich_sales(sales_df, products_df, stores_df)

# Current-period slice (based on selected filters)
filtered = apply_sales_filters(sales_enriched, date_range, f_city, f_channel, f_category, f_brand, f_fulfillment)

# Prior-period slice (equal-length window immediately preceding current selection)
prev_range = prior_date_range(date_range)
filtered_prev = apply_sales_filters(sales_enriched, prev_range, f_city, f_channel, f_category, f_brand, f_fulfillment) if prev_range else pd.DataFrame(columns=filtered.columns)

paid_filtered = filtered[filtered["payment_status"] == "Paid"].copy()
ref_filtered = filtered[filtered["payment_status"] == "Refunded"].copy()

gross_rev_f = float(paid_filtered["revenue"].sum())
refund_f = float(ref_filtered["revenue"].sum())
net_rev_f = gross_rev_f - refund_f
cogs_f = float(paid_filtered["cogs"].sum())
margin_aed_f = gross_rev_f - cogs_f
margin_pct_f = (margin_aed_f / gross_rev_f * 100) if gross_rev_f > 0 else 0.0

# Prior-period KPIs for delta indicators (same filters, prior equal-length window)
paid_prev = filtered_prev[filtered_prev["payment_status"] == "Paid"].copy() if len(filtered_prev) else filtered_prev.copy()
ref_prev = filtered_prev[filtered_prev["payment_status"] == "Refunded"].copy() if len(filtered_prev) else filtered_prev.copy()

gross_rev_prev = float(paid_prev["revenue"].sum()) if len(paid_prev) else 0.0
refund_prev = float(ref_prev["revenue"].sum()) if len(ref_prev) else 0.0
net_rev_prev = gross_rev_prev - refund_prev

cogs_prev = float(paid_prev["cogs"].sum()) if len(paid_prev) else 0.0
margin_aed_prev = gross_rev_prev - cogs_prev
margin_pct_prev = (margin_aed_prev / gross_rev_prev * 100) if gross_rev_prev > 0 else 0.0

has_prev_period = bool(prev_range) and (len(paid_prev) + len(ref_prev) > 0)

# Time-based Ops rates for Manager View
return_rate_pct_f = compute_return_rate_pct(filtered)
return_rate_pct_prev = compute_return_rate_pct(filtered_prev) if has_prev_period else 0.0

payment_failure_pct_f = compute_payment_failure_pct(filtered)
payment_failure_pct_prev = compute_payment_failure_pct(filtered_prev) if has_prev_period else 0.0

# KPIs (baseline) + simulator preparation
# Note: some external sales extracts may already contain columns like 'city'/'channel'/'category'.
# The promo simulator merges these in from stores/products; to avoid merge suffixes (city_x/city_y),
# we rename any pre-existing columns in the sales extract.
sales_for_sim = normalize_sales_for_simulator(sales_df).copy()
for _col in ("city", "channel", "category"):
    if _col in sales_for_sim.columns:
        sales_for_sim.rename(columns={_col: f"{_col}_src"}, inplace=True)

# Ensure stores table provides required dimensions for baseline demand
missing_store_dims = [c for c in ("city", "channel") if c not in stores_df.columns]
if missing_store_dims:
    st.error(
        "Stores table is missing required columns for simulation: "
        + ", ".join(missing_store_dims)
        + ". Please map these fields in the External Dataset schema mapping step."
    )
    st.stop()

base_kpis, _ = sim.calculate_historical_kpis(sales_for_sim, products_df)

# Baseline demand & simulation
baseline_df = sim.calculate_baseline_demand(sales_for_sim, products_df, stores_df, lookback_days=30)
sim_filters = {
    "city": f_city,
    "channel": f_channel,
    "category": f_category,
    "min_baseline_daily_qty": float(min_baseline_daily_qty),
    "exclude_bottom_demand_pct": int(exclude_bottom_demand_pct),
    "min_days_of_cover_elig": float(min_days_of_cover_elig),
}
sim_out, constraints, sim_kpis = sim.run_simulation(
    baseline_df=baseline_df,
    products_df=products_df,
    inventory_df=inventory_df,
    filters=sim_filters,
    discount_pct=float(discount_pct),
    promo_budget_aed=float(promo_budget_aed),
    margin_floor_pct=float(margin_floor_pct),
    sim_days=int(sim_days),
)

# Baseline scenario (no discount) for scenario deltas (same scope & constraints)
sim_out_base, constraints_base, sim_kpis_base = sim.run_simulation(
    baseline_df=baseline_df,
    products_df=products_df,
    inventory_df=inventory_df,
    filters=sim_filters,
    discount_pct=0.0,
    promo_budget_aed=float(promo_budget_aed),
    margin_floor_pct=float(margin_floor_pct),
    sim_days=int(sim_days),
)

# Inventory risk (demand-aware)
inv_risk = inventory_risk(inventory_df, products_df, stores_df, baseline_df)
inv_risk_f = inv_risk.copy()
if isinstance(f_city, (list, tuple, set)) and len(f_city): inv_risk_f = inv_risk_f[inv_risk_f["city"].isin(list(f_city))]
if isinstance(f_channel, (list, tuple, set)) and len(f_channel): inv_risk_f = inv_risk_f[inv_risk_f["channel"].isin(list(f_channel))]
if isinstance(f_category, (list, tuple, set)) and len(f_category): inv_risk_f = inv_risk_f[inv_risk_f["category"].isin(list(f_category))]
if isinstance(f_brand, (list, tuple, set)) and len(f_brand): inv_risk_f = inv_risk_f[inv_risk_f["brand"].isin(list(f_brand))]
if isinstance(f_fulfillment, (list, tuple, set)) and len(f_fulfillment): inv_risk_f = inv_risk_f[inv_risk_f["fulfillment_type"].isin(list(f_fulfillment))]


# Stockout risk deltas: compare latest snapshot vs the previous available snapshot (if present)
inv_dates = pd.to_datetime(inventory_df.get("snapshot_date"), errors="coerce").dropna().sort_values().unique()
prev_inv_date = inv_dates[-2] if len(inv_dates) > 1 else None

inv_risk_prev = (
    inventory_risk(
        inventory_df[pd.to_datetime(inventory_df["snapshot_date"], errors="coerce") <= prev_inv_date],
        products_df,
        stores_df,
        baseline_df,
    )
    if prev_inv_date is not None else pd.DataFrame()
)

inv_risk_prev_f = inv_risk_prev.copy()
if len(inv_risk_prev_f):
    if isinstance(f_city, (list, tuple, set)) and len(f_city): inv_risk_prev_f = inv_risk_prev_f[inv_risk_prev_f["city"].isin(list(f_city))]
    if isinstance(f_channel, (list, tuple, set)) and len(f_channel): inv_risk_prev_f = inv_risk_prev_f[inv_risk_prev_f["channel"].isin(list(f_channel))]
    if isinstance(f_category, (list, tuple, set)) and len(f_category): inv_risk_prev_f = inv_risk_prev_f[inv_risk_prev_f["category"].isin(list(f_category))]
    if isinstance(f_brand, (list, tuple, set)) and len(f_brand): inv_risk_prev_f = inv_risk_prev_f[inv_risk_prev_f["brand"].isin(list(f_brand))]
    if isinstance(f_fulfillment, (list, tuple, set)) and len(f_fulfillment): inv_risk_prev_f = inv_risk_prev_f[inv_risk_prev_f["fulfillment_type"].isin(list(f_fulfillment))]
stockout_risk_pct = (inv_risk_f["risk_flag"].mean() * 100) if len(inv_risk_f) else 0.0
stockout_risk_pct_prev = (inv_risk_prev_f["risk_flag"].mean() * 100) if len(inv_risk_prev_f) else 0.0
high_risk_skus = int(inv_risk_f["risk_flag"].sum()) if len(inv_risk_f) else 0

tax_avg = pd.to_numeric(products_df.get("tax_rate"), errors="coerce").dropna().mean() if "tax_rate" in products_df.columns else np.nan


# -------------------------
# Scenario summary metrics (for Executive KPI cards)
# -------------------------
promo_spend = float(sim_kpis.get('promo_spend', constraints.get('budget_used', 0.0)) or 0.0) if isinstance(sim_kpis, dict) else float(constraints.get('budget_used', 0.0) or 0.0)
promo_spend_base = float(sim_kpis_base.get('promo_spend', constraints_base.get('budget_used', 0.0)) or 0.0) if isinstance(sim_kpis_base, dict) else float(constraints_base.get('budget_used', 0.0) or 0.0)
profit_proxy = float(sim_kpis.get('profit_proxy', 0.0) or 0.0) if isinstance(sim_kpis, dict) else 0.0
profit_proxy_base = float(sim_kpis_base.get('profit_proxy', 0.0) or 0.0) if isinstance(sim_kpis_base, dict) else 0.0
profit_proxy_delta = profit_proxy - profit_proxy_base
util = (promo_spend / max(float(promo_budget_aed), 1e-9)) * 100 if float(promo_budget_aed) > 0 else 0.0
util_base = (promo_spend_base / max(float(promo_budget_aed), 1e-9)) * 100 if float(promo_budget_aed) > 0 else 0.0
util_delta_pp = util - util_base


# -------------------------
# Top summary row
# -------------------------
a, b, c = st.columns([1.5, 1.1, 1.4])
with a:
    st.markdown("<div class='section-title'>Current Scope</div>", unsafe_allow_html=True)
    city_txt = fmt_multi_selection(f_city, city_vals)
    channel_txt = fmt_multi_selection(f_channel, channel_vals)
    category_txt = fmt_multi_selection(f_category, cat_vals)
    brand_txt = fmt_multi_selection(f_brand, brand_vals)
    fulfillment_txt = fmt_multi_selection(f_fulfillment, fulf_vals)
    st.markdown(
        f"<div class='callout'><div class='subtle'>Filters applied to charts</div>"
        f"<div><b>City</b>: {city_txt} | <b>Channel</b>: {channel_txt} | <b>Category</b>: {category_txt}</div>"
        f"<div><b>Brand</b>: {brand_txt} | <b>Fulfillment</b>: {fulfillment_txt}</div></div>",
        unsafe_allow_html=True,
    )
with b:
    st.markdown("<div class='section-title'>Tax Context</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='callout'><div class='subtle'>KPIs are pre-tax</div>"
        f"<div><b>Avg tax rate</b>: {fmt_pct(tax_avg if pd.notna(tax_avg) else 0.0, 2)}</div>"
        f"<div class='small-note'>Tax_rate is shown for context only.</div></div>",
        unsafe_allow_html=True,
    )
with c:
    st.markdown("<div class='section-title'>Data Quality Snapshot</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='callout'><div class='subtle'>Issues logged by cleaner</div>"
        f"<div><b>Total issues</b>: {len(issues_df):,}</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


# -------------------------
# Executive View
# -------------------------
if view_mode == "Executive View":
    st.markdown("<div class='section-title'>Executive View</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Financial health, revenue drivers, and promo decisioning.</div>", unsafe_allow_html=True)

    # KPI cards with deltas (arrows/color reflect improvement vs prior period / baseline scenario)
    # KPI cards with deltas
    # - Financial metrics use normal coloring (higher is better)
    # - Risk/quality metrics use inverse coloring (higher is worse)
    r1c1, r1c2, r1c3 = st.columns(3)
    r2c1, r2c2, r2c3 = st.columns(3)

    net_rev_delta = net_rev_f - net_rev_prev
    margin_pp_delta = margin_pct_f - margin_pct_prev
    return_pp_delta = return_rate_pct_f - return_rate_pct_prev
    refund_delta = refund_f - refund_prev

    r1c1.metric(
        "Net Revenue (Filtered)",
        fmt_aed(net_rev_f),
        delta=(fmt_delta_aed(net_rev_delta) if has_prev_period else "n/a"),
        delta_color="normal",
    )
    r1c2.metric(
        "Gross Margin % (Paid)",
        fmt_pct(margin_pct_f),
        delta=(fmt_delta_pp(margin_pp_delta) if has_prev_period else "n/a"),
        delta_color="normal",
    )
    r1c3.metric(
        "Return Rate % (Paid)",
        fmt_pct(return_rate_pct_f),
        delta=(fmt_delta_pp(return_pp_delta) if has_prev_period else "n/a"),
        delta_color="inverse",
    )

    r2c1.metric(
        "Refunded Amount (AED)",
        fmt_aed(refund_f),
        delta=(fmt_delta_aed(refund_delta) if has_prev_period else "n/a"),
        delta_color="inverse",
    )
    r2c2.metric(
        "Profit Proxy (Scenario)",
        fmt_aed(profit_proxy),
        delta=fmt_delta_aed(profit_proxy_delta),
        delta_color="normal",
    )
    r2c3.metric(
        "Budget Utilization (Scenario)",
        fmt_pct(util),
        delta=fmt_delta_pp(util - util_base),
        delta_color="normal",
    )
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("#### Net Revenue Trend (Monthly)")
        ts = net_revenue_ts(filtered)
        if len(ts):
            ts["date"] = pd.to_datetime(ts["date"])
            ts["month"] = ts["date"].dt.to_period("M").dt.to_timestamp()
            m = ts.groupby("month", as_index=False)["net_revenue"].sum().sort_values("month")
            m["roll_3m"] = m["net_revenue"].rolling(3).mean()
            m["roll_6m"] = m["net_revenue"].rolling(6).mean()

            fig = go.Figure()
            fig.add_trace(go.Bar(x=m["month"], y=m["net_revenue"], name="Net Revenue (Monthly)"))
            fig.add_trace(go.Scatter(x=m["month"], y=m["roll_3m"], name="Rolling Avg (3M)", mode="lines+markers", line=dict(width=3)))
            fig.add_trace(go.Scatter(x=m["month"], y=m["roll_6m"], name="Rolling Avg (6M)", mode="lines+markers", line=dict(width=3, dash="dash")))

            fig = style_fig(fig, height=360)
            apply_bar_value_labels(fig, value_kind='aed', max_points=12)
            fig.update_xaxes(title_text="Month", tickformat="%b %Y", dtick="M1")
            fig.update_yaxes(title_text="Net Revenue (AED)", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to plot net revenue trend in the selected scope.")

    with c2:
        st.markdown("#### Revenue by City & Channel (Paid)")
        if not {"city","channel"}.issubset(set(paid_filtered.columns)):
            st.warning("Cannot plot Revenue by City & Channel because required columns are missing in the enriched sales view.")
            grp = pd.DataFrame(columns=["city","channel","revenue"])
        else:
            grp = paid_filtered.groupby(["city", "channel"], as_index=False)["revenue"].sum()
        if len(grp):
            fig = px.bar(grp, x="city", y="revenue", color="channel", barmode="group", labels={"city":"City","revenue":"Paid Revenue (AED)","channel":"Channel"})
            fig = style_fig(fig, height=360)
            apply_bar_value_labels(fig, value_kind='aed', max_points=12)
            fig.update_yaxes(tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No paid sales in the selected scope.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Margin % by Category (Paid)")
        cat = paid_filtered.groupby("category", as_index=False).agg(revenue=("revenue", "sum"), cogs=("cogs", "sum"))
        if len(cat):
            cat["margin_pct"] = np.where(cat["revenue"] > 0, (cat["revenue"] - cat["cogs"]) / cat["revenue"] * 100, 0)
            cat = cat.sort_values("margin_pct", ascending=False)
            fig = px.bar(
                cat,
                x="category",
                y="margin_pct",
                color="category",
                color_discrete_sequence=VIBRANT_COLORWAY,
                labels={"category": "Category", "margin_pct": "Gross Margin (%)"},
            )
            fig.update_layout(showlegend=False)
            fig = style_fig(fig, height=340)
            apply_bar_value_labels(fig, value_kind='pct', max_points=15)
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to compute margin by category.")


    with c4:
        st.markdown("#### Profit Proxy vs Discount (Scenario Curve)")
        discounts = list(range(0, 31, 5))
        rows = []
        for d in discounts:
            _, cons, sk = sim.run_simulation(
                baseline_df=baseline_df,
                products_df=products_df,
                inventory_df=inventory_df,
                filters=sim_filters,
                discount_pct=float(d),
                promo_budget_aed=float(promo_budget_aed),
                margin_floor_pct=float(margin_floor_pct),
                sim_days=int(sim_days),
            )
            rows.append({"discount_pct": d, "profit_proxy": float(sk.get("profit_proxy", 0.0))})
        sens = pd.DataFrame(rows)
        fig = px.line(sens, x="discount_pct", y="profit_proxy", markers=True)
        fig.add_scatter(x=[discount_pct], y=[float(sim_kpis.get("profit_proxy", 0.0))], mode="markers", name="Selected")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("#### Decision Recommendation")
    rec = recommendation(constraints, sim_kpis, sim_kpis_base, float(discount_pct), float(promo_budget_aed))
    st.markdown(f"<div class='callout'><pre style='margin:0; white-space:pre-wrap; font-family: inherit;'>{rec}</pre></div>", unsafe_allow_html=True)

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("#### Constraint Check")
    render_constraints(constraints)

    with st.expander("Simulation output (top rows)"):
        if sim_out is not None and len(sim_out):
            st.dataframe(sim_out.head(50), use_container_width=True, height=360)
            st.download_button("Download simulation_output.csv", sim_out.to_csv(index=False).encode("utf-8"), "simulation_output.csv", "text/csv")
        else:
            st.info("Simulation output is empty for the selected scope (no baseline rows after filters).")


# -------------------------

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("#### Campaign Plan (Upcoming)")
    if len(campaign_df):
        dfc = campaign_df.copy()
        if "start_date" in dfc.columns and "end_date" in dfc.columns and dfc["start_date"].notna().any():
            today = pd.Timestamp.today().normalize()
            dfc = dfc[(dfc["end_date"] >= today) | (dfc["start_date"] >= today)]
        st.dataframe(dfc.head(50), use_container_width=True, height=260)
    else:
        st.info("No campaign plan rows available in the current dataset.")


# Manager View
# -------------------------
else:
    st.markdown("<div class='section-title'>Manager View</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtle'>Operational risk, data quality, and SKU prioritization.</div>", unsafe_allow_html=True)

    # KPI cards with deltas (risk metrics use inverse coloring: increases are worse)
    m1, m2, m3, m4 = st.columns(4)

    stockout_delta = stockout_risk_pct - (stockout_risk_pct_prev if "stockout_risk_pct_prev" in globals() else 0.0)
    high_risk_prev = int(inv_risk_prev_f[inv_risk_prev_f["risk_flag"] == 1].shape[0]) if ("inv_risk_prev_f" in globals() and len(inv_risk_prev_f)) else 0
    high_risk_delta = high_risk_skus - high_risk_prev

    m1.metric(
        "Stockout Risk (%)",
        fmt_pct(stockout_risk_pct),
        delta=(fmt_delta_pp(stockout_delta) if ("stockout_risk_pct_prev" in globals() and len(inv_risk_prev_f)) else "n/a"),
        delta_color="inverse",
    )
    m2.metric(
        "Return Rate (%)",
        fmt_pct(return_rate_pct_f),
        delta=(fmt_delta_pp(return_rate_pct_f - return_rate_pct_prev) if has_prev_period else "n/a"),
        delta_color="inverse",
    )
    m3.metric(
        "Payment Failure (%)",
        fmt_pct(payment_failure_pct_f),
        delta=(fmt_delta_pp(payment_failure_pct_f - payment_failure_pct_prev) if has_prev_period else "n/a"),
        delta_color="inverse",
    )
    m4.metric(
        "# High-Risk SKU–Store",
        f"{high_risk_skus:,}",
        delta=(f"{high_risk_delta:+,}" if ("inv_risk_prev_f" in globals() and len(inv_risk_prev_f)) else "n/a"),
        delta_color=("inverse" if high_risk_delta > 0 else "normal"),
    )

    c1, c2 = st.columns([1.05, 0.95])
    with c1:
        st.markdown("#### Stockout Risk by City & Channel")
        grp = inv_risk_f.groupby(["city", "channel"], as_index=False).agg(risk_pct=("risk_flag", "mean"), count=("risk_flag", "size"))
        if len(grp):
            grp["risk_pct"] = grp["risk_pct"] * 100
            fig = px.bar(grp, x="city", y="risk_pct", color="channel", barmode="group", hover_data=["count"], labels={"city":"City","risk_pct":"Stockout Risk (%)","channel":"Channel","count":"SKU–Store Rows"})
            fig = style_fig(fig, height=360)
            apply_bar_value_labels(fig, value_kind='number', max_points=12)
            fig.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No inventory rows in scope.")
    with c2:
        st.markdown("#### Days of Cover (Distribution)")
        if "days_of_cover" in inv_risk_f.columns and len(inv_risk_f):
            tab_a, tab_b = st.tabs(["Distribution (Risk Bands)", "Heatmap (City × Category)"])

            with tab_a:
                # Risk-band distribution keeps the chart readable and allows value labels without clutter.
                doc = inv_risk_f[["days_of_cover"]].copy()
                doc["days_of_cover"] = pd.to_numeric(doc["days_of_cover"], errors="coerce")
                doc = doc.dropna(subset=["days_of_cover"])
                if len(doc) == 0:
                    st.info("No valid days-of-cover values in scope.")
                else:
                    bins = [-np.inf, 7, 14, 30, np.inf]
                    labels = ["< 7 days (Stockout risk)", "7–14 days (Watch)", "14–30 days (Healthy)", "> 30 days (Overstock risk)"]
                    doc["doc_band"] = pd.cut(doc["days_of_cover"], bins=bins, labels=labels, right=False)
                    band = doc.groupby("doc_band", as_index=False).size().rename(columns={"size": "count"})
                    band["doc_band"] = band["doc_band"].astype(str)
                    total = float(band["count"].sum()) if len(band) else 0.0
                    band["pct"] = np.where(total > 0, band["count"] / total * 100.0, 0.0)

                    fig = px.bar(
                        band,
                        x="doc_band",
                        y="count",
                        color="doc_band",
                        text=[f"{c:,} ({p:.0f}%)" for c, p in zip(band["count"], band["pct"])],
                        color_discrete_sequence=VIBRANT_COLORWAY,
                        labels={"doc_band": "Coverage band", "count": "SKU–Store rows"},
                    )
                    fig.update_layout(showlegend=False)
                    fig = style_fig(fig, height=360)
                    fig.update_traces(textposition="outside", cliponaxis=False)
                    fig.update_xaxes(tickangle=15)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("<div class='small-note'>Coverage interpretation: low DoC = stockout risk; high DoC = overstock risk.</div>", unsafe_allow_html=True)

            with tab_b:
                # Heatmap: Median days-of-cover by City × Category for quick risk scanning.
                if {"city", "category", "days_of_cover"}.issubset(set(inv_risk_f.columns)):
                    hm = inv_risk_f[["city", "category", "days_of_cover"]].copy()
                    hm["days_of_cover"] = pd.to_numeric(hm["days_of_cover"], errors="coerce")
                    hm = hm.dropna(subset=["days_of_cover"])
                    if len(hm) == 0:
                        st.info("No valid days-of-cover values to build heatmap.")
                    else:
                        grid = hm.groupby(["city", "category"], as_index=False)["days_of_cover"].median()
                        pivot = grid.pivot(index="category", columns="city", values="days_of_cover").sort_index()
                        fig = px.imshow(
                            pivot,
                            text_auto=".0f",
                            aspect="auto",
                            color_continuous_scale="Turbo",
                            labels=dict(x="City", y="Category", color="Median DoC"),
                        )
                        fig = style_fig(fig, height=360)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Heatmap unavailable (requires city, category, and days_of_cover).")
        else:
            st.info("Days-of-cover unavailable (baseline demand missing).")

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    t1, t2 = st.columns([1.1, 0.9])
    with t1:
        st.markdown("#### Top 10 High-Risk Items (SKU–Store)")
        top = inv_risk_f[inv_risk_f["risk_flag"] == 1].sort_values("risk_score", ascending=False).head(10)
        if len(top):
            cols_show = [c for c in ["product_id","store_id","category","brand","city","channel","stock_on_hand","lead_time_days","baseline_daily_qty","days_of_cover","risk_score"] if c in top.columns]
            st.dataframe(top[cols_show], use_container_width=True, height=320)
        else:
            st.info("No high-risk items in the selected scope.")
    with t2:
        st.markdown("#### Data Quality Pareto (Issues)")
        if len(issues_df):
            pareto = (
                issues_df.groupby("issue_type", as_index=False)
                .size()
                .sort_values("size", ascending=False)
                .head(12)
                .reset_index(drop=True)
            )
            total = float(pareto["size"].sum()) if len(pareto) else 0.0
            pareto["cum"] = pareto["size"].cumsum()
            pareto["cum_pct"] = np.where(total > 0, pareto["cum"] / total * 100.0, 0.0)

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(
                    x=pareto["issue_type"],
                    y=pareto["size"],
                    name="Issue Count",
                    text=pareto["size"].astype(int).astype(str),
                    textposition="outside",
                    cliponaxis=False,
                    marker=dict(line=dict(width=0)),
                ),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=pareto["issue_type"],
                    y=pareto["cum_pct"],
                    name="Cumulative %",
                    mode="lines+markers+text",
                    text=[f"{v:.0f}%" for v in pareto["cum_pct"].values],
                    textposition="top center",
                    line=dict(width=3),
                ),
                secondary_y=True,
            )

            fig.update_layout(
                height=320,
                margin=dict(l=12, r=12, t=46, b=12),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                colorway=VIBRANT_COLORWAY,
            )
            fig.update_xaxes(tickangle=25, title_text="Issue Type")
            fig.update_yaxes(title_text="Count", secondary_y=False)
            fig.update_yaxes(title_text="Cumulative %", range=[0, 100], ticksuffix="%", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No issues logged by the cleaner.")


    with st.expander("Issues log (drill-down)"):
        if len(issues_df):
            st.dataframe(issues_df, use_container_width=True, height=360)
            st.download_button("Download issues.csv", issues_df.to_csv(index=False).encode("utf-8"), "issues.csv", "text/csv")
        else:
            st.write("No issues to display.")


# -------------------------
# Downloads
# -------------------------
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Downloads</div>", unsafe_allow_html=True)

tab_can, tab_read = st.tabs(["Canonical (pipeline)", "Readable (with units)"])

with tab_can:
    st.caption("Canonical files preserve the exact schema used by the dashboard and the schema-mapping wizard (recommended for re-upload/testing).")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.download_button("products_clean.csv", products_df.to_csv(index=False).encode("utf-8"), "products_clean.csv", "text/csv")
    c2.download_button("stores_clean.csv", stores_df.to_csv(index=False).encode("utf-8"), "stores_clean.csv", "text/csv")
    c3.download_button("sales_clean.csv", sales_df.to_csv(index=False).encode("utf-8"), "sales_clean.csv", "text/csv")
    c4.download_button("inventory_clean.csv", inventory_df.to_csv(index=False).encode("utf-8"), "inventory_clean.csv", "text/csv")
    c5.download_button("campaign_plan_clean.csv", campaign_df.to_csv(index=False).encode("utf-8"), "campaign_plan_clean.csv", "text/csv")

with tab_read:
    st.caption("Readable versions annotate key numeric/date fields with units in the header for stakeholder interpretation. The underlying values are unchanged.")

    UNITS_MAP = {
        "products": {
            "base_price_aed": "AED",
            "unit_cost_aed": "AED",
            "tax_rate": "%",
        },
        "stores": {},
        "sales_raw": {
            "order_time": "datetime",
            "qty": "units",
            "selling_price_aed": "AED",
            "discount_pct": "%",
        },
        "inventory_snapshot": {
            "snapshot_date": "date",
            "stock_on_hand": "units",
            "reorder_point": "units",
            "lead_time_days": "days",
        },
        "campaign_plan": {
            "start_date": "date",
            "end_date": "date",
            "discount_pct": "%",
            "promo_budget_aed": "AED",
        },
    }

    def annotate_units(df: pd.DataFrame, table_key: str) -> pd.DataFrame:
        unit_map = UNITS_MAP.get(table_key, {}) or {}
        rename = {c: f"{c} [{unit_map[c]}]" for c in df.columns if c in unit_map and unit_map[c]}
        return df.rename(columns=rename)

    products_u = annotate_units(products_df, "products")
    stores_u = annotate_units(stores_df, "stores")
    sales_u = annotate_units(sales_df, "sales_raw")
    inventory_u = annotate_units(inventory_df, "inventory_snapshot")
    campaign_u = annotate_units(campaign_df, "campaign_plan")

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.download_button("products_clean_with_units.csv", products_u.to_csv(index=False).encode("utf-8"), "products_clean_with_units.csv", "text/csv")
    r2.download_button("stores_clean_with_units.csv", stores_u.to_csv(index=False).encode("utf-8"), "stores_clean_with_units.csv", "text/csv")
    r3.download_button("sales_clean_with_units.csv", sales_u.to_csv(index=False).encode("utf-8"), "sales_clean_with_units.csv", "text/csv")
    r4.download_button("inventory_clean_with_units.csv", inventory_u.to_csv(index=False).encode("utf-8"), "inventory_clean_with_units.csv", "text/csv")
    r5.download_button("campaign_plan_clean_with_units.csv", campaign_u.to_csv(index=False).encode("utf-8"), "campaign_plan_clean_with_units.csv", "text/csv")

    units_rows = []
    for t, cols in UNITS_MAP.items():
        for col, unit in (cols or {}).items():
            units_rows.append({"table": t, "column": col, "unit": unit})
    units_df = pd.DataFrame(units_rows).sort_values(["table", "column"]) if units_rows else pd.DataFrame(columns=["table", "column", "unit"])
    st.download_button("units_dictionary.csv", units_df.to_csv(index=False).encode("utf-8"), "units_dictionary.csv", "text/csv")

st.caption("If Streamlit/Plotly are not in requirements.txt, add: streamlit, plotly, openpyxl (for XLSX uploads).")
