# generator.py
"""Synthetic *dirty* data generator for the UAE Retail dashboard.

Design goals:
- Deterministic with a seed
- Parameterizable sizes (products/stores/orders/days)
- Can either (a) return DataFrames in-memory (recommended for Streamlit) or (b) write CSVs.

Expected output keys (used by app.py cleaning pipeline):
- products, stores, sales_raw, inventory_snapshot, campaign_plan
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class GenConfig:
    seed: int = 42
    n_products: int = 300
    n_stores: int = 18
    n_orders: int = 30000
    days_history: int = 120
    inventory_days: int = 30
    # Optional extension set (e.g., ["Sports"])
    extra_categories: Optional[List[str]] = None


def _random_date(start: datetime, end: datetime) -> datetime:
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))


def generate_dirty_data(
    *,
    seed: int = 42,
    n_products: int = 300,
    n_stores: int = 18,
    n_orders: int = 30000,
    days_history: int = 120,
    inventory_days: int = 30,
    extra_categories: Optional[List[str]] = None,
    write_csv: bool = False,
    output_path: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Generate dirty datasets.

    If write_csv=True, CSVs are written to output_path (default: ./dirty_data/).

    Returns:
        dict of DataFrames with keys:
        products, stores, sales_raw, inventory_snapshot, campaign_plan
    """

    # Seeds
    np.random.seed(seed)
    random.seed(seed)

    # -----------------------------
    # 1) PRODUCTS TABLE
    # -----------------------------
    base_categories = ["Electronics", "Fashion", "Grocery", "Home", "Beauty"]
    if extra_categories:
        # Keep title-case canonical category values; cleaner will normalize variants anyway.
        for c in extra_categories:
            c_norm = str(c).strip().title()
            if c_norm and c_norm not in base_categories:
                base_categories.append(c_norm)

    dict_category_dirty = {
        "Electronics": ["ELECTRONICS", "electronics", "Elec-Gadgets"],
        "Fashion": ["FASHION", "fashion", "Fashion-Wear"],
        "Grocery": ["GROCERY", "grocery", "Groceries"],
        "Home": ["HOME", "home", "Home-Decor"],
        "Beauty": ["BEAUTY", "beauty", "Beauty-Cosmetics"],
    }

    # Add generic dirty variants for any extra categories (e.g., Sports)
    for c in base_categories:
        if c not in dict_category_dirty:
            dict_category_dirty[c] = [c.upper(), c.lower(), f"{c} - Misc", f" {c}  "]

    brands = [f"Brand_{i}" for i in range(1, 41)]

    products = []
    for pid in range(1, n_products + 1):
        base_price = round(np.random.uniform(10, 800), 2)
        unit_cost = round(base_price * np.random.uniform(0.5, 0.8), 2)
        cat_truth = base_categories[(pid - 1) % max(1, len(base_categories))]
        cat_dirty = random.choice([cat_truth] + dict_category_dirty.get(cat_truth, []))
        products.append(
            {
                "product_id": pid,
                "category": cat_dirty,
                "brand": random.choice(brands),
                "base_price_aed": base_price,
                "unit_cost_aed": unit_cost,
                "tax_rate": random.choice([0.0, 0.05]),
                "launch_flag": random.choice(["New", "Regular"]),
            }
        )

    products_df = pd.DataFrame(products)
    # Inject missing unit_cost (≈1–2%)
    mask = products_df.sample(frac=0.015, random_state=seed).index
    products_df.loc[mask, "unit_cost_aed"] = np.nan

    # -----------------------------
    # 2) STORES TABLE
    # -----------------------------
    cities_clean = ["Dubai", "Abu Dhabi", "Sharjah"]
    dict_city_dirty = {
        "Dubai": ["DUBAI", "dubai", "Dubayy"],
        "Abu Dhabi": ["ABU DHABI", "abu dhabi", "Abu-Dhabi"],
        "Sharjah": ["SHARJAH", "sharjah", "Sharjah-Emirate"],
    }

    channels_clean = ["App", "Web", "Marketplace"]
    dict_channel_dirty = {
        "App": ["APP", "app"],
        "Web": ["WEB", "web"],
        "Marketplace": ["MARKETPLACE", "marketplace"],
    }

    # Flatten for use in campaign section
    channels_all: List[str] = []
    for k, v in dict_channel_dirty.items():
        channels_all.append(k)
        channels_all.extend(v)

    fulfillment = ["Own", "3PL"]

    # Create unique City×Channel combinations first (masters), then create duplicates with dirty variants
    unique_combinations = [{"city": c, "channel": ch} for c in cities_clean for ch in channels_clean]
    random.shuffle(unique_combinations)

    n_unique = min(12, len(unique_combinations))
    stores = []
    sid = 1
    master_stores = []

    # Master stores
    for i in range(n_unique):
        combo = unique_combinations[i]
        city_variant = random.choice([combo["city"]] + dict_city_dirty[combo["city"]])
        channel_variant = random.choice([combo["channel"]] + dict_channel_dirty[combo["channel"]])
        store_obj = {
            "store_id": sid,
            "city": city_variant,
            "channel": channel_variant,
            "fulfillment_type": random.choice(fulfillment),
            "true_city": combo["city"],
            "true_channel": combo["channel"],
        }
        stores.append(store_obj)
        master_stores.append(store_obj)
        sid += 1

    # Dirty duplicates
    while sid <= n_stores:
        parent = random.choice(master_stores)
        city_variant = random.choice([parent["true_city"]] + dict_city_dirty[parent["true_city"]])
        channel_variant = random.choice([parent["true_channel"]] + dict_channel_dirty[parent["true_channel"]])
        stores.append(
            {
                "store_id": sid,
                "city": city_variant,
                "channel": channel_variant,
                "fulfillment_type": random.choice(fulfillment),
                "true_city": parent["true_city"],
                "true_channel": parent["true_channel"],
            }
        )
        sid += 1

    stores_df = pd.DataFrame(stores).drop(columns=["true_city", "true_channel"])

    # -----------------------------
    # 3) SALES_RAW TABLE (DIRTY)
    # -----------------------------
    start_date = datetime.now() - timedelta(days=days_history)
    end_date = datetime.now()

    sales = []
    # Vectorize sampling ids for speed
    product_ids = products_df["product_id"].to_numpy()
    store_ids = stores_df["store_id"].to_numpy()

    for oid in range(1, n_orders + 1):
        pid = int(np.random.choice(product_ids))
        sid = int(np.random.choice(store_ids))
        product = products_df.loc[products_df["product_id"] == pid].iloc[0]

        qty = int(np.random.poisson(2))
        discount = float(np.random.choice([0, 5, 10, 15, 20]))
        selling_price = round(float(product.base_price_aed) * (1 - discount / 100.0), 2)

        sales.append(
            {
                "order_id": oid,
                "order_time": _random_date(start_date, end_date).strftime("%Y-%m-%d %H:%M:%S"),
                "product_id": pid,
                "store_id": sid,
                "qty": qty,
                "selling_price_aed": selling_price,
                "discount_pct": discount,
                "payment_status": random.choice(["Paid", "Failed", "Refunded"]),
                "return_flag": random.choice([0, 1]),
            }
        )

    sales_df = pd.DataFrame(sales)
    # Ensure chronological order
    sales_df["order_time"] = pd.to_datetime(sales_df["order_time"], errors="coerce")
    sales_df.sort_values("order_time", inplace=True)
    sales_df["order_time"] = sales_df["order_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Inject missing discount_pct (≈2–4%)
    sales_df.loc[sales_df.sample(frac=0.03, random_state=seed).index, "discount_pct"] = np.nan
    # Duplicate order_id (≈0.5–1%)
    dupes = sales_df.sample(frac=0.008, random_state=seed + 1)
    sales_df = pd.concat([sales_df, dupes], ignore_index=True)
    # Corrupted timestamps (≈1–2%)
    bad_time_idx = sales_df.sample(frac=0.015, random_state=seed + 2).index
    sales_df.loc[bad_time_idx, "order_time"] = random.choice(["not_a_time", "2024-13-40", "invalid_date"])
    # Outliers (qty=50 OR price×10)
    outlier_idx = sales_df.sample(frac=0.004, random_state=seed + 3).index
    half = len(outlier_idx) // 2
    sales_df.loc[outlier_idx[:half], "qty"] = 50
    sales_df.loc[outlier_idx[half:], "selling_price_aed"] = sales_df.loc[outlier_idx[half:], "selling_price_aed"] * 10

    # -----------------------------
    # 4) INVENTORY SNAPSHOT
    # -----------------------------
    inventory = []
    for day in range(inventory_days):
        snapshot_date = (datetime.now() - timedelta(days=day)).date()
        for _ in range(50):  # fixed random combos per day
            pid = int(np.random.choice(product_ids))
            sid = int(np.random.choice(store_ids))
            stock = int(np.random.randint(0, 500))
            inventory.append(
                {
                    "snapshot_date": snapshot_date,
                    "product_id": pid,
                    "store_id": sid,
                    "stock_on_hand": stock,
                    "reorder_point": int(np.random.randint(20, 80)),
                    "lead_time_days": int(np.random.randint(2, 14)),
                }
            )

    inventory_df = pd.DataFrame(inventory)
    # Impossible inventory values
    bad_inv_idx = inventory_df.sample(frac=0.01, random_state=seed + 4).index
    half = len(bad_inv_idx) // 2
    inventory_df.loc[bad_inv_idx[:half], "stock_on_hand"] = -5
    inventory_df.loc[bad_inv_idx[half:], "stock_on_hand"] = 9999

    # -----------------------------
    # 5) CAMPAIGN PLAN
    # -----------------------------
    campaigns = []
    for i in range(1, 11):
        campaigns.append(
            {
                "campaign_id": i,
                "start_date": datetime.now().date(),
                "end_date": (datetime.now() + timedelta(days=14)).date(),
                "city": random.choice(["All", "Dubai", "Abu Dhabi", "Sharjah"]),
                "channel": random.choice(["All"] + channels_all),
                "category": random.choice(["All"] + base_categories),
                "discount_pct": random.choice([5, 10, 15, 20]),
                "promo_budget_aed": int(random.randint(200000, 800000)),
            }
        )
    campaign_df = pd.DataFrame(campaigns)

    out = {
        "products": products_df,
        "stores": stores_df,
        "sales_raw": sales_df,
        "inventory_snapshot": inventory_df,
        "campaign_plan": campaign_df,
    }

    if write_csv:
        out_dir = output_path or os.getenv("OUTPUT_PATH", "dirty_data/")
        os.makedirs(out_dir, exist_ok=True)
        if not out_dir.endswith(os.sep):
            out_dir = out_dir + os.sep
        products_df.to_csv(out_dir + "products.csv", index=False)
        stores_df.to_csv(out_dir + "stores.csv", index=False)
        sales_df.to_csv(out_dir + "sales_raw.csv", index=False)
        inventory_df.to_csv(out_dir + "inventory_snapshot.csv", index=False)
        campaign_df.to_csv(out_dir + "campaign_plan.csv", index=False)

    return out


if __name__ == "__main__":
    # Support env overrides (useful if you run: OUTPUT_PATH=/tmp python generator.py)
    cfg = GenConfig(
        seed=int(os.getenv("SEED", "42")),
        n_products=int(os.getenv("N_PRODUCTS", "300")),
        n_stores=int(os.getenv("N_STORES", "18")),
        n_orders=int(os.getenv("N_ORDERS", "30000")),
        days_history=int(os.getenv("DAYS_HISTORY", "120")),
        inventory_days=int(os.getenv("INVENTORY_DAYS", "30")),
        extra_categories=None,
    )
    generate_dirty_data(
        seed=cfg.seed,
        n_products=cfg.n_products,
        n_stores=cfg.n_stores,
        n_orders=cfg.n_orders,
        days_history=cfg.days_history,
        inventory_days=cfg.inventory_days,
        extra_categories=cfg.extra_categories,
        write_csv=True,
        output_path=os.getenv("OUTPUT_PATH", "dirty_data/"),
    )
    print("✅ Dirty datasets generated successfully.")
