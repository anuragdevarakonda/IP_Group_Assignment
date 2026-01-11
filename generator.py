# generator.py
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# -----------------------------
# CONFIG
# -----------------------------
N_PRODUCTS = 300
N_STORES = 18
N_ORDERS = 30000
DAYS_HISTORY = 120
INVENTORY_DAYS = 30

OUTPUT_PATH = "/Users/anish/Desktop/IP Final/dirty_data/"

# -----------------------------
# HELPERS
# -----------------------------
def random_date(start, end):
    return start + timedelta(
        seconds=random.randint(0, int((end - start).total_seconds()))
    )

# -----------------------------
# 1. PRODUCTS TABLE
# -----------------------------
categories_clean = ["Electronics", "Fashion", "Grocery", "Home", "Beauty"]
dict_category_dirty = {
    "Electronics": ["ELECTRONICS", "electronics", "Elec-Gadgets"],
    "Fashion": ["FASHION", "fashion", "Fashion-Wear"],
    "Grocery": ["GROCERY", "grocery", "Groceries"],
    "Home": ["HOME", "home", "Home-Decor"],
    "Beauty": ["BEAUTY", "beauty", "Beauty-Cosmetics"]
}

brands = [f"Brand_{i}" for i in range(1, 41)]

products = []
for pid in range(1, N_PRODUCTS + 1):
    base_price = round(np.random.uniform(10, 800), 2)
    unit_cost = round(base_price * np.random.uniform(0.5, 0.8), 2)
    products.append({
        "product_id": pid,
        "category": random.choice([categories_clean[(pid - 1) % 5]] + dict_category_dirty[categories_clean[(pid - 1) % 5]]), # Random dirty variant

        "brand": random.choice(brands),
        "base_price_aed": base_price,
        "unit_cost_aed": unit_cost,
        "tax_rate": random.choice([0.0, 0.05]),
        "launch_flag": random.choice(["New", "Regular"])
    })

products_df = pd.DataFrame(products)
# Inject missing unit_cost (1–2%)
mask = products_df.sample(frac=0.015).index
products_df.loc[mask, "unit_cost_aed"] = np.nan

# -----------------------------
# 2. STORES TABLE
# -----------------------------
# Expanded city and channel variants for realism
cities_clean = ["Dubai", "Abu Dhabi", "Sharjah"]
dict_city_dirty = {
    "Dubai": ["DUBAI", "dubai", "Dubayy"],
    "Abu Dhabi": ["ABU DHABI", "abu dhabi", "Abu-Dhabi"],
    "Sharjah": ["SHARJAH", "sharjah", "Sharjah-Emirate"]
}

# Re-construct flattened list
cities_dirty = []
for k, v in dict_city_dirty.items():
    cities_dirty.append(k)
    cities_dirty.extend(v)

channels_clean = ["App", "Web", "Marketplace"]
dict_channel_dirty = {
    "App": ["APP", "app"],
    "Web": ["WEB", "web"],
    "Marketplace": ["MARKETPLACE", "marketplace"]
}

# Re-construct flattened lists for compatibility with Campaign/Inventory sections
channels = []
for k, v in dict_channel_dirty.items():
    channels.append(k)
    channels.extend(v)

fulfillment = ["Own", "3PL"]

# Phase 1: Generate Master Unique Stores
# We want to ensure we have a good spread of unique City/Channel combos first
unique_combinations = []
for c in cities_clean:
    for ch in channels_clean:
        unique_combinations.append({"city": c, "channel": ch})

random.shuffle(unique_combinations)

# We'll use the first N_UNIQUE stores as "Master" records (clean or slightly dirty but unique intent)
# Let's say we want 12 unique masters, and the rest (N_STORES - 12) are duplicates
N_UNIQUE = 12
master_stores = []

stores = []
sid = 1

# Create Master Stores
for i in range(min(N_UNIQUE, len(unique_combinations))):
    combo = unique_combinations[i]
    # Master stores are created "clean" or "standard" first
    # We can randomly dirty them a bit here if we want, but let's keep them recognizable
    # Actually, let's allow the *Master* to have a specific casing, and the *Duplicate* to have a different one
    
    # For the master entry, we pick one cleaning variant (e.g. title case) to be the "truth" logic
    # But to match the "dirty source" vibe, even the unique ones can be messy, 
    # as long as they are UNIQUE combos.
    
    city_variant = random.choice([combo["city"]] + dict_city_dirty[combo["city"]])
    channel_variant = random.choice([combo["channel"]] + dict_channel_dirty[combo["channel"]])
    
    store_obj = {
        "store_id": sid,
        "city": city_variant,
        "channel": channel_variant,
        "fulfillment_type": random.choice(fulfillment),
        "true_city": combo["city"],      # hidden field for logic if needed
        "true_channel": combo["channel"] # hidden field
    }
    stores.append(store_obj)
    master_stores.append(store_obj)
    sid += 1

# Phase 2: Create Dirty Duplicates
# The remaining stores are strictly derived from the existing master stores
while sid <= N_STORES:
    # Pick a random parent from the master list
    parent = random.choice(master_stores)
    
    # Create a dirty variation
    # 1. Maybe change casing of city
    # 2. Maybe change casing of channel
    # 3. Fulfillment might be different (data entry error?) or same
    
    # Get a DIFFERENT variant if possible, or same
    city_options = [parent["true_city"]] + dict_city_dirty[parent["true_city"]]
    # Try to pick one that isn't exactly the current one, to force "dirty", but random is fine
    city_variant = random.choice(city_options)
    
    channel_options = [parent["true_channel"]] + dict_channel_dirty[parent["true_channel"]]
    channel_variant = random.choice(channel_options)
    
    stores.append({
        "store_id": sid,
        "city": city_variant,
        "channel": channel_variant,
        "fulfillment_type": random.choice(fulfillment), # random fulfillment for the dupe
        "true_city": parent["true_city"],
        "true_channel": parent["true_channel"]
    })
    sid += 1

stores_df = pd.DataFrame(stores).drop(columns=["true_city", "true_channel"])

# -----------------------------
# 3. SALES_RAW TABLE (DIRTY)
# -----------------------------
start_date = datetime.now() - timedelta(days=DAYS_HISTORY)
end_date = datetime.now()

sales = []
for oid in range(1, N_ORDERS + 1):
    product = products_df.sample(1).iloc[0]
    store = stores_df.sample(1).iloc[0]
    qty = np.random.poisson(2)
    discount = round(np.random.choice([0, 5, 10, 15, 20]), 2)
    selling_price = round(product.base_price_aed * (1 - discount / 100), 2)
    sales.append({
        "order_id": oid,
        "order_time": random_date(start_date, end_date).strftime("%Y-%m-%d %H:%M:%S"),
        "product_id": product.product_id,
        "store_id": store.store_id,
        "qty": qty,
        "selling_price_aed": selling_price,
        "discount_pct": discount,
        "payment_status": random.choice(["Paid", "Failed", "Refunded"]),
        "return_flag": random.choice([0, 1])
    })

sales_df = pd.DataFrame(sales)
# Ensure chronological order (ascending)
sales_df["order_time"] = pd.to_datetime(sales_df["order_time"])
sales_df.sort_values("order_time", inplace=True)
sales_df["order_time"] = sales_df["order_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
# Inject missing discount_pct (2–4%)
sales_df.loc[sales_df.sample(frac=0.03).index, "discount_pct"] = np.nan
# Duplicate order_id (0.5–1%)
dupes = sales_df.sample(frac=0.008)
sales_df = pd.concat([sales_df, dupes], ignore_index=True)
# Corrupted timestamps (1–2%)
bad_time_idx = sales_df.sample(frac=0.015).index
sales_df.loc[bad_time_idx, "order_time"] = random.choice([
    "not_a_time", "2024-13-40", "invalid_date"
])
# Outliers (qty = 50 or price ×10)
outlier_idx = sales_df.sample(frac=0.004).index
sales_df.loc[outlier_idx[:len(outlier_idx)//2], "qty"] = 50
sales_df.loc[outlier_idx[len(outlier_idx)//2:], "selling_price_aed"] *= 10

# -----------------------------
# 4. INVENTORY SNAPSHOT
# -----------------------------
inventory = []
for day in range(INVENTORY_DAYS):
    snapshot_date = (datetime.now() - timedelta(days=day)).date()
    for _ in range(50):  # fixed number of random combos per day
        product = products_df.sample(1).iloc[0]
        store = stores_df.sample(1).iloc[0]
        stock = np.random.randint(0, 500)
        inventory.append({
            "snapshot_date": snapshot_date,
            "product_id": product.product_id,
            "store_id": store.store_id,
            "stock_on_hand": stock,
            "reorder_point": np.random.randint(20, 80),
            "lead_time_days": np.random.randint(2, 14)
        })

inventory_df = pd.DataFrame(inventory)
# Impossible inventory values
bad_inv_idx = inventory_df.sample(frac=0.01).index
inventory_df.loc[bad_inv_idx[:len(bad_inv_idx)//2], "stock_on_hand"] = -5
inventory_df.loc[bad_inv_idx[len(bad_inv_idx)//2:], "stock_on_hand"] = 9999

# -----------------------------
# 5. CAMPAIGN PLAN (OPTIONAL)
# -----------------------------
campaigns = []
for i in range(1, 11):
    campaigns.append({
        "campaign_id": i,
        "start_date": datetime.now().date(),
        "end_date": (datetime.now() + timedelta(days=14)).date(),
        "city": random.choice(["All", "Dubai", "Abu Dhabi", "Sharjah"]),
        "channel": random.choice(["All"] + channels),
        "category": random.choice(["All"] + categories_clean),
        "discount_pct": random.choice([5, 10, 15, 20]),
        "promo_budget_aed": random.randint(200000, 800000)
    })

campaign_df = pd.DataFrame(campaigns)

# -----------------------------
# SAVE FILES
# -----------------------------
products_df.to_csv(OUTPUT_PATH + "products.csv", index=False)
stores_df.to_csv(OUTPUT_PATH + "stores.csv", index=False)
sales_df.to_csv(OUTPUT_PATH + "sales_raw.csv", index=False)
inventory_df.to_csv(OUTPUT_PATH + "inventory_snapshot.csv", index=False)
campaign_df.to_csv(OUTPUT_PATH + "campaign_plan.csv", index=False)

print("✅ Dirty datasets generated successfully.")
