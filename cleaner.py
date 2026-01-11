import pandas as pd
import numpy as np
import re
from datetime import datetime

# ---------------------------------------------------------
# CONFIG / CONSTANTS
# ---------------------------------------------------------
INPUT_DIR = "/Users/anish/Desktop/IP Final/dirty_data/"
OUTPUT_DIR = "/Users/anish/Desktop/IP Final/clean_data/"

# Valid Reference Values
VALID_CITIES = ["Dubai", "Abu Dhabi", "Sharjah"]
VALID_CITIES = ["Dubai", "Abu Dhabi", "Sharjah"]
VALID_CHANNELS = ["App", "Web", "Marketplace"]
VALID_CATEGORIES = ["Electronics", "Fashion", "Grocery", "Home", "Beauty"]

ISSUES_LOG = []

ISSUE_TYPE_MAP = {
    "Normalization": "DATA_NORMALIZATION",
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
    "Deduplication (Merged)": "DUPLICATE_ENTITY"
}

def log_issue(table, record_id_col, record_id_val, field, original, new, issue_type_raw):
    """Helper to log an issue in the required format."""
    
    # Map raw type to standard type
    issue_type = ISSUE_TYPE_MAP.get(issue_type_raw, issue_type_raw.upper().replace(" ", "_"))
    
    # Construct Action Taken
    if new == "DROPPED" or new == "DROPPED (Unparsable)":
        action = "Dropped Record"
    elif issue_type_raw == "Deduplication (Merged)":
        action = f"Merged into Store {new}"
    elif issue_type_raw == "Normalization":
        action = "Standardized"
    elif original is np.nan or pd.isna(original): # Check for nan properly
        action = f"Imputed with {new}"
    else:
        action = f"Corrected to {new}"

    ISSUES_LOG.append({
        "record identifier": f"{table} | {record_id_col}: {record_id_val}",
        "issue_type": issue_type,
        "issue_detail": f"Field '{field}' had value '{original}'",
        "action_taken": action
    })

# ---------------------------------------------------------
# 1. CLEAN STORES
# ---------------------------------------------------------
def clean_stores(df):
    print("--- Cleaning Stores ---")
    df_clean = df.copy()

    # Regex Patterns for Normalization
    city_patterns = {
        r'(?i)^du.*': "Dubai",
        r'(?i)^ab.*': "Abu Dhabi",
        r'(?i)^sh.*': "Sharjah"
    }
    channel_patterns = {
        r'(?i)^ap.*': "App",
        r'(?i)^we.*': "Web",
        r'(?i)^ma.*': "Marketplace"
    }

    # Clean City
    for idx, row in df_clean.iterrows():
        orig_city = str(row['city']).strip()
        new_city = orig_city
        
        # Check against patterns
        matched = False
        for pattern, valid_val in city_patterns.items():
            if re.match(pattern, orig_city):
                new_city = valid_val
                matched = True
                break
        
        if orig_city != new_city:
            log_issue("stores", "store_id", row['store_id'], "city", orig_city, new_city, "Normalization")
            df_clean.at[idx, 'city'] = new_city
        elif not matched and orig_city not in VALID_CITIES:
             # Log unknown if checks fail (though regex covers the dirty generation logic)
            log_issue("stores", "store_id", row['store_id'], "city", orig_city, "UNKNOWN", "Unknown Value")

    # Clean Channel
    for idx, row in df_clean.iterrows():
        orig_channel = str(row['channel']).strip()
        new_channel = orig_channel
        
        matched = False
        for pattern, valid_val in channel_patterns.items():
            if re.match(pattern, orig_channel):
                new_channel = valid_val
                matched = True
                break
        
        if orig_channel != new_channel:
            log_issue("stores", "store_id", row['store_id'], "channel", orig_channel, new_channel, "Normalization")
            df_clean.at[idx, 'channel'] = new_channel

    # --- DEDUPLICATION LOGIC ---
    # Group by (City, Channel) to identify duplicates
    # We keep the 'first' store_id as the canonical ID for simplicity, or generate new ones.
    # To preserve history, let's pick the MIN store_id in each group as the 'Survivor'.
    
    # Create a mapping: {old_store_id: new_store_id}
    store_mapping = {}
    
    # unique combinations
    unique_groups = df_clean.groupby(['city', 'channel'])
    
    deduped_rows = []
    
    for (city, channel), group in unique_groups:
        survivor_id = group['store_id'].min()
        
        # Add to mapping
        for old_id in group['store_id'].unique():
            store_mapping[old_id] = survivor_id
            if old_id != survivor_id:
                log_issue("stores", "store_id", old_id, "store_id", old_id, survivor_id, "Deduplication (Merged)")
        
        # Keep the survivor row (taking the first occurrence's metadata if any differs, though they shouldn't)
        survivor_row = group.loc[group['store_id'] == survivor_id].iloc[0].copy()
        deduped_rows.append(survivor_row)
        
    df_deduped = pd.DataFrame(deduped_rows)
    # Sort for neatness
    df_deduped.sort_values('store_id', inplace=True)
    
    return df_deduped, store_mapping

# ---------------------------------------------------------
# 2. CLEAN PRODUCTS
# ---------------------------------------------------------
def clean_products(df):
    print("--- Cleaning Products ---")
    df_clean = df.copy()

    # Regex Patterns for Normalization
    category_patterns = {
        r'(?i)^elec.*': "Electronics",
        r'(?i)^fash.*': "Fashion",
        r'(?i)^groc.*': "Grocery",
        r'(?i)^home.*': "Home",
        r'(?i)^beauty.*': "Beauty"
    }

    # Clean Category
    for idx, row in df_clean.iterrows():
        orig_cat = str(row['category']).strip()
        new_cat = orig_cat
        
        matched = False
        for pattern, valid_val in category_patterns.items():
            if re.match(pattern, orig_cat):
                new_cat = valid_val
                matched = True
                break
        
        if orig_cat != new_cat:
            log_issue("products", "product_id", row['product_id'], "category", orig_cat, new_cat, "Normalization")
            df_clean.at[idx, 'category'] = new_cat
        elif not matched and orig_cat not in VALID_CATEGORIES:
            log_issue("products", "product_id", row['product_id'], "category", orig_cat, "UNKNOWN", "Unknown Value")


    # 1. Handle Missing Unit Cost
    # Strategy: Group by Category, calculate average Ratio (Unit Cost / Base Price), apply to nulls.
    
    # Calculate ratio where data exists
    valid_mask = df_clean['unit_cost_aed'].notna() & (df_clean['base_price_aed'] > 0)
    temp_df = df_clean[valid_mask].copy()
    temp_df['cost_ratio'] = temp_df['unit_cost_aed'] / temp_df['base_price_aed']
    
    avg_ratios = temp_df.groupby('category')['cost_ratio'].mean().to_dict()
    
    # Fill Missing
    missing_cost_mask = df_clean['unit_cost_aed'].isna()
    for idx in df_clean[missing_cost_mask].index:
        cat = df_clean.at[idx, 'category']
        base = df_clean.at[idx, 'base_price_aed']
        ratio = avg_ratios.get(cat, 0.6) # Default to 0.6 if category unknown
        
        new_cost = round(base * ratio, 2)
        log_issue("products", "product_id", df_clean.at[idx, 'product_id'], "unit_cost_aed", np.nan, new_cost, "Imputation")
        df_clean.at[idx, 'unit_cost_aed'] = new_cost

    # 2. Validate Unit Cost <= Base Price
    # If Cost > Base Price, cap at Base Price (or use ratio? User said cap/check). 
    # Let's cap at Base Price to be safe for "break-even", or slightly below. 
    # Re-reading prompt: "Unit Cost <= Base Price ... I want your suggestion." -> Plan says Cap or Adjust.
    invalid_cost_mask = df_clean['unit_cost_aed'] > df_clean['base_price_aed']
    for idx in df_clean[invalid_cost_mask].index:
        orig = df_clean.at[idx, 'unit_cost_aed']
        limit = df_clean.at[idx, 'base_price_aed']
        log_issue("products", "product_id", df_clean.at[idx, 'product_id'], "unit_cost_aed", orig, limit, "Logic Error (Cost > Price)")
        df_clean.at[idx, 'unit_cost_aed'] = limit

    return df_clean

# ---------------------------------------------------------
# 3. CLEAN SALES
# ---------------------------------------------------------
def clean_sales(df, products_df, store_mapping):
    print("--- Cleaning Sales ---")
    df_clean = df.copy()
    
    # --- A. Timestamp Cleaning ---
    # Convert to datetime, coerce errors
    df_clean['order_time_clean'] = pd.to_datetime(df_clean['order_time'], errors='coerce')
    
    # Handle NaTs
    nat_mask = df_clean['order_time_clean'].isna()
    if nat_mask.any():
        # Drop or specific fix? "remove it" or "cap it". If it's pure garbage "not_a_time", remove.
        # But if it's just out of range, cap.
        # Here we have Parse Failures. We remove invalid strings.
        for idx in df_clean[nat_mask].index:
             log_issue("sales", "order_id", df_clean.at[idx, 'order_id'], "order_time", df_clean.at[idx, 'order_time'], "DROPPED (Unparsable)", "Parsing Error")
        df_clean = df_clean.dropna(subset=['order_time_clean'])

    # Handle Range (2020+)
    # Assume generic sensible range is recent history ~120 days. 
    # Let's verify 'current' max date
    max_date = df_clean['order_time_clean'].max()
    min_date = df_clean['order_time_clean'].quantile(0.01) # Use 1st percentile as soft lower bound to detect extreme outliers
    
    # If date < 2020 (random heuristic from prompt), fix it.
    out_of_range_mask = df_clean['order_time_clean'].dt.year < 2020
    for idx in df_clean[out_of_range_mask].index:
        orig = df_clean.at[idx, 'order_time_clean']
        log_issue("sales", "order_id", df_clean.at[idx, 'order_id'], "order_time", orig, min_date, "Date Range Outlier")
        df_clean.at[idx, 'order_time_clean'] = min_date

    # Update the main column
    df_clean['order_time'] = df_clean['order_time_clean']
    df_clean.drop(columns=['order_time_clean'], inplace=True)

    # --- Update Store IDs (Deduplication) ---
    # Apply mapping
    df_clean['store_id'] = df_clean['store_id'].map(store_mapping)
    # Check for unmapped stores (shouldn't happen if validation is good, but potential data issue)
    if df_clean['store_id'].isna().any():
        print("WARNING: Found sales records with unknown store_ids after mapping!")
        df_clean = df_clean.dropna(subset=['store_id'])
        df_clean['store_id'] = df_clean['store_id'].astype(int)

    # --- B. Duplicates ---
    # Check order_id duplicates
    dup_mask = df_clean.duplicated(subset=['order_id'], keep='first')
    if dup_mask.any():
        for idx in df_clean[dup_mask].index:
            log_issue("sales", "order_id", df_clean.at[idx, 'order_id'], "order_id", df_clean.at[idx, 'order_id'], "DROPPED", "Duplicate ID")
        df_clean = df_clean[~dup_mask]

    # --- C. IQR Outliers (Qty & Price) ---
    
    # QTY
    Q1_q = df_clean['qty'].quantile(0.25)
    Q3_q = df_clean['qty'].quantile(0.75)
    IQR_q = Q3_q - Q1_q
    upper_q = Q3_q + 1.5 * IQR_q
    
    # "outlier or missing qty, just transform those to the average." -> Prompt
    # Note: Generator uses poisson(2), outliers are 50. Mean is small.
    mean_qty = int(df_clean['qty'].mean())
    
    qty_outlier_mask = df_clean['qty'] > upper_q # Only verify upper for Qty usually
    for idx in df_clean[qty_outlier_mask].index:
        orig = df_clean.at[idx, 'qty']
        log_issue("sales", "order_id", df_clean.at[idx, 'order_id'], "qty", orig, mean_qty, "Outlier (IQR)")
        df_clean.at[idx, 'qty'] = mean_qty

    # PRICE
    # First, handle Missing Price (before IQR)
    if df_clean['selling_price_aed'].isna().any() or (df_clean['selling_price_aed'] == 0).any():
        # Merge with products to get base_price
        df_clean = df_clean.merge(products_df[['product_id', 'base_price_aed']], on='product_id', how='left')
        
        missing_price_mask = df_clean['selling_price_aed'].isna()
        for idx in df_clean[missing_price_mask].index:
            base = df_clean.at[idx, 'base_price_aed']
            log_issue("sales", "order_id", df_clean.at[idx, 'order_id'], "selling_price_aed", np.nan, base, "Missing Value")
            df_clean.at[idx, 'selling_price_aed'] = base
        
        df_clean.drop(columns=['base_price_aed'], inplace=True) # Check if merge keeps name? yes.

    # Now IQR for Price
    Q1_p = df_clean['selling_price_aed'].quantile(0.25)
    Q3_p = df_clean['selling_price_aed'].quantile(0.75)
    IQR_p = Q3_p - Q1_p
    lower_cat = Q1_p - 1.5 * IQR_p
    upper_cat = Q3_p + 1.5 * IQR_p
    
    price_high_mask = df_clean['selling_price_aed'] > upper_cat
    for idx in df_clean[price_high_mask].index:
        orig = df_clean.at[idx, 'selling_price_aed']
        log_issue("sales", "order_id", df_clean.at[idx, 'order_id'], "selling_price_aed", orig, upper_cat, "Outlier (High)")
        df_clean.at[idx, 'selling_price_aed'] = round(upper_cat, 2)

    # --- D. Missing Discount ---
    # User: "If price is missing, use base. If discount missing... ?"
    # Plan: Infer discount.
    if df_clean['discount_pct'].isna().any():
        # We need base price again to infer, or just set to 0?
        # Let's assume 0 if we can't infer easy, or fill with median. 
        # Actually safer to fill with 0 (no discount) if unknown.
        nan_disc_mask = df_clean['discount_pct'].isna()
        for idx in df_clean[nan_disc_mask].index:
            log_issue("sales", "order_id", df_clean.at[idx, 'order_id'], "discount_pct", np.nan, 0, "Missing Value")
            df_clean.at[idx, 'discount_pct'] = 0.0

    # --- FINAL SORT ---
    # Ensure chronological/ID sort
    df_clean.sort_values("order_time", inplace=True)

    return df_clean

# ---------------------------------------------------------
# 4. CLEAN INVENTORY
# ---------------------------------------------------------
def clean_inventory(df, store_mapping):
    print("--- Cleaning Inventory ---")
    df_clean = df.copy()

    # Apply Store Mapping
    df_clean['store_id'] = df_clean['store_id'].map(store_mapping)
    df_clean.dropna(subset=['store_id'], inplace=True)
    df_clean['store_id'] = df_clean['store_id'].astype(int)

    # Aggregation due to merged stores
    # If Store 14 and 15 merged, we now have two entries for Product X date Y. We should sum them.
    # But wait, inventory snapshot usually means "stock at end of day". Summing 50 + 50 = 100 might be right 
    # if they were literally two piles of stock. 
    # Given the "Shadow" nature (duplicate entry error), it's likely the SAME stock counted twice or split?
    # Actually, "Master/Shadow" implies Store 15 IS Store 14. 
    # If the system generated inventory for both, we probably have double counting.
    # However, taking the MAX or MEAN might be safer than Sum if it's a snapshot error.
    # But if they represent "Sales from App" vs "Sales from APP", the inventory might be partitioned?
    # Let's assume SUM is safer to allow reconciliation, or MEAN?
    # Let's stick to standard dedupe: sum them up implies we found stock in both "locations" (schema wise).
    
    # We will groupby and sum 'stock_on_hand'
    # We should also aggregate reorder_point (maybe max?) and lead_time (max?).
    
    df_clean = df_clean.groupby(['snapshot_date', 'store_id', 'product_id'], as_index=False).agg({
        'stock_on_hand': 'sum',
        'reorder_point': 'max',
        'lead_time_days': 'max'
    })

    # Negative Stock
    neg_mask = df_clean['stock_on_hand'] < 0
    for idx in df_clean[neg_mask].index:
        orig = df_clean.at[idx, 'stock_on_hand']
        log_issue("inventory", "idx", idx, "stock_on_hand", orig, 0, "Logic Error (Negative)")
        df_clean.at[idx, 'stock_on_hand'] = 0
        
    # Unreasonable High Stock (Outlier 9999)
    # Using simple quantile check or specific value check if generated known '9999'
    # Plan said "replace with median"
    median_stock = df_clean['stock_on_hand'].median()
    # Let's say anything > 5000 is error (generator used 9999)
    high_mask = df_clean['stock_on_hand'] > 5000
    for idx in df_clean[high_mask].index:
        orig = df_clean.at[idx, 'stock_on_hand']
        log_issue("inventory", "idx", idx, "stock_on_hand", orig, median_stock, "Outlier (Extreme)")
        df_clean.at[idx, 'stock_on_hand'] = int(median_stock)

    return df_clean

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    print("Loading datasets...")
    try:
        stores = pd.read_csv(INPUT_DIR + "stores.csv")
        products = pd.read_csv(INPUT_DIR + "products.csv")
        sales = pd.read_csv(INPUT_DIR + "sales_raw.csv")
        inventory = pd.read_csv(INPUT_DIR + "inventory_snapshot.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Process
    # 1. Stores (returns deduped df AND mapping)
    clean_stores_df, store_mapping = clean_stores(stores)
    print(f"Store Mapping Created: {len(store_mapping)} original IDs -> {clean_stores_df['store_id'].nunique()} unique IDs")
    
    # 2. Products
    clean_products_df = clean_products(products)
    
    # 3. Sales (needs mapping)
    clean_sales_df = clean_sales(sales, clean_products_df, store_mapping) 
    
    # 4. Inventory (needs mapping)
    clean_inventory_df = clean_inventory(inventory, store_mapping)

    # Save Clean Files
    print("Saving cleaned datasets...")
    clean_stores_df.to_csv(OUTPUT_DIR + "stores_clean.csv", index=False)
    clean_products_df.to_csv(OUTPUT_DIR + "products_clean.csv", index=False)
    clean_sales_df.to_csv(OUTPUT_DIR + "sales_clean.csv", index=False)
    clean_inventory_df.to_csv(OUTPUT_DIR + "inventory_clean.csv", index=False)

    # Save Issues Log
    if ISSUES_LOG:
        issues_df = pd.DataFrame(ISSUES_LOG)
        issues_df.to_csv(OUTPUT_DIR + "issues.csv", index=False)
        print(f"Issues Log created: issues.csv with {len(issues_df)} records.")
    else:
        print("No issues found (clean run).")

    print("Cleaning Complete.")

if __name__ == "__main__":
    main()
