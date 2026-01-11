import pandas as pd
import os
import numpy as np
import re
from datetime import datetime, timedelta

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
INPUT_DIR = os.getenv("INPUT_DIR", "clean_data/")# ---------------------------------------------------------
# UPLIFT LOGIC CONSTANTS
# ---------------------------------------------------------
# Base uplift multipliers by discount tier
DISCOUNT_TIERS = {
    (0, 5): 1.05,      # 0-5% discount → 5% demand increase
    (5, 10): 1.15,     # 5-10% discount → 15% demand increase
    (10, 15): 1.30,    # 10-15% discount → 30% demand increase
    (15, 20): 1.50,    # 15-20% discount → 50% demand increase
    (20, 100): 1.75,   # 20%+ discount → 75% demand increase
}

# Channel sensitivity (multiplier on base uplift)
CHANNEL_SENSITIVITY = {
    "App": 1.0,           # Baseline
    "Web": 0.9,           # Slightly less responsive
    "Marketplace": 1.2,   # Most price-sensitive
}

# Category sensitivity (multiplier on base uplift)
CATEGORY_SENSITIVITY = {
    "Electronics": 1.3,   # High ticket, price-sensitive
    "Fashion": 1.1,       # Moderate sensitivity
    "Grocery": 0.8,       # Low elasticity
    "Home": 1.0,          # Baseline
    "Beauty": 1.05,       # Slightly above baseline
}


DEFAULT_CATEGORY_SENSITIVITY = 1.0  # Used for any new/unseen category

# ---------------------------------------------------------
# HELPERS (ROBUST TO OPEN-VOCAB CATEGORIES / EXTERNAL ENCODINGS)
# ---------------------------------------------------------
def normalize_free_text_category(x):
    """Open-vocabulary category normalization for consistent grouping/filters."""
    if pd.isna(x):
        return "Other"
    s = str(x).strip()
    # Handle artifacts like "sports/SPORTS/Sports"
    if "/" in s:
        s = s.split("/")[0].strip()
    s = re.sub(r"\s+", " ", s)
    return s.title()

def _boolish_to_int(series: pd.Series) -> pd.Series:
    """Convert boolean-like encodings to 0/1 numeric for safe KPI arithmetic."""
    if series is None:
        return series
    num = pd.to_numeric(series, errors="coerce")
    if len(series) and float(num.notna().mean()) >= 0.85:
        return num.fillna(0).clip(0, 1)

    s = series.astype(str).str.strip().str.lower()
    true_set = {"1","true","t","yes","y","returned","return","r"}
    false_set = {"0","false","f","no","n","not returned","not_returned","nr","nan","","none"}
    out = s.map(lambda v: 1 if v in true_set else (0 if v in false_set else np.nan))
    return pd.to_numeric(out, errors="coerce").fillna(0).clip(0, 1)

def _coalesce_column(df: pd.DataFrame, base: str, candidates: tuple) -> pd.DataFrame:
    """Ensure df[base] exists by taking the first available candidate column."""
    if base in df.columns:
        return df
    for c in candidates:
        if c in df.columns:
            df[base] = df[c]
            return df
    df[base] = np.nan
    return df

# ---------------------------------------------------------
# LOAD CLEAN DATA
# ---------------------------------------------------------
def load_clean_data():
    """Load all cleaned datasets"""
    try:
        products = pd.read_csv(INPUT_DIR + "products_clean.csv")
        stores = pd.read_csv(INPUT_DIR + "stores_clean.csv")
        sales = pd.read_csv(INPUT_DIR + "sales_clean.csv")
        inventory = pd.read_csv(INPUT_DIR + "inventory_clean.csv")
        
        # Convert dates
        sales['order_time'] = pd.to_datetime(sales['order_time'])
        inventory['snapshot_date'] = pd.to_datetime(inventory['snapshot_date'])
        
        return products, stores, sales, inventory
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

# ---------------------------------------------------------
# HISTORICAL KPI CALCULATIONS
# ---------------------------------------------------------
def calculate_historical_kpis(sales_df, products_df):
    """
    Calculate 14 historical KPIs from cleaned sales data
    Returns a dictionary of KPIs
    """
    # Merge with products to get cost data
    sales_enriched = sales_df.merge(
        products_df[['product_id', 'unit_cost_aed', 'category']], 
        on='product_id', 
        how='left'
    )

    # Normalize return_flag to numeric (robust to external encodings)
    if 'return_flag' in sales_enriched.columns:
        sales_enriched['return_flag'] = _boolish_to_int(sales_enriched['return_flag'])
    else:
        sales_enriched['return_flag'] = 0

    
    # Filter for paid transactions only (for revenue calculations)
    paid_sales = sales_enriched[sales_enriched['payment_status'] == 'Paid'].copy()
    
    # Calculate derived fields
    paid_sales['revenue'] = paid_sales['selling_price_aed'] * paid_sales['qty']
    paid_sales['cogs'] = paid_sales['unit_cost_aed'] * paid_sales['qty']
    paid_sales['gross_margin_aed'] = paid_sales['revenue'] - paid_sales['cogs']
    
    # Refunded transactions
    refunded_sales = sales_enriched[sales_enriched['payment_status'] == 'Refunded'].copy()
    refunded_sales['refund_amount'] = refunded_sales['selling_price_aed'] * refunded_sales['qty']
    
    kpis = {}
    
    # 1. Gross Revenue (Paid only)
    kpis['gross_revenue'] = paid_sales['revenue'].sum()
    
    # 2. Refund Amount
    kpis['refund_amount'] = refunded_sales['refund_amount'].sum() if len(refunded_sales) > 0 else 0
    
    # 3. Net Revenue
    kpis['net_revenue'] = kpis['gross_revenue'] - kpis['refund_amount']
    
    # 4. COGS
    kpis['cogs'] = paid_sales['cogs'].sum()
    
    # 5. Gross Margin (AED)
    kpis['gross_margin_aed'] = paid_sales['gross_margin_aed'].sum()
    
    # 6. Gross Margin %
    kpis['gross_margin_pct'] = (kpis['gross_margin_aed'] / kpis['gross_revenue'] * 100) if kpis['gross_revenue'] > 0 else 0
    
    # 7. Average Discount %
    kpis['avg_discount_pct'] = paid_sales['discount_pct'].mean()
    
    # 8. Total Orders
    kpis['total_orders'] = len(paid_sales)
    
    # 9. Average Order Value
    kpis['avg_order_value'] = kpis['gross_revenue'] / kpis['total_orders'] if kpis['total_orders'] > 0 else 0
    
    # 10. Return Rate %
    total_orders_all_status = len(sales_enriched)
    returned_orders = sales_enriched['return_flag'].sum()
    kpis['return_rate_pct'] = (returned_orders / total_orders_all_status * 100) if total_orders_all_status > 0 else 0
    
    # 11. Payment Failure Rate %
    failed_payments = len(sales_enriched[sales_enriched['payment_status'] == 'Failed'])
    kpis['payment_failure_pct'] = (failed_payments / total_orders_all_status * 100) if total_orders_all_status > 0 else 0
    
    # 12. Units Sold
    kpis['units_sold'] = paid_sales['qty'].sum()
    
    # 13. Revenue per Unit
    kpis['revenue_per_unit'] = kpis['gross_revenue'] / kpis['units_sold'] if kpis['units_sold'] > 0 else 0
    
    # 14. Unique Products Sold
    kpis['unique_products'] = paid_sales['product_id'].nunique()
    
    return kpis, sales_enriched

# ---------------------------------------------------------
# BASELINE DEMAND CALCULATION
# ---------------------------------------------------------
def calculate_baseline_demand(sales_df, products_df, stores_df, lookback_days=30):
    """
    Calculate baseline daily demand per product-store combination
    from recent sales history (last N days)
    
    Returns: DataFrame with columns [product_id, store_id, category, channel, baseline_daily_qty]
    """
    # Get recent sales only
    cutoff_date = sales_df['order_time'].max() - timedelta(days=lookback_days)
    recent_sales = sales_df[
        (sales_df['order_time'] >= cutoff_date) & 
        (sales_df['payment_status'] == 'Paid')
    ].copy()
    
    # Merge with products and stores to get category and channel
    # Note: external sales extracts may already contain 'category'/'city'/'channel'; merges can create *_x/*_y.
    # Use suffixes + coalescing to guarantee canonical columns exist.

    recent_sales = recent_sales.merge(
        products_df[['product_id', 'category']],
        on='product_id',
        how='left',
        suffixes=('', '_prod')
    )
    recent_sales = _coalesce_column(recent_sales, 'category', ('category', 'category_prod', 'category_x', 'category_y'))
    recent_sales['category'] = recent_sales['category'].apply(normalize_free_text_category)

    recent_sales = recent_sales.merge(
        stores_df[['store_id', 'channel', 'city']],
        on='store_id',
        how='left',
        suffixes=('', '_store')
    )
    recent_sales = _coalesce_column(recent_sales, 'city', ('city', 'city_store', 'city_x', 'city_y'))
    recent_sales = _coalesce_column(recent_sales, 'channel', ('channel', 'channel_store', 'channel_x', 'channel_y'))

    # Light-touch standardization for display/grouping
    recent_sales['city'] = recent_sales['city'].astype(str).str.strip().str.title()
    recent_sales['channel'] = recent_sales['channel'].astype(str).str.strip().str.title()
# Calculate total qty per product-store over the period
    # Ensure required dims exist (defensive)
    for _c in ['category','channel','city']:
        if _c not in recent_sales.columns:
            recent_sales[_c] = np.nan

    baseline = recent_sales.groupby(
        ['product_id', 'store_id', 'category', 'channel', 'city'], 
        as_index=False
    )['qty'].sum()
    
    # Convert to daily average
    baseline['baseline_daily_qty'] = baseline['qty'] / lookback_days
    baseline.drop(columns=['qty'], inplace=True)
    
    # Handle products/stores with no recent sales (set baseline to 0.1 to allow small promo effect)
    baseline['baseline_daily_qty'] = baseline['baseline_daily_qty'].fillna(0.1)
    
    return baseline

# ---------------------------------------------------------
# DEMAND UPLIFT CALCULATOR
# ---------------------------------------------------------
def calculate_demand_uplift(discount_pct, channel, category):
    """
    Calculate demand uplift multiplier based on discount, channel, and category
    
    Args:
        discount_pct: Promotional discount percentage
        channel: Sales channel (App/Web/Marketplace)
        category: Product category
    
    Returns:
        float: Uplift multiplier (e.g., 1.5 = 50% increase)
    """
    # Get base uplift from discount tier
    base_uplift = 1.0
    for (min_d, max_d), uplift in DISCOUNT_TIERS.items():
        if min_d <= discount_pct < max_d:
            base_uplift = uplift
            break
    
    # Apply channel sensitivity
    channel_norm = str(channel).strip().title() if not pd.isna(channel) else channel
    channel_factor = CHANNEL_SENSITIVITY.get(channel_norm, 1.0)
    
    # Apply category sensitivity (open vocabulary)
    category_norm = normalize_free_text_category(category)
    category_factor = CATEGORY_SENSITIVITY.get(category_norm, DEFAULT_CATEGORY_SENSITIVITY)
    
    # Combined uplift
    total_uplift = base_uplift * channel_factor * category_factor
    
    return total_uplift

# ---------------------------------------------------------
# SIMULATION ENGINE
# ---------------------------------------------------------
def run_simulation(
    baseline_df, 
    products_df, 
    inventory_df,
    filters,
    discount_pct,
    promo_budget_aed,
    margin_floor_pct,
    sim_days=14
):
    """
    Run promotional simulation with constraints
    
    Args:
        baseline_df: Baseline demand DataFrame
        products_df: Products master data
        inventory_df: Current inventory snapshot
        filters: Dict with keys 'city', 'channel', 'category' (can be 'All')
        discount_pct: Promotional discount %
        promo_budget_aed: Maximum promotional budget
        margin_floor_pct: Minimum acceptable gross margin %
        sim_days: Simulation window in days
    
    Returns:
        simulation_results: DataFrame with projected metrics
        constraints_status: Dict with constraint check results
    """
    # Apply filters to baseline
    sim_baseline = baseline_df.copy()
    
    if filters['city'] != 'All':
        sim_baseline = sim_baseline[sim_baseline['city'] == filters['city']]
    
    if filters['channel'] != 'All':
        sim_baseline = sim_baseline[sim_baseline['channel'] == filters['channel']]
    
    if filters['category'] != 'All':
        sim_baseline = sim_baseline[sim_baseline['category'] == filters['category']]
    
    if len(sim_baseline) == 0:
        return pd.DataFrame(), {
            'budget_ok': True, 
            'margin_ok': True, 
            'stock_ok': True,
            'violations': []
        }
    
    # Merge with products to get pricing
    sim_baseline = sim_baseline.merge(
        products_df[['product_id', 'base_price_aed', 'unit_cost_aed']], 
        on='product_id', 
        how='left'
    )
    
    # Get latest inventory snapshot
    latest_inventory = inventory_df.groupby(
        ['product_id', 'store_id'], 
        as_index=False
    )['stock_on_hand'].last()
    
    sim_baseline = sim_baseline.merge(
        latest_inventory,
        on=['product_id', 'store_id'],
        how='left'
    )
    sim_baseline['stock_on_hand'] = sim_baseline['stock_on_hand'].fillna(0)
    
    # Calculate uplift for each row
    sim_baseline['uplift_factor'] = sim_baseline.apply(
        lambda row: calculate_demand_uplift(
            discount_pct, 
            row['channel'], 
            row['category']
        ),
        axis=1
    )
    
    # Projected demand over simulation window
    sim_baseline['projected_daily_qty'] = sim_baseline['baseline_daily_qty'] * sim_baseline['uplift_factor']
    sim_baseline['projected_total_qty'] = sim_baseline['projected_daily_qty'] * sim_days
    
    # Cap by available stock
    sim_baseline['simulated_qty'] = sim_baseline[['projected_total_qty', 'stock_on_hand']].min(axis=1)
    
    # Financial calculations
    sim_baseline['promo_price'] = sim_baseline['base_price_aed'] * (1 - discount_pct / 100)
    sim_baseline['promo_revenue'] = sim_baseline['simulated_qty'] * sim_baseline['promo_price']
    sim_baseline['promo_cogs'] = sim_baseline['simulated_qty'] * sim_baseline['unit_cost_aed']
    sim_baseline['promo_margin_aed'] = sim_baseline['promo_revenue'] - sim_baseline['promo_cogs']
    sim_baseline['promo_margin_pct'] = (
        sim_baseline['promo_margin_aed'] / sim_baseline['promo_revenue'] * 100
    ).fillna(0)
    
    # Promotional spend (discount amount)
    sim_baseline['discount_amount'] = sim_baseline['simulated_qty'] * sim_baseline['base_price_aed'] * (discount_pct / 100)
    
    # Constraint checks
    total_promo_spend = sim_baseline['discount_amount'].sum()
    avg_margin_pct = (sim_baseline['promo_margin_aed'].sum() / sim_baseline['promo_revenue'].sum() * 100) if sim_baseline['promo_revenue'].sum() > 0 else 0
    
    # Stockout risk
    sim_baseline['stockout_risk'] = (sim_baseline['projected_total_qty'] > sim_baseline['stock_on_hand']).astype(int)
    sim_baseline['unmet_demand'] = (sim_baseline['projected_total_qty'] - sim_baseline['stock_on_hand']).clip(lower=0)
    
    # Constraint status
    constraints = {
        'budget_ok': total_promo_spend <= promo_budget_aed,
        'budget_used': total_promo_spend,
        'budget_limit': promo_budget_aed,
        'budget_util_pct': (total_promo_spend / promo_budget_aed * 100) if promo_budget_aed > 0 else 0,
        
        'margin_ok': avg_margin_pct >= margin_floor_pct,
        'margin_achieved': avg_margin_pct,
        'margin_floor': margin_floor_pct,
        
        'stock_ok': sim_baseline['stockout_risk'].sum() == 0,
        'stockout_count': sim_baseline['stockout_risk'].sum(),
        'stockout_risk_pct': (sim_baseline['stockout_risk'].sum() / len(sim_baseline) * 100) if len(sim_baseline) > 0 else 0,
        
        'violations': []
    }
    
    # Top violators
    if not constraints['budget_ok']:
        top_budget = sim_baseline.nlargest(10, 'discount_amount')[
            ['product_id', 'store_id', 'category', 'channel', 'discount_amount']
        ]
        constraints['violations'].append({
            'type': 'BUDGET_EXCEEDED',
            'top_contributors': top_budget.to_dict('records')
        })
    
    if not constraints['margin_ok']:
        low_margin = sim_baseline[sim_baseline['promo_margin_pct'] < margin_floor_pct].nsmallest(10, 'promo_margin_pct')[
            ['product_id', 'store_id', 'category', 'promo_margin_pct']
        ]
        constraints['violations'].append({
            'type': 'MARGIN_BELOW_FLOOR',
            'top_contributors': low_margin.to_dict('records')
        })
    
    if not constraints['stock_ok']:
        top_stockout = sim_baseline[sim_baseline['stockout_risk'] == 1].nlargest(10, 'unmet_demand')[
            ['product_id', 'store_id', 'category', 'channel', 'stock_on_hand', 'projected_total_qty', 'unmet_demand']
        ]
        constraints['violations'].append({
            'type': 'STOCKOUT_RISK',
            'top_contributors': top_stockout.to_dict('records')
        })
    
    # Simulation KPIs
    sim_kpis = {
        'promo_revenue': sim_baseline['promo_revenue'].sum(),
        'promo_cogs': sim_baseline['promo_cogs'].sum(),
        'promo_margin_aed': sim_baseline['promo_margin_aed'].sum(),
        'promo_margin_pct': avg_margin_pct,
        'promo_spend': total_promo_spend,
        'profit_proxy': sim_baseline['promo_margin_aed'].sum() - total_promo_spend,  # Margin minus promo cost
        'units_projected': sim_baseline['simulated_qty'].sum(),
        'stockout_risk_pct': constraints['stockout_risk_pct'],
        'high_risk_skus': constraints['stockout_count']
    }
    
    return sim_baseline, constraints, sim_kpis

# ---------------------------------------------------------
# MAIN EXECUTION (FOR TESTING)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Loading clean data...")
    products, stores, sales, inventory = load_clean_data()
    
    if products is None:
        print("Failed to load data. Exiting.")
        exit(1)
    
    print("\n--- Historical KPI Calculation ---")
    kpis, sales_enriched = calculate_historical_kpis(sales, products)
    
    print("\nHistorical KPIs:")
    for key, value in kpis.items():
        if 'pct' in key or 'rate' in key:
            print(f"  {key}: {value:.2f}%")
        elif 'aed' in key or 'revenue' in key or 'margin' in key or 'cogs' in key or 'refund' in key:
            print(f"  {key}: AED {value:,.2f}")
        else:
            print(f"  {key}: {value:,.0f}")
    
    print("\n--- Baseline Demand Calculation ---")
    baseline = calculate_baseline_demand(sales, products, stores, lookback_days=30)
    print(f"Baseline calculated for {len(baseline)} product-store combinations")
    print(f"Average daily demand per SKU: {baseline['baseline_daily_qty'].mean():.2f} units")
    
    print("\n--- Running Sample Simulation ---")
    filters = {
        'city': 'All',
        'channel': 'All',
        'category': 'Electronics'
    }
    
    sim_results, constraints, sim_kpis = run_simulation(
        baseline_df=baseline,
        products_df=products,
        inventory_df=inventory,
        filters=filters,
        discount_pct=15,
        promo_budget_aed=500000,
        margin_floor_pct=20,
        sim_days=14
    )
    
    print("\nSimulation KPIs:")
    for key, value in sim_kpis.items():
        if 'pct' in key:
            print(f"  {key}: {value:.2f}%")
        elif 'aed' in key or 'revenue' in key or 'margin' in key or 'spend' in key or 'profit' in key:
            print(f"  {key}: AED {value:,.2f}")
        else:
            print(f"  {key}: {value:,.0f}")
    
    print("\nConstraint Status:")
    print(f"  Budget OK: {constraints['budget_ok']} (Used: AED {constraints['budget_used']:,.2f} / {constraints['budget_limit']:,.2f})")
    print(f"  Margin OK: {constraints['margin_ok']} (Achieved: {constraints['margin_achieved']:.2f}% / Floor: {constraints['margin_floor']:.2f}%)")
    print(f"  Stock OK: {constraints['stock_ok']} (Risk Items: {constraints['stockout_count']})")
    
    if constraints['violations']:
        print(f"\n⚠️  {len(constraints['violations'])} constraint violation(s) detected")
        for v in constraints['violations']:
            print(f"    - {v['type']}")
    
    print("\n✅ Simulator test complete.")
