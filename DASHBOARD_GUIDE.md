# UAE Retail – Data Rescue & Promo Pulse  
Dashboard Guide (Team Reference)

## 1) What this dashboard is for
This Streamlit dashboard is designed to help professors and stakeholders evaluate an e-commerce business scenario using **five operational datasets** and a **promotion simulator**. It supports two primary objectives:

1. **Data rescue / governance:** demonstrate how noisy, inconsistent operational data is cleaned, validated, and made analysis-ready (with an auditable issues log).
2. **Decision support:** simulate promotional strategies under realistic constraints (budget, margin floor, stockout risk) and recommend actions.

The dashboard has two role-based perspectives:
- **Executive View:** financial performance, trend clarity, and promo decisioning.
- **Manager View:** operational risk, inventory coverage, and data quality.

---

## 2) Data model (canonical tables)
The dashboard operates on **canonical versions** of the following tables (uploaded as separate CSVs):

1. **products.csv**
   - Purpose: product master data used for category/brand rollups and margin calculations.
   - Typical fields: `product_id`, `category`, `brand`, `base_price_aed`, `unit_cost_aed`, …

2. **stores.csv**
   - Purpose: store/channel dimension used for rollups and filtering.
   - Typical fields: `store_id`, `city`, `channel`, …

3. **sales_raw.csv**
   - Purpose: transactional order lines used for revenue, margin, returns, payment failure, and demand.
   - Typical fields: `order_id`, `order_time`, `store_id`, `product_id`, `qty`, `selling_price_aed`, `discount_pct`, `payment_status`, `return_flag`, `tax_rate`, `fulfillment_type`, …

4. **inventory_snapshot.csv**
   - Purpose: on-hand stock and replenishment metadata used for risk and days-of-cover.
   - Typical fields: `snapshot_date`, `store_id`, `product_id`, `stock_on_hand`, `reorder_point`, `lead_time_days`, …

5. **campaign_plan.csv**
   - Purpose: planned/upcoming campaigns (used for “what’s coming” context).
   - Typical fields: campaign dates, targeted category/channel/city, planned discounts/budgets, etc.

### Canonicalization and schema mapping
Because external datasets can vary in column naming and formats, the app includes a **mapping wizard**:

- **Step 1 — Table assignment:** assign each uploaded CSV to the correct canonical table.
- **Step 2 — Column mapping:** map uploaded columns → canonical column names via dropdowns.

Once canonical tables are built, the dashboard caches them so that switching views/filters does **not** require rebuilding unless the dataset changes.

---

## 3) Data preparation and quality strategy
The data pipeline is designed to be **auditable and resilient**:

### 3.1 Cleaning principles
- **Normalization:** standardizes common inconsistencies (case, whitespace, synonyms).
- **Type safety:** coerces numeric fields (prices, qty, cost, stock) and timestamps.
- **Deduplication / mapping:** resolves store duplicates and ensures downstream joins succeed.
- **Imputation:** fills missing values using defensible heuristics (e.g., category median/ratio).
- **Outlier handling:** caps or replaces extreme values (e.g., implausible stock).

### 3.2 Open-vocabulary categories
Categories are treated as **open vocabulary**:
- New categories (e.g., “Sports”, “SPORTS”, “ Sports”, hidden whitespace) are normalized into a single canonical label (e.g., “Sports”).
- This prevents charts and filters from splitting the same category into multiple groups.

### 3.3 Issues log (governance)
All repairs are recorded in an **issues log** (table name, field, original value, action taken).  
This supports “data rescue” storytelling: not only *what* was fixed, but *why*.

---

## 4) How to use the dashboard (recommended flow)
1. **Upload the 5 CSVs** (or point the app to repo files if using fixed paths).
2. Run the **mapping wizard** (only needed when columns differ from canonical schema).
3. Click **Build canonical tables** once.
4. Use:
   - **Dashboard View** selector (Executive vs Manager),
   - **Filters** (city/channel/category/brand/fulfillment/date range),
   - **Simulation Controls** (discount, budget, margin floor, and any scenario knobs).
5. Interpret outputs:
   - Start with KPI deltas,
   - validate data quality,
   - review constraints and recommendation,
   - iterate simulation controls until feasible and attractive.

---

## 5) Global interpretation rules (applies everywhere)
### 5.1 “Paid” focus
Most financial visuals use **Paid orders** to reflect realized revenue.

### 5.2 Tax handling
KPIs are **pre-tax** by design (tax is informational/context only).  
The dashboard may display an average tax rate for context without adding tax into revenue.

### 5.3 Deltas and arrows
KPI cards show **change vs prior period** (a matching-length window immediately preceding the selected date range).  
Interpretation:
- Up arrow / positive delta: improvement (financial KPIs).
- For risk metrics (stockouts/returns/payment failure), higher is worse; the dashboard uses “inverse” delta coloring.

---

## 6) Executive View — visuals explained
The Executive View answers: **How is the business performing and what promo strategy should we run?**

### 6.1 KPI cards (top row)
- **Net Revenue (Filtered):** Net revenue for the current filter scope and date range.
- **Gross Margin % (Filtered):** Margin ratio for the same scope.
- **Profit Proxy (Scenario):** Simulator output summarizing expected profitability under current promo settings.
- **Budget Utilization (Scenario):** Estimated promo spend as a % of the promo budget.

Use the deltas to quickly judge whether performance is trending up/down versus the prior period.

### 6.2 Net Revenue Trend (Monthly) + Rolling Averages
A monthly time series of net revenue, plus:
- **Rolling 3-month average:** short-term smoothing for trend visibility.
- **Rolling 6-month average:** medium-term smoothing for strategic direction.

How to read:
- If the raw monthly bars fluctuate, the rolling averages reveal the underlying direction.
- Crossing trends can indicate inflection points (e.g., improvement beginning mid-period).

### 6.3 Revenue by City & Channel (Paid)
Shows where revenue is coming from:
- Compare cities and channels to identify the dominant route-to-market.
- Large gaps can signal channel dependency risk or growth opportunity.

### 6.4 Margin % by Category (Paid)
Shows profitability by category:
- High-margin categories are safer targets for deeper discounts.
- Low-margin categories require cautious promo planning (margin floor constraint risk).

### 6.5 Profit Proxy vs Discount (Scenario Curve)
Shows how simulated profitability changes across discount levels:
- Helps identify diminishing returns (higher discounts do not always increase profit).
- Use it to select a discount range before finalizing a scenario.

### 6.6 Decision Recommendation (dynamic)
A concise decision summary that updates with the simulation:
- **PROCEED:** constraints satisfied and profit proxy is not deteriorating versus baseline.
- **ADJUST:** one constraint failing, or profit proxy declines even if constraints pass.
- **DO NOT PROCEED:** multiple constraints failing or the scenario is clearly infeasible.

The recommendation is driven by:
- Constraint results (budget/margin/stock),
- Profit proxy change vs baseline,
- Budget utilization,
- Stockout risk.

### 6.7 Constraint Check (Budget / Margin / Stock)
A feasibility gate for the scenario:
- **Budget:** promo spend must remain within the promo budget.
- **Margin:** simulated promo margin must remain above the margin floor.
- **Stock:** stockout-risk should remain acceptable.

Each tile shows:
- PASS/FAIL,
- actuals, thresholds, and deltas,
- actionable suggestions (e.g., reduce discount, narrow scope, exclude low-margin categories, focus on high-cover SKUs).

If violations exist, “Top contributors” highlights what is causing the failure.

---

## 7) Manager View — visuals explained
The Manager View answers: **Where are the operational risks and which items need attention?**

### 7.1 KPI cards (top row)
- **Stockout Risk (%):** portion of store–SKU rows projected to face stock pressure.
- **Return Rate (%):** operational quality signal (and revenue leakage).
- **Payment Failure (%):** conversion/checkout reliability indicator.
- **# High-Risk SKU–Store:** count of items in the highest risk band (based on inventory coverage).

### 7.2 Stockout Risk by City & Channel
A segmentation of risk:
- Identify where demand/stock mismatch is concentrated.
- Helps direct replenishment effort, service-level adjustments, or targeted promos.

### 7.3 Days of Cover — Distribution (Risk Bands)
Days of Cover (DoC) estimates how long inventory will last:

\[
\text{Days of Cover} = \frac{\text{Stock on Hand}}{\text{Baseline Daily Demand}}
\]

The distribution is banded for clarity:
- **< 7 days:** stockout risk
- **7–14 days:** watch list
- **14–30 days:** healthy
- **> 30 days:** overstock risk

Counts and % labels are shown directly on the bars for presentation readability.

### 7.4 Days of Cover — Heatmap (City × Category)
A heatmap of **median DoC** by city and category:
- Quickly highlights where coverage is dangerously low or excessively high.
- Useful for cross-functional discussions (category management + supply chain + city ops).

### 7.5 Top 10 High-Risk Items (SKU–Store)
A prioritized list of the most at-risk store–SKU combinations:
- Use this as an action list for replenishment, reorder point review, or promo exclusion rules.

### 7.6 Data Quality Pareto (Issues)
A Pareto chart of cleaning issues:
- **Bars:** issue counts by type
- **Line:** cumulative percentage (dual axis)

How to read:
- The few issue types responsible for most problems appear early and drive the cumulative line upward quickly.
- Use this to focus remediation (e.g., if 80% of issues are from 2 types, fix those upstream).

Labels are displayed on the cumulative line to avoid hover dependency during presentations.

### 7.7 Campaign Plan (Upcoming)
A forward-looking context view:
- Helps align simulation and operational readiness with what is planned.

---

## 8) Simulation strategy (what it is modeling)
The simulator is intentionally lightweight and explainable for an academic setting.

### 8.1 Baseline demand
- Baseline demand is derived from **recent historical sales** within the filtered scope.
- Demand is aggregated at a granular level (typically store–product plus segmentation fields such as city/channel/category).
- The result is a **baseline daily demand** estimate used for:
  - projected sales under promotion,
  - days-of-cover calculations,
  - stockout-risk detection.

### 8.2 Promotion mechanics
- A discount is applied as a scenario parameter.
- The simulator estimates uplift (increased demand) based on discount magnitude (simplified elasticity-style logic).
- Promo spend is approximated from the “value given away” via discounting, compared against budget.

### 8.3 Profit proxy
Profit proxy is an interpretable measure (not GAAP profit):
- It combines revenue, unit cost, and promotional spend impacts in a consistent way to compare scenarios.
- It is used for **relative scenario ranking** rather than exact accounting.

### 8.4 Constraints
The simulator checks whether the scenario is feasible:
- **Budget constraint:** promo spend must remain within budget.
- **Margin floor:** promo margin must stay above the floor.
- **Stockout risk:** scenario should not create excessive inventory risk.

### 8.5 Decisioning approach
A scenario is recommended when:
- constraints pass, and
- profit proxy improves (or does not meaningfully deteriorate) vs baseline.

If not, the dashboard suggests adjustments:
- reduce discount,
- narrow scope,
- exclude low-margin categories,
- focus on high-cover SKUs,
- remove low-stock store–SKU pairs from promo eligibility.

---

## 9) Presentation narrative (suggested talk track)
A simple structure for presenting in class:
1. **Data rescue:** show issues Pareto + explain governance logging.
2. **Business health:** Net Revenue (monthly) + rolling averages.
3. **Where money comes from:** Revenue by City & Channel.
4. **Where profit comes from:** Margin % by Category.
5. **Operational feasibility:** Days of Cover distribution + heatmap.
6. **Promo decision:** show constraint check + recommendation, iterate one control to show dynamic behavior.

---

## 10) Team operating notes
- Rebuild canonical tables **only** when the dataset changes.
- Filters and view switching should be instantaneous (cached canonical tables).
- New categories are supported automatically; any split categories usually indicate missing normalization upstream.

---

*End of guide.*
