# UAE Retail — Data Rescue & Promo Pulse Dashboard (Streamlit)

This project delivers a presentation-ready Streamlit dashboard for a UAE omnichannel retailer scenario, combining:
1) **Data Rescue Toolkit**: validation + cleaning pipeline that produces a structured issues log.
2) **Promo Pulse Simulator**: rule-based promo “what-if” simulation with constraints (budget, margin floor, stock limits).
3) **Interactive Dashboard**: Executive View + Manager View with KPI deltas, filters, and drill-downs.

Key constraints (per assignment): **No ML/DL**, no forecasting models. Python + Streamlit + Plotly only.

---

## Repository Structure (Recommended)

Place everything in a single repository root:

.
├── app.py # Streamlit dashboard (use the final version provided)
├── generator.py # Generates dirty datasets into /dirty_data
├── cleaner.py # Cleans dirty datasets into /clean_data + issues.csv
├── simulator_promo.py # KPI + simulation engine
├── dirty_data/
│ ├── products.csv
│ ├── stores.csv
│ ├── sales_raw.csv
│ ├── inventory_snapshot.csv
│ └── campaign_plan.csv
├── clean_data/
│ ├── products_clean.csv
│ ├── stores_clean.csv
│ ├── sales_clean.csv
│ ├── inventory_clean.csv
│ └── issues.csv
├── Strategy_Data_Generation.md # Optional supporting docs
├── Strategy_Data_Cleaning.md # Optional supporting docs
└── requirements.txt

yaml
Copy code

Important:
- Do **not** commit `env/` or `.git/` folders from a copied ZIP into a new repo.
- Ensure the folder names match exactly: `dirty_data/` and `clean_data/`.

---

## Quick Start (Local)

### 1) Create and activate a virtual environment
**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate