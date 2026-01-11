# Data Generation Strategy

## Overview
This document outlines the strategy used to generate realistic, complex, and "dirty" mock data for the retail analytics project. The goal is to simulate a real-world retail environment with messy data that requires robust cleaning pipelines.

## 1. Core Philosophy: "Master & Shadow"
To ensure that duplicates and errors are traceable and realistic (rather than purely random noise), we employ a **Master & Shadow** entity generation strategy, particularly for Stores.

### A. Store Generation
1.  **Master Stores**: We first generate a set of unique, valid stores (e.g., 12 stores) with distinct **City** (Dubai, Abu Dhabi, Sharjah) and **Channel** (App, Web, Marketplace) combinations.
2.  **Shadow Stores**: We then generate "dirty duplicates" of these masters.
    *   *Mechanism*: A shadow store copies a master store but slightly alters its metadata (e.g., casing changes like "Dubai" $\rightarrow$ "DUBAI").
    *   *Purpose*: This creates semantic duplicates that the cleaning process must identify and merge, rather than just random strings.

## 2. Sales Data Generation
Sales data is the core transactional table. We ensure logical consistency while injecting specific quality control challenges.

### A. Chronological Integrity
1.  **Time Generation**: We generate random timestamps within a defined range (e.g., Jan 2024 - Dec 2025).

### B. "Dirty" Data Injection
After generating clean base data, we purposely corrupt specific fields to test the cleaning pipeline:
*   **Timestamps**: A small percentage of dates are replaced with invalid strings (e.g., `not_a_time`) or out-of-range years (< 2020).
*   **Quantities**: Extreme outliers (e.g., 50 units) are injected using a Poisson distribution mixed with random spikes.
*   **Prices**: Some prices are excessively high (outliers) or missing (NaN).
*   **Refunds/Failures**: Realistic statuses (`Paid`, `Refunded`, `Failed`) are assigned with weighted probabilities.

## 3. Inventory Snapshots
Inventory is generated as a daily snapshot.
*   **Logic**: Each store-product combination gets a `stock_on_hand` value.
*   **Errors**:
    *   Negative stock values (logic error).
    *   Extreme stock values (e.g., 9999) representing system glitches.

## 4. Product Catalog
*   **Categories**: Products belong to fixed categories (Electronics, Fashion, etc.) but with injected casing inconsistencies ("Elec-Gadgets", "FASHION").
*   **Costing**: `unit_cost` and `base_price` are generated. We occasionally inject logic errors where `unit_cost` > `selling_price`.

## Summary of Files Generated
| File | Description | Key "Dirty" Features |
| :--- | :--- | :--- |
| `stores.csv` | Store metadata | Duplicate stores, inconsistent casing (Dubai vs DUBAI). |
| `products.csv` | Product catalog | Inconsistent category naming, missing costs. |
| `sales_raw.csv` | Transactional data | Invalid dates, unsorted IDs (post-corruption), price/qty outliers. |
| `inventory_snapshot.csv` | Stock levels | Negative stock, massive outliers. |
