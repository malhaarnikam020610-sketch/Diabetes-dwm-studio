# Diabetes Data Warehouse & Data Mining (DWM) Studio

This project upgrades the original diabetes prediction DBMS/ML app into a coursework-ready Data Warehousing and Data Mining project while retaining the original prediction app and model files.

## Features
- ETL: extracts the supplied CSV, removes duplicate rows, derives age/glucose/BMI bands, and loads SQLite tables.
- Star schema: one fact table (`fact_patient`) plus five dimensions (`dim_patient_profile`, `dim_age`, `dim_glucose`, `dim_bmi`, `dim_outcome`).
- OLAP: roll-up, drill-down, slice, dice, pivot/crosstab and measure aggregation, with SQL examples.
- Data mining: Decision Tree and Random Forest classification; accuracy, precision, recall, F1 and confusion matrix; K-means clustering; correlation matrix and feature importance.
- SQL Explorer: read-only single SELECT queries against the warehouse.
- Streamlit dashboard and a Windows launch script.

## Run (Windows)
1. Extract the ZIP.
2. Install Python 3.10 or newer.
3. Double-click `run_windows.bat`, or open a terminal in the folder and run:
   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run dwm_app.py
   ```
4. Open the local URL printed in the terminal (usually `http://localhost:8501`).

The SQLite warehouse `diabetes_warehouse.db` is generated automatically on first run. Use **Run ETL / rebuild warehouse** in the sidebar to recreate it from the CSV.

## Original application
The original `app.py`, `model_training.py`, `data_generation.py`, trained model, dataset, and plots are retained. To run the original prediction dashboard instead, use `python -m streamlit run app.py`.

## Suggested demonstration
1. Overview & ETL: show extraction, transformation and load.
2. Warehouse Schema: explain the fact table grain, dimensions, keys and measures.
3. OLAP Queries: demonstrate roll-up, drill-down, slice, dice and pivot.
4. Data Mining: compare classification models and inspect clustering.
5. SQL Explorer: execute the sample group-by query.

## Disclaimer
For academic demonstration only. Dataset labels and model predictions are not a medical diagnosis or validated clinical advice.
