# PATH-HTE: Predictive Approaches to Treatment Effect Heterogeneity

End-to-end implementation of risk-stratified Heterogeneity of Treatment Effect (HTE)
analysis on the International Stroke Trial (IST) and (later) the Digitalis
Investigation Group (DIG) trial, following the **PATH Statement** (Kent et al., 2020).

## Goals

1. Build and validate a baseline (placebo-arm) outcome risk model.
2. Stratify patients by predicted baseline risk (typically quartiles).
3. Estimate absolute and relative treatment effects within each risk stratum.
4. Quantify uncertainty (bootstrap CIs).
5. Expose results in an interactive **Streamlit** dashboard.

## Project structure

```
path-hte/
├── data/
│   ├── raw/          # Original downloaded data (gitignored)
│   └── processed/    # Cleaned, analysis-ready datasets
├── src/              # Reusable Python modules
│   ├── config.py     # Paths, constants, column groups
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── risk_model.py
│   ├── hte_analysis.py
│   └── viz.py
├── notebooks/        # Exploratory analysis (numbered 01_, 02_, ...)
├── dashboard/        # Streamlit app
├── reports/          # Figures, tables, written outputs
├── tests/            # Unit tests for the pipeline
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get the data (manual, one-time)
# Download IST_corrected.csv from:
#   https://datashare.ed.ac.uk/handle/10283/124
# Place it at: data/raw/IST_corrected.csv

# 4. Run the EDA notebook
jupyter lab notebooks/01_eda.ipynb

# 5. Launch the dashboard (after pipeline is built)
streamlit run dashboard/app.py
```

## Methodology references

- Kent DM, et al. *The Predictive Approaches to Treatment effect Heterogeneity
  (PATH) Statement.* Ann Intern Med. 2020;172:35-45.
- Sandercock PAG, Niewada M, Członkowska A. *The International Stroke Trial
  database.* Trials. 2011;12:101.
