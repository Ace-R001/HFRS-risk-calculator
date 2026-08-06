# Severe HFRS Risk Calculator

Web-based prototype for early prediction of **severe hemorrhagic fever with renal syndrome (HFRS)**
in patients with Hantavirus infection, using a final **random forest** model built on nine routinely
available clinical variables.

## Model summary

| Item | Value |
|---|---|
| Final model | Random forest (n_estimators = 100, max_depth = 5, min_samples_leaf = 1) |
| Fixed threshold (θ) | 0.612 (Youden index from out-of-fold predictions) |
| Development cohort | 160 patients (2015–2020), Jingzhou Central Hospital |
| Temporal validation | 48 patients (2021–2025): AUROC 0.955 (95% CI 0.885–0.998), Brier 0.119 |

## Nine predictors

CK (U/L), CK-MB (U/L), CR (μmol/L), D-dimer (ng/mL), Hypotension (yes/no), MYO (ng/mL),
NEU% (%), NPR (= NEU / PLT), 24-h urine output (mL).

> **Note:** The development cohort contained no patients with D-dimer within the normal reference
> range (≤ 232 ng/mL). Inputs outside the development training range trigger an in-app warning;
> such predictions are extrapolative and should be interpreted with caution.

## Files

- `app.py` — Streamlit application
- `final_model.pkl` — fitted random forest model + threshold (pickle)
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io (or the Streamlit dashboard), click **Create app**.
3. Select the repository, branch, and set Main file path to `app.py`.
4. Click **Deploy**. The app builds and goes live at a public `*.streamlit.app` URL.

## Disclaimer

This tool is intended for research and clinical decision support only; it does not replace
professional medical judgment.
