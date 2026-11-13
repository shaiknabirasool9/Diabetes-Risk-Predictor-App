# Diabetes Risk Predictor

A Streamlit demo that predicts a diabetes label from eight health measurements using a trained logistic-regression model.

## Run locally

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run models/app.py
```

The app loads the model and scaler from `models/`. The training notebook is `models/LogisticRegression.ipynb`; it expects `dataset/diabetes_logistic_regression_40000.csv` in the project root.

## Project files

- `models/app.py` — Streamlit interface
- `models/diabetes_model.pkl` and `models/scaler.pkl` — trained model and feature scaler
- `models/LogisticRegression.ipynb` — training and evaluation
- `dataset/diabetes_logistic_regression_40000.csv` — expected training/evaluation data (not included in this repository until its source and redistribution rights are confirmed)

To reproduce training, obtain the dataset from an authorized source and place it at the path above. It contains health-related measurements even though the columns shown do not include direct identifiers.

## Limitations

This is an educational demonstration, not a medical device or diagnostic tool. Predictions may be incorrect and must not be used to make healthcare decisions. The notebook's recorded test accuracy is 70.89%; accuracy alone does not establish clinical usefulness.
