# SolarSense AI

Solar panel fault detection and power prediction, with a Streamlit dashboard.

## What it does

One solar panel is drawn with its sensors: three thermal sensors, four light sensors, a dust sensor, voltage and current sensors, an inverter temperature sensor and a weather mast.

- Press Load random readings (the fault is hidden) or move any sensor slider, and the model detects the fault: Normal, Soiling, Partial Shading, Hotspot, Inverter Fault or Degradation
- Each sensor marker on the panel turns green, amber or red, and hovering shows the fault it watches for
- A contribution card shows which sensors contributed how much to the verdict (TreeSHAP, shown as a percentage share)
- A step-by-step guide explains how to resolve the detected fault
- Mark the fault as resolved and the affected sensors return to healthy readings
- Record any set of readings with a note, view them in a log and download it as CSV

## Tech stack

| Component | Technology |
|---|---|
| Programming | Python |
| Dataset generation | NumPy + Pandas |
| Model training | Google Colab |
| Fault detection | Random Forest + XGBoost |
| Power prediction | Scikit-learn regression |
| Explainability | SHAP (TreeSHAP, computed by XGBoost) |
| Frontend + dashboard | Streamlit |
| Charts | Plotly |
| Model saving | Joblib |
| Deployment | Streamlit Community Cloud |
| Version control | GitHub |

## Project layout

```
solarsense-ai/
  app.py
  requirements.txt
  .streamlit/config.toml
  notebooks/train_on_colab.ipynb
  src/
    data_generator.py
    train.py
    predict.py
    explain.py
    ui.py
    sensors.py
```

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

A pre-trained model is included in `models/solarsense_bundle.joblib`, so the app starts straight away. To retrain it:

```
python src/train.py
```

If the model file is missing or was saved with different library versions, the app retrains on first launch (about 30 seconds).

## Train on Google Colab

Open `notebooks/train_on_colab.ipynb` in Colab, run all cells, then download `solarsense_bundle.joblib` into the `models/` folder of this repo and commit it. Keep `scikit-learn` and `xgboost` at the versions pinned in `requirements.txt` so the file loads on Streamlit Cloud.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository, including the `models/` folder.
2. Open share.streamlit.io and sign in with GitHub.
3. Choose New app, pick the repository and branch, and set the main file to `app.py`.
4. Under Advanced settings, choose Python 3.12.
5. Deploy. The first build installs the packages, which takes a couple of minutes. Later restarts are quick.

## Note

The dataset is synthetic. Swap `generate_dataset` for your plant's real telemetry before using the results for decisions.
