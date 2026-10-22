# SolarSense AI

Solar panel fault detection and power prediction, with a Streamlit dashboard.

## What it does

One solar panel is drawn with its sensors: three thermal sensors, four light sensors, a dust sensor, voltage and current sensors, an inverter temperature sensor and a weather mast.

- Change any sensor reading and the model detects the fault: Normal, Soiling, Partial Shading, Hotspot, Inverter Fault or Degradation
- Each sensor shows OK, Warning or Alert, and which fault it watches for
- SHAP shows which sensors pushed the model toward its verdict
- A step-by-step guide explains how to resolve the detected fault

## Tech stack

| Component | Technology |
|---|---|
| Programming | Python |
| Dataset generation | NumPy + Pandas |
| Model training | Google Colab |
| Fault detection | Random Forest + XGBoost |
| Power prediction | Scikit-learn regression |
| Explainability | SHAP |
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
python src/train.py
streamlit run app.py
```

If you skip the training step, the app trains the models by itself on first launch and saves them with Joblib.

## Train on Google Colab

Open `notebooks/train_on_colab.ipynb` in Colab, run all cells, then download `solarsense_bundle.joblib` into the `models/` folder of this repo and commit it. The app loads that file instead of retraining.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Open share.streamlit.io and sign in with GitHub.
3. Choose New app, pick the repository and branch, and set the main file to `app.py`.
4. Under Advanced settings, choose Python 3.11.
5. Deploy. If no trained file exists in `models/`, the first launch trains the models, which takes about a minute. After that the saved file is reused.

## Note

The dataset is synthetic. Swap `generate_dataset` for your plant's real telemetry before using the results for decisions.
