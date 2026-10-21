# SolarSense AI

Solar panel fault detection and power prediction, with a Streamlit dashboard.

## What it does

- Array monitor: change the sensor readings and every panel in a 40-panel array is scanned automatically; faulty panels light up on a live map
- Detects six conditions: Normal, Soiling, Partial Shading, Hotspot, Inverter Fault, Degradation
- Predicts the power a panel should produce from weather and panel condition
- Explains every verdict with SHAP
- Shows a fleet dashboard of 2,500 simulated panels with pickable chart axes and a CSV export
- Has a what-if explorer that sweeps any reading and shows how the diagnosis shifts

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

The Model lab page has a Retrain button that rebuilds the dataset and models from inside the deployed app.

## Note

The dataset is synthetic. Swap `generate_dataset` for your plant's real telemetry before using the results for decisions.
