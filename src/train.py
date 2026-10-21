import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_generator import CLF_FEATURES, FAULTS, REG_FEATURES, generate_dataset

MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
BUNDLE_PATH = MODEL_DIR / "solarsense_bundle.joblib"


def train_all(n_samples=12000, seed=42, save=True):
    df = generate_dataset(n_samples, seed)

    X = df[CLF_FEATURES]
    y = df["fault_id"]
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.2, stratify=y, random_state=seed
    )

    candidates = {
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=14, min_samples_leaf=2, n_jobs=-1, random_state=seed
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=seed,
        ),
    }

    clf_scores = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        clf_scores[name] = {
            "accuracy": float(accuracy_score(y_te, pred)),
            "f1_macro": float(f1_score(y_te, pred, average="macro")),
        }
        fitted[name] = model

    best_clf_name = max(clf_scores, key=lambda k: clf_scores[k]["f1_macro"])
    best_clf = fitted[best_clf_name]
    best_pred = best_clf.predict(X_te)
    cm = confusion_matrix(y_te, best_pred, labels=list(range(len(FAULTS))))

    Xr = df[REG_FEATURES]
    yr = df["power_kw"]
    Xr_tr = Xr.loc[idx_tr]
    Xr_te = Xr.loc[idx_te]
    yr_tr = yr.loc[idx_tr]
    yr_te = yr.loc[idx_te]

    reg_candidates = {
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=seed
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.08, random_state=seed
        ),
    }

    reg_scores = {}
    reg_fitted = {}
    for name, model in reg_candidates.items():
        model.fit(Xr_tr, yr_tr)
        pred = model.predict(Xr_te)
        reg_scores[name] = {
            "r2": float(r2_score(yr_te, pred)),
            "mae": float(mean_absolute_error(yr_te, pred)),
            "rmse": float(np.sqrt(mean_squared_error(yr_te, pred))),
        }
        reg_fitted[name] = model

    best_reg_name = max(reg_scores, key=lambda k: reg_scores[k]["r2"])
    best_reg = reg_fitted[best_reg_name]
    reg_pred = best_reg.predict(Xr_te)

    bundle = {
        "clf": best_clf,
        "clf_name": best_clf_name,
        "clf_scores": clf_scores,
        "reg": best_reg,
        "reg_name": best_reg_name,
        "reg_scores": reg_scores,
        "classes": FAULTS,
        "clf_features": CLF_FEATURES,
        "reg_features": REG_FEATURES,
        "confusion": cm,
        "reg_test": {
            "actual": yr_te.to_numpy()[:600],
            "predicted": reg_pred[:600],
        },
        "background": X_te.sample(300, random_state=seed).reset_index(drop=True),
        "n_samples": int(n_samples),
    }

    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)
        joblib.dump(bundle, BUNDLE_PATH, compress=3)
        df.to_csv(DATA_DIR / "solar_panel_data.csv", index=False)

    return bundle


def load_bundle():
    return joblib.load(BUNDLE_PATH)


def ensure_bundle():
    if BUNDLE_PATH.exists():
        try:
            return load_bundle()
        except Exception:
            pass
    return train_all()


if __name__ == "__main__":
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    result = train_all(samples)
    print("Best classifier:", result["clf_name"], result["clf_scores"][result["clf_name"]])
    print("Best regressor:", result["reg_name"], result["reg_scores"][result["reg_name"]])
    print("Saved to", BUNDLE_PATH)
