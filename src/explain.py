import numpy as np
import pandas as pd


def _values_for_class(model, frame, class_id):
    if hasattr(model, "get_booster"):
        import xgboost as xgb

        raw = model.get_booster().predict(xgb.DMatrix(frame), pred_contribs=True)
        if raw.ndim == 3:
            return raw[0, class_id, :-1]
        return raw[0, :-1]
    try:
        import shap

        raw = shap.TreeExplainer(model).shap_values(frame)
    except ImportError:
        return np.asarray(model.feature_importances_)
    if isinstance(raw, list):
        return np.asarray(raw[class_id])[0]
    raw = np.asarray(raw)
    if raw.ndim == 3:
        return raw[0, :, class_id]
    return raw[0]


def local_contributions(model, frame, class_id):
    values = _values_for_class(model, frame, class_id)
    out = pd.DataFrame(
        {
            "feature": frame.columns,
            "value": frame.iloc[0].to_numpy(),
            "shap": values,
        }
    )
    out["impact"] = out["shap"].abs()
    return out.sort_values("impact", ascending=True).reset_index(drop=True)
