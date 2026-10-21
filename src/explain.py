import numpy as np
import pandas as pd
import shap


def make_explainer(model):
    return shap.TreeExplainer(model)


def _class_slice(values, class_id):
    if isinstance(values, list):
        return np.asarray(values[class_id])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, class_id]
    return values


def local_contributions(explainer, frame, class_id):
    values = _class_slice(explainer.shap_values(frame), class_id)
    out = pd.DataFrame(
        {
            "feature": frame.columns,
            "value": frame.iloc[0].to_numpy(),
            "shap": values[0],
        }
    )
    out["impact"] = out["shap"].abs()
    return out.sort_values("impact", ascending=True).reset_index(drop=True)


def global_importance(explainer, background):
    raw = explainer.shap_values(background)
    if isinstance(raw, list):
        stacked = np.stack([np.abs(np.asarray(v)) for v in raw], axis=0)
        score = stacked.mean(axis=(0, 1))
    else:
        raw = np.asarray(raw)
        if raw.ndim == 3:
            score = np.abs(raw).mean(axis=(0, 2))
        else:
            score = np.abs(raw).mean(axis=0)
    out = pd.DataFrame({"feature": background.columns, "importance": score})
    return out.sort_values("importance", ascending=True).reset_index(drop=True)
