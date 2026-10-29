"""Separate pooled, per-material and per-publication aggregation definitions."""
import numpy as np
import pandas as pd

def regression(y, p, weights=None):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) != len(p) or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metric inputs must be finite, aligned, and nonempty")
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    if w.shape != y.shape or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("Metric weights must be finite and positive")
    mse = np.average((y - p)**2, weights=w)
    variance = np.average((y - np.average(y, weights=w))**2, weights=w)
    return {"R2": float(1 - mse / variance) if len(y) > 1 and variance > 0 else None,
            "RMSE": float(np.sqrt(mse)), "MAE": float(np.average(abs(y - p), weights=w))}

def material_weights(materials):
    s = pd.Series(np.asarray(materials).astype(str))
    return (1. / s.map(s.value_counts())).to_numpy()

def evaluate(frame):
    y, p = frame.Y_true.to_numpy(), frame.Prediction.to_numpy()
    per_material = [regression(z.Y_true, z.Prediction) for _, z in frame.groupby("Material_ID")]
    r2s = [r["R2"] for r in per_material if r["R2"] is not None]
    result = {"record_weighted": regression(y, p),
              "pooled_material_balanced": regression(y, p, material_weights(frame.Material_ID)),
              "material_macro_RMSE": float(np.mean([r["RMSE"] for r in per_material])),
              "material_macro_R2_defined_only": float(np.mean(r2s)) if r2s else None,
              "materials_without_defined_R2": len(per_material) - len(r2s),
              "records": len(frame), "materials": len(per_material)}
    if "Source_DOI" in frame:
        rows = [regression(z.Y_true, z.Prediction, material_weights(z.Material_ID)) for _, z in frame.groupby("Source_DOI")]
        defined = [r["R2"] for r in rows if r["R2"] is not None]
        result["publication_macro_material_balanced_R2_defined_only"] = float(np.mean(defined)) if defined else None
        result["publications_without_defined_R2"] = len(rows) - len(defined)
    return result

def reconstruction_metrics(targets, predictions, flags):
    """R2 per standardized descriptor on artificially masked, known targets.

    Macro R2 averages defined descriptor R2 values; undefined counts are reported.
    MSE weights each eligible cell equally. Natural missing values are excluded
    by the separate reconstruction-loss mask supplied by the dataset.
    """
    targets, predictions = np.asarray(targets,float), np.asarray(predictions,float)
    flags = np.asarray(flags,bool)
    if targets.shape != predictions.shape or targets.shape != flags.shape or targets.ndim != 2 or targets.shape[1] != 5:
        raise ValueError("Reconstruction arrays must all have shape (records, 5)")
    per_descriptor=[]
    for j in range(5):
        selected=flags[:,j]
        per_descriptor.append(regression(targets[selected,j],predictions[selected,j]) if selected.any() else None)
    defined=[s['R2'] for s in per_descriptor if s is not None and s['R2'] is not None]
    return {"R2":float(np.mean(defined)) if defined else None,
            "MSE":float(np.mean((targets[flags]-predictions[flags])**2)) if flags.any() else None,
            "eligible_cells":int(flags.sum()),"defined_descriptors":len(defined),
            "per_descriptor":per_descriptor}
