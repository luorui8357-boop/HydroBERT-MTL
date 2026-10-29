"""Explicit aggregation for publication-balanced and repeated-CV reports."""
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import evaluate, regression
from .data import write_json

R2_DEFINITIONS = ("record_weighted", "pooled_material_balanced", "material_macro_R2_defined_only")

def validate_predictions(frame):
    required = {"Record_ID", "Material_ID", "Model", "Mask_rate", "Y_true", "Prediction"}
    if not required <= set(frame):
        raise ValueError(f"Missing prediction columns: {sorted(required-set(frame))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing prediction values or identifiers")
    if not np.isfinite(frame[["Y_true", "Prediction", "Mask_rate"]].to_numpy(float)).all():
        raise ValueError("Nonfinite predictions")
    if frame.duplicated(["Model", "Mask_rate", "Record_ID"]).any():
        raise ValueError("Each record must be predicted once per model/rate; separate repetitions first")
    reference = None
    for _, part in frame.groupby(["Model", "Mask_rate"]):
        ids = set(part.Record_ID)
        if reference is None:
            reference = ids
        elif reference != ids:
            raise ValueError("Models and rates must evaluate exactly the same record IDs")
    if frame.groupby("Record_ID").Y_true.nunique().max() != 1:
        raise ValueError("Targets disagree between models/rates")
    if frame.groupby("Record_ID").Material_ID.nunique().max() != 1:
        raise ValueError("Material identities disagree")

def lopo_tables(frame, r2_definition, expected_publications=12):
    validate_predictions(frame)
    if r2_definition not in R2_DEFINITIONS:
        raise ValueError("An explicit within-publication R2 definition is required")
    if "Source_DOI" not in frame or frame.Source_DOI.isna().any():
        raise ValueError("Source_DOI is required")
    if frame.groupby("Record_ID").Source_DOI.nunique().max() != 1:
        raise ValueError("Publication identities disagree")
    if frame.groupby("Material_ID").Source_DOI.nunique().max() != 1:
        raise ValueError("Resolve materials shared between publications before LOPO reporting")
    if frame.Source_DOI.nunique() != expected_publications:
        raise ValueError(f"Expected all {expected_publications} publications")
    if "Outer_split" in frame:
        for _, z in frame.groupby(["Model", "Mask_rate"]):
            if z.groupby("Outer_split").Source_DOI.nunique().max() != 1 or z.groupby("Source_DOI").Outer_split.nunique().max() != 1:
                raise ValueError("A LOPO outer fold must correspond to exactly one publication")
    rows = []
    for (model, rate, doi), part in frame.groupby(["Model", "Mask_rate", "Source_DOI"], sort=True):
        scores = evaluate(part)
        r2 = scores[r2_definition] if r2_definition == R2_DEFINITIONS[2] else scores[r2_definition]["R2"]
        if r2 is None or (r2_definition == R2_DEFINITIONS[2] and scores["materials_without_defined_R2"]):
            raise ValueError("Undefined R2; choose and disclose a valid reporting policy, do not silently omit materials")
        rows.append({"Model": model, "Mask_rate": rate, "Source_DOI": doi,
                     "Records": len(part), "Materials": scores["materials"], "R2": r2,
                     "RMSE": scores["material_macro_RMSE"]})
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["Model", "Mask_rate"]).agg(
        Publications=("Source_DOI", "size"), R2_mean=("R2", "mean"), R2_SD=("R2", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_SD=("RMSE", "std")).reset_index()
    return detail, summary

def repetition_scores(frame, aggregation, expected_folds=5):
    validate_predictions(frame)
    if aggregation not in {"pooled_oof", "fold_mean"}:
        raise ValueError("Choose pooled_oof or fold_mean explicitly")
    if "Outer_split" not in frame:
        raise ValueError("Outer_split is required")
    rows=[]
    for (model, rate), z in frame.groupby(["Model", "Mask_rate"]):
        if z.Outer_split.nunique() != expected_folds:
            raise ValueError("Incomplete cross-validation repetition")
        if z.groupby("Material_ID").Outer_split.nunique().max() != 1:
            raise ValueError("A material occurs in more than one test fold")
        scores = [regression(p.Y_true, p.Prediction) for _, p in z.groupby("Outer_split")] if aggregation == "fold_mean" else [regression(z.Y_true, z.Prediction)]
        if any(s["R2"] is None for s in scores):
            raise ValueError("Undefined R2; do not silently omit folds")
        rows.append({"Model":model, "Mask_rate":rate,
                     "R2":float(np.mean([s["R2"] for s in scores])),
                     "RMSE":float(np.mean([s["RMSE"] for s in scores]))})
    return pd.DataFrame(rows)

def export_lopo(predictions, output, r2_definition, expected_publications=12):
    frame=pd.read_csv(predictions)
    if "Mask_rate" not in frame:
        raise ValueError("Mask_rate is required")
    frame=frame.loc[frame.Mask_rate.isin([0.,1.])]
    if set(frame.Mask_rate)!={0.,1.}:
        raise ValueError("Both complete-input and 100% masking predictions are required")
    detail, summary = lopo_tables(frame, r2_definition, expected_publications)
    dest=Path(output)
    dest.mkdir(parents=True, exist_ok=False)
    detail.to_csv(dest/"publication_metrics.csv", index=False)
    summary.to_csv(dest/"table_s4_summary.csv", index=False)
    write_json(dest/"metric_definitions.json", {
        "within_publication_R2":r2_definition,
        "within_publication_RMSE":"arithmetic mean of per-material RMSE",
        "between_publication_summary":"equal publication weights; mean and sample SD (ddof=1)",
        "negative_R2":"retained", "expected_publications":expected_publications})
    from .plots import plot_lopo
    plot_lopo(detail, dest)
