"""Descriptor schema, explicit provenance and leakage-safe material groups."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

FEATURES = ["C(wt%)", "H(wt%)", "N(wt%)", "O(wt%)", "S_BET(m2/g)",
            "V_t(cm3/g)", "V_m(cm3/g)", "V_um(cm3/g)", "V_meso(cm3/g)",
            "Temp(K)", "Pressure(bar)"]
STRUCTURES = FEATURES[4:9]
TARGET = "H2_Uptake(wt%)"

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_data(path, metadata=None, labels_required=True):
    path = Path(path)
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    missing = set(FEATURES) - set(df)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    if "Record_ID" not in df:
        raise ValueError("Explicit stable Record_ID values are required; do not infer row ordering.")
    df["Record_ID"] = df.Record_ID.astype(str)
    if df.Record_ID.duplicated().any():
        raise ValueError("Record_ID must be unique")
    if metadata is not None:
        m = pd.read_csv(metadata, dtype=str)
        if m.Record_ID.duplicated().any():
            raise ValueError("Duplicate metadata Record_ID")
        m = m.set_index("Record_ID")
        if not set(df.Record_ID) <= set(m.index):
            raise ValueError("Metadata does not cover all input records")
        for c in ("Material_ID", "Source_DOI", "Isotherm_ID"):
            if c in m:
                mapped = df.Record_ID.map(m[c])
                if c in df and not df[c].astype(str).eq(mapped).all():
                    raise ValueError(f"Input and metadata disagree on {c}")
                df[c] = mapped
    if "Material_ID" not in df or df.Material_ID.isna().any():
        raise ValueError("Verified Material_ID required. Descriptor equality alone cannot establish material identity.")
    df["Material_ID"] = df.Material_ID.astype(str)
    if "Source_DOI" in df:
        if df.Source_DOI.isna().any():
            raise ValueError("Source_DOI contains blanks")
        if df.groupby("Material_ID").Source_DOI.nunique().max() > 1:
            raise ValueError("A material occurs in several publications; resolve cross-study material overlap first")
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="raise")
    # Only structural descriptors may be naturally missing in the supplied framework.
    required_observed = FEATURES[:4] + FEATURES[9:]
    if not np.isfinite(df[required_observed].to_numpy(float)).all():
        raise ValueError("Elemental composition, temperature and pressure must be observed and finite")
    if np.isinf(df[STRUCTURES].to_numpy(float)).any():
        raise ValueError("Structural descriptors contain infinity")
    if labels_required:
        if TARGET not in df:
            raise ValueError(f"Target column required: {TARGET}")
        df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")
        if not np.isfinite(df[TARGET]).all():
            raise ValueError("Targets must be observed and finite")
    return df.reset_index(drop=True)

def grouped_splits(df, n_splits=5, shuffle=False, seed=42):
    if df.Material_ID.nunique() < n_splits:
        raise ValueError(f"At least {n_splits} materials required")
    if shuffle:
        # Randomize MATERIALS, never individual adsorption records. This explicit
        # algorithm is shared across supported scikit-learn versions.
        materials = np.array(sorted(df.Material_ID.unique()))
        materials = np.random.default_rng(seed).permutation(materials)
        return [(np.flatnonzero(~df.Material_ID.isin(part)),
                 np.flatnonzero(df.Material_ID.isin(part)))
                for part in np.array_split(materials, n_splits)]
    return list(GroupKFold(n_splits=n_splits).split(df, groups=df.Material_ID))

def outer_splits(df, protocol, cfg):
    # Frozen assignments prevent tied material sizes changing fold identities
    # across NumPy/scikit-learn versions. Apply only to the matching record set.
    manifest = cfg.get("split_manifests", {}).get(protocol)
    if manifest:
        path = Path(manifest)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        saved = pd.read_csv(path, dtype={"Record_ID": str, "Material_ID": str})
        if "Record_ID" in df and set(df.Record_ID) == set(saved.Record_ID):
            if saved.Record_ID.duplicated().any():
                raise ValueError("Frozen split contains duplicate record IDs")
            saved = saved.set_index("Record_ID").loc[df.Record_ID].reset_index()
            if not saved.Material_ID.eq(df.Material_ID.to_numpy()).all():
                raise ValueError("Material identities disagree with frozen split manifest")
            if "Source_DOI" in saved and "Source_DOI" in df and not saved.Source_DOI.eq(df.Source_DOI.to_numpy()).all():
                raise ValueError("Source provenance disagrees with frozen split manifest")
            if protocol == "material_holdout":
                return [("material_holdout", np.flatnonzero(saved.Partition.eq("train_validation")),
                         np.flatnonzero(saved.Partition.eq("holdout")))]
            column = "Test_fold" if protocol == "grouped_cv" else "Test_publication_fold"
            label = "fold" if protocol == "grouped_cv" else "publication"
            return [(f"{label}_{int(f):02d}", np.flatnonzero(saved[column].ne(f)), np.flatnonzero(saved[column].eq(f)))
                    for f in sorted(saved[column].unique())]
    if protocol == "grouped_cv":
        return [(f"fold_{i:02d}", a, b) for i, (a, b) in enumerate(grouped_splits(
            df, cfg["n_splits"], cfg.get("shuffle_groups", False), cfg.get("split_seed", cfg["seed"])) , 1)]
    if protocol == "material_holdout":
        # Preserves the archived 80/20 material-level, target-quartile-stratified split.
        material = df.groupby("Material_ID", sort=True)[TARGET].max().reset_index()
        bins = pd.qcut(material[TARGET], q=4, labels=False, duplicates="drop")
        a, b = train_test_split(material.Material_ID, test_size=cfg["holdout_fraction"],
                                stratify=bins, random_state=cfg["seed"])
        return [("material_holdout", np.flatnonzero(df.Material_ID.isin(a)), np.flatnonzero(df.Material_ID.isin(b)))]
    if protocol == "lopo":
        if "Source_DOI" not in df:
            raise ValueError("Source_DOI required for leave-one-publication-out evaluation")
        return [(f"publication_{i:02d}", np.flatnonzero(df.Source_DOI.ne(doi)), np.flatnonzero(df.Source_DOI.eq(doi)))
                for i, doi in enumerate(sorted(df.Source_DOI.unique()), 1)]
    raise ValueError(protocol)

def validate_split(df, train, test, publication=False):
    if not len(train) or not len(test) or set(train) & set(test):
        raise ValueError("Empty or overlapping partitions")
    if set(df.iloc[train].Material_ID) & set(df.iloc[test].Material_ID):
        raise ValueError("Material leakage across partitions")
    if publication and set(df.iloc[train].Source_DOI) & set(df.iloc[test].Source_DOI):
        raise ValueError("Publication leakage across partitions")

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
