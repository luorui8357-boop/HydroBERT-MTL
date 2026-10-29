"""Reproducible commands; no training or downloads occur at module import."""
from pathlib import Path
import argparse
import json
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
import pandas as pd
from .data import load_data, sha256, write_json, outer_splits, validate_split

def parser():
    ap = argparse.ArgumentParser(description="HydroBERT-MTL manuscript-aligned implementation")
    sub = ap.add_subparsers(dest="action", required=True)
    for command in ("prepare", "train", "baselines", "repeated-cv"):
        p = sub.add_parser(command)
        p.add_argument("--data", required=True)
        p.add_argument("--metadata")
        p.add_argument("--config", default="configs/paper.json")
        p.add_argument("--protocol", choices=["grouped_cv", "material_holdout", "lopo"], default="grouped_cv")
        p.add_argument("--output", required=True)
        if command != "prepare":
            p.add_argument("--device", default="cuda:0")
            p.add_argument("--outer-split", help="Optional named split for distributed jobs; never selects by performance")
        if command in {"train", "repeated-cv"}:
            p.add_argument("--backbone", help="Local SciBERT path instead of the named public backbone")
            p.add_argument("--variant", choices=["full", "no_reconstruction", "no_dynamic_mask", "text_only", "numerical_only"], default="full")
            p.add_argument("--checkpoint-policy", choices=["fixed", "nested_refit"], help="Must match the actual experiment; no inferred historical default")
            p.add_argument("--epochs", type=int, help="Required for fixed checkpoint policy")
        if command == "repeated-cv":
            p.add_argument("--repetitions",type=int,default=100)
            p.add_argument("--aggregation",choices=["pooled_oof","fold_mean"],help="Required unless specified in config")
        if command == "baselines":
            p.add_argument("--baseline-config", default="configs/baselines.json")
            p.add_argument("--models", nargs="+", default=["Ridge", "RF", "XGBoost", "FT-Transformer", "TabPFN", "TabR", "CatBoost"])
            p.add_argument("--tabpfn-path")
    p = sub.add_parser("predict")
    p.add_argument("--data", required=True)
    p.add_argument("--metadata")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--mask-rate", type=float, default=0.)
    p.add_argument("--tokenizer", help="Defaults to checkpoint's sibling tokenizer directory")
    p = sub.add_parser("metrics")
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("plot")
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("lopo-report",help="Generate Table S4 and Fig. S4 from all publication predictions")
    p.add_argument("--predictions",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--expected-publications",type=int,default=12)
    p.add_argument("--r2-definition",choices=["record_weighted","pooled_material_balanced","material_macro_R2_defined_only"],required=True)
    return ap

def main():
    args = parser().parse_args()
    if args.action == "lopo-report":
        from .reporting import export_lopo
        export_lopo(args.predictions,args.output,args.r2_definition,args.expected_publications)
        return
    if args.action == "metrics":
        from .metrics import evaluate
        f = pd.read_csv(args.predictions)
        out = {}
        for (name, rate), z in f.groupby(["Model", "Mask_rate"]):
            if z.Record_ID.duplicated().any():
                raise ValueError("Repeated predictions detected; separate repetitions before pooling metrics")
            out[f"{name}|{rate}"] = evaluate(z)
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        write_json(args.output, out)
        return
    if args.action == "plot":
        from .plots import plot
        plot(args.predictions, args.output)
        return
    if args.action == "predict":
        from transformers import AutoTokenizer
        from .engine import load_checkpoint, predict, prediction_table
        df = load_data(args.data, args.metadata, labels_required=False)
        model, checkpoint, sx, sy = load_checkpoint(args.checkpoint, args.device)
        if set(df.Record_ID) & set(checkpoint["train_record_ids"]) or set(df.Material_ID) & set(checkpoint["train_material_ids"]):
            raise ValueError("Prediction materials overlap checkpoint training materials; use disjoint evaluation data")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or str(Path(args.checkpoint).parent/"tokenizer"))
        raw, primary = predict(model, df, tokenizer, sx, sy, checkpoint["config"], args.device, args.mask_rate)
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        prediction_table(df, raw, primary, "HydroBERT-MTL", checkpoint["outer_split"], args.mask_rate).to_csv(args.output, index=False)
        return
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    df = load_data(args.data, args.metadata)
    if args.action == "prepare":
        dest = Path(args.output)
        dest.mkdir(parents=True, exist_ok=False)
        rows = []
        for label, a, b in outer_splits(df, args.protocol, cfg):
            validate_split(df, a, b, args.protocol == "lopo")
            row = df[[c for c in ("Record_ID", "Material_ID", "Source_DOI") if c in df]].copy()
            row["Outer_split"], row["Role"] = label, "outer_training"
            row.loc[b, "Role"] = "outer_test"
            rows.append(row)
        pd.concat(rows).to_csv(dest/"assignments.csv", index=False)
        write_json(dest/"manifest.json", {"input_sha256":sha256(args.data), "records":len(df),
                   "materials":int(df.Material_ID.nunique()), "publications":int(df.Source_DOI.nunique()) if "Source_DOI" in df else None,
                   "protocol":args.protocol, "config":cfg})
    elif args.action in {"train", "repeated-cv"}:
        from .engine import train_experiment
        cfg["variant"] = args.variant
        if args.backbone:
            cfg["backbone"] = args.backbone
        if args.checkpoint_policy:
            cfg["checkpoint_policy"]=args.checkpoint_policy
        if args.epochs is not None:
            cfg["fixed_epochs"]=args.epochs
        if args.action == "repeated-cv":
            if args.protocol != "grouped_cv" or args.outer_split:
                raise ValueError("Repeated CV requires all grouped folds")
            from .repeated import repeated_cv
            repeated_cv(df,cfg,args.output,args.repetitions,args.aggregation or cfg.get("repeated_cv_aggregation"),args.device)
        else:
            train_experiment(df, cfg, args.protocol, args.output, args.device, args.outer_split)
        write_json(Path(args.output)/"input_manifest.json", {"input_sha256":sha256(args.data), "rows":len(df)})
    else:
        from .baselines import MODELS, run_baselines
        if not set(args.models) <= set(MODELS):
            raise ValueError("Unrecognized baseline name")
        baseline = json.loads(Path(args.baseline_config).read_text(encoding="utf-8"))
        run_baselines(df, cfg, baseline, args.protocol, args.output, args.models, args.device, args.tabpfn_path, args.outer_split)
        write_json(Path(args.output)/"input_manifest.json", {"input_sha256":sha256(args.data), "rows":len(df)})

if __name__ == "__main__":
    main()
