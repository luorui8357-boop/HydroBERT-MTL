"""Repeated five-fold evaluation with new MATERIAL partitions each repetition."""
import copy
from pathlib import Path
import pandas as pd
from .data import write_json
from .reporting import repetition_scores

def repeated_cv(df, cfg, output, repetitions, aggregation, device="cpu"):
    if repetitions < 2:
        raise ValueError("At least two repetitions are needed for sample SD")
    if aggregation not in {"pooled_oof", "fold_mean"}:
        raise ValueError("Specify the per-repetition metric aggregation")
    from .engine import train_experiment
    from .engine import validate_checkpoint_policy
    validate_checkpoint_policy(cfg)
    from .plots import plot_repetitions
    dest=Path(output)
    dest.mkdir(parents=True, exist_ok=False)
    write_json(dest/"repeated_cv_config.json", {"repetitions":repetitions,"n_splits":cfg["n_splits"],
        "aggregation":aggregation,"SD_ddof":1,"base_seed":cfg["seed"],
        "split_method":"shuffle sorted unique Material_ID values, partition into five equal-sized material sets",
        "historical_reproduction":"Requires agreement with original split and checkpoint settings"})
    results=[]
    for repetition in range(1,repetitions+1):
        current=copy.deepcopy(cfg)
        current["seed"]=cfg["seed"]+repetition-1
        current["split_seed"]=current["seed"]
        current["shuffle_groups"]=True
        current["split_manifests"]={}
        current["evaluation_masks"]=[0.,1.]
        run=dest/f"repetition_{repetition:03d}"
        train_experiment(df,current,"grouped_cv",run,device)
        scores=repetition_scores(pd.read_csv(run/"predictions.csv"),aggregation,cfg["n_splits"])
        scores["Repetition"]=repetition
        scores["Seed"]=current["seed"]
        results.append(scores)
        pd.concat(results).to_csv(dest/"repetition_metrics.csv",index=False)
    result=pd.concat(results)
    summary=result.groupby(["Model","Mask_rate"]).agg(Repetitions=("Repetition","size"),
        R2_mean=("R2","mean"),R2_SD=("R2","std"),RMSE_mean=("RMSE","mean"),RMSE_SD=("RMSE","std")).reset_index()
    summary.to_csv(dest/"repeated_cv_summary.csv",index=False)
    plot_repetitions(result,dest)
