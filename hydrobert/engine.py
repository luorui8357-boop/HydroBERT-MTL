"""Train-only scaling, nested epoch selection and auditable out-of-fold prediction."""
from pathlib import Path
import importlib.metadata
import random
import time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertConfig, BertModel, get_linear_schedule_with_warmup
from .data import FEATURES, TARGET, grouped_splits, outer_splits, validate_split, write_json
from .dataset import HydroDataset
from .metrics import evaluate, regression, reconstruction_metrics
from .model import HydroBERTMTL
from .losses import joint_loss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def scaler_state(scaler):
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "var": scaler.var_.tolist(),
            "n_features": int(scaler.n_features_in_), "samples": np.asarray(scaler.n_samples_seen_).tolist()}

def restore_scaler(state):
    s = StandardScaler()
    s.mean_, s.scale_, s.var_ = (np.asarray(state[k], float) for k in ("mean", "scale", "var"))
    s.n_features_in_ = state["n_features"]
    s.n_samples_seen_ = np.asarray(state["samples"])
    return s

def runtime():
    result = {}
    for name in ("torch", "transformers", "numpy", "pandas", "scikit-learn", "xgboost", "catboost", "tabpfn"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    result["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name()
    return result

def loaders(df, tokenizer, sx, sy, cfg, probability, training=False, mask_seed=None):
    dataset = HydroDataset(df, tokenizer, sx, sy, cfg["max_length"], probability, mask_seed)
    if training and len(dataset) < 2:
        raise ValueError("Batch normalization requires at least two training records")
    # Avoid a singleton final training batch; never drop evaluation records.
    drop = training and len(dataset) % cfg["batch_size"] == 1
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=training,
                      drop_last=drop, num_workers=0)

def predict(model, frame, tokenizer, sx, sy, cfg, device, rate=0., mask_seed=42):
    model.eval()
    raw = []
    with torch.no_grad():
        for batch in loaders(frame, tokenizer, sx, sy, cfg, rate, mask_seed=mask_seed):
            x = {k: v.to(device) for k, v in batch.items()}
            p, _ = model(x["input_ids"], x["attention_mask"], x["num_feats_input"], x["input_mask_flags"])
            raw.extend(p.reshape(-1).cpu().tolist())
    raw = sy.inverse_transform(np.asarray(raw).reshape(-1, 1)).reshape(-1)
    if cfg["prediction_clipping"] == "none":
        primary = raw.copy()
    elif cfg["prediction_clipping"] == "zero":
        primary = np.maximum(raw, 0.)
    else:
        raise ValueError("Unknown prediction clipping policy")
    if not np.isfinite(primary).all():
        raise ValueError("Nonfinite model prediction")
    return raw, primary

def validate_reconstruction(model, frame, tokenizer, sx, sy, cfg, device):
    model.eval()
    targets, predictions, flags = [], [], []
    for batch in loaders(frame, tokenizer, sx, sy, cfg, cfg["train_mask_probability"], mask_seed=cfg["seed"]):
        x={k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            _, values=model(x["input_ids"],x["attention_mask"],x["num_feats_input"],x["input_mask_flags"])
        predictions.append(values.cpu().numpy())
        targets.append(batch["targets_struct"].numpy())
        flags.append(batch["loss_mask_flags"].numpy())
    return reconstruction_metrics(np.concatenate(targets),np.concatenate(predictions),np.concatenate(flags))

def fit(train, tokenizer, cfg, device, epochs, dest, validation=None):
    set_seed(cfg["seed"])
    sx = StandardScaler().fit(train[FEATURES].to_numpy(float))
    if not np.isfinite(sx.mean_).all():
        raise ValueError("A feature has no observed values in this training fold")
    sy = StandardScaler().fit(train[[TARGET]].to_numpy(float))
    probability = 0. if cfg.get("variant") == "no_dynamic_mask" else cfg["train_mask_probability"]
    auxiliary = 0. if cfg.get("variant") in {"no_reconstruction", "numerical_only", "text_only"} else cfg["auxiliary_loss_weight"]
    loader = loaders(train, tokenizer, sx, sy, cfg, probability, training=True)
    model = HydroBERTMTL(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
                                  eps=cfg["adam_epsilon"], betas=tuple(cfg["adam_betas"]), foreach=False, fused=False)
    schedule = get_linear_schedule_with_warmup(optimizer, cfg["warmup_steps"], len(loader) * epochs)
    curve = []
    start = time.time()
    dest.mkdir(parents=True, exist_ok=False)
    for epoch in range(1, epochs + 1):
        model.train()
        sums = np.zeros(3)
        count = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            pred, structures = model(batch["input_ids"], batch["attention_mask"], batch["num_feats_input"], batch["input_mask_flags"])
            total, main, aux = joint_loss(pred, structures, batch, auxiliary)
            if not torch.isfinite(total):
                raise ValueError("Training loss became nonfinite")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], foreach=False)
            optimizer.step()
            schedule.step()
            n = len(pred)
            sums += n * np.array([total.item(), main.item(), aux.item()])
            count += n
        row = dict(Epoch=epoch, Total_loss=sums[0]/count, Prediction_loss=sums[1]/count,
                   Reconstruction_loss=sums[2]/count, Learning_rate=schedule.get_last_lr()[0],
                   Elapsed_seconds=time.time()-start)
        if validation is not None:
            # Outer test records are never supplied as validation to this function.
            raw, _ = predict(model, validation, tokenizer, sx, sy, cfg, device)
            row["Validation_R2"] = regression(validation[TARGET], raw)["R2"]
            rec=validate_reconstruction(model,validation,tokenizer,sx,sy,cfg,device)
            row["Validation_reconstruction_R2"]=rec["R2"]
            row["Validation_reconstruction_MSE"]=rec["MSE"]
            row["Reconstruction_eligible_cells"]=rec["eligible_cells"]
            row["Reconstruction_defined_descriptors"]=rec["defined_descriptors"]
            for j, score in enumerate(rec["per_descriptor"]):
                row[f"Validation_reconstruction_R2_descriptor_{j+1}"]=None if score is None else score["R2"]
        curve.append(row)
        pd.DataFrame(curve).to_csv(dest / "training_curve.csv", index=False)
        print(str(dest.name), row, flush=True)
    return model, sx, sy, pd.DataFrame(curve)

def select_epochs(train, tokenizer, cfg, device, dest):
    splits = grouped_splits(train, cfg["inner_splits"])
    curves, assignments = [], []
    for fold, (a, b) in enumerate(splits, 1):
        validate_split(train, a, b)
        role = train[["Record_ID", "Material_ID"]].copy()
        role["Inner_fold"] = fold
        role["Role"] = "inner_training"
        role.loc[b, "Role"] = "inner_validation"
        assignments.append(role)
        model, sx, sy, curve = fit(train.iloc[a], tokenizer, cfg, device, cfg["max_epochs"],
                                   dest / f"inner_{fold:02d}", validation=train.iloc[b])
        curve["Inner_fold"] = fold
        curves.append(curve)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.concat(assignments).to_csv(dest / "inner_assignments.csv", index=False)
    scores = pd.concat(curves).groupby("Epoch").Validation_R2.mean()
    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Undefined inner validation R2; revise grouping rather than silently ignoring folds")
    selected = int(scores.idxmax())
    scores.rename("Mean_inner_validation_R2").to_csv(dest / "epoch_selection.csv")
    return selected

def prediction_table(frame, raw, primary, model, outer, rate):
    columns = [c for c in ("Record_ID", "Material_ID", "Source_DOI", "Isotherm_ID") if c in frame]
    out = frame[columns].copy()
    if TARGET in frame:
        out["Y_true"] = frame[TARGET].to_numpy()
    out["Prediction_raw"], out["Prediction"] = raw, primary
    out["Model"], out["Outer_split"], out["Mask_rate"] = model, outer, rate
    return out

def validate_checkpoint_policy(cfg):
    policy=cfg.get("checkpoint_policy", "explicit")
    if policy not in {"fixed", "nested_refit"}:
        raise ValueError("The manuscript does not specify checkpoint selection. Set --checkpoint-policy and, for fixed, --epochs from the actual experimental protocol.")
    if policy == "fixed" and (not isinstance(cfg.get("fixed_epochs"), int) or not 1 <= cfg["fixed_epochs"] <= cfg["max_epochs"]):
        raise ValueError("Fixed training requires an explicit epoch count between 1 and max_epochs")
    return policy

def train_experiment(df, cfg, protocol, output, device="cuda:0", selected_outer=None):
    policy=validate_checkpoint_policy(cfg)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(cfg["backbone"])
    write_json(output / "config.json", cfg)
    write_json(output / "runtime.json", runtime())
    assignments, predictions, summaries = [], [], []
    splits = outer_splits(df, protocol, cfg)
    if selected_outer and selected_outer not in {label for label,_,_ in splits}:
        raise ValueError(f"Unknown outer split: {selected_outer}")
    for label, a, b in splits:
        validate_split(df, a, b, publication=protocol == "lopo")
        role = df[[c for c in ("Record_ID", "Material_ID", "Source_DOI") if c in df]].copy()
        role["Outer_split"] = label
        role["Role"] = "outer_training"
        role.loc[b, "Role"] = "outer_test"
        assignments.append(role)
    pd.concat(assignments).to_csv(output / "outer_assignments.csv", index=False)
    for label, a, b in splits:
        if selected_outer and label != selected_outer:
            continue
        dest = output / label
        dest.mkdir()
        training = df.iloc[a].reset_index(drop=True)
        selected = select_epochs(training, tokenizer, cfg, device, dest) if policy == "nested_refit" else cfg["fixed_epochs"]
        model, sx, sy, curve = fit(training, tokenizer, cfg, device, selected, dest / "refit")
        tokenizer.save_pretrained(dest / "tokenizer")
        checkpoint = {"format_version": 1, "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                      "scaler_x": scaler_state(sx), "scaler_y": scaler_state(sy), "config": cfg,
                      "bert_config": model.bert.config.to_dict(), "selected_epochs": selected,
                      "train_record_ids": training.Record_ID.tolist(), "train_material_ids": sorted(training.Material_ID.unique()),
                      "protocol": protocol, "outer_split": label}
        torch.save(checkpoint, dest / "checkpoint.pt")
        for rate in cfg["evaluation_masks"]:
            raw, primary = predict(model, df.iloc[b], tokenizer, sx, sy, cfg, device, rate, cfg["seed"])
            tab = prediction_table(df.iloc[b], raw, primary, "HydroBERT-MTL", label, rate)
            predictions.append(tab)
            summaries.append(dict(Outer_split=label, Mask_rate=rate, Selected_epochs=selected, Metrics=evaluate(tab)))
        pd.concat(predictions).to_csv(output / "predictions.csv", index=False)
        write_json(output / "fold_metrics.json", summaries)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(output / "pooled_metrics.json", {
        str(rate): evaluate(pd.concat(predictions).loc[lambda z: z.Mask_rate.eq(rate)]) for rate in cfg["evaluation_masks"]})

def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Not a release-format checkpoint; inspect legacy provenance before converting")
    cfg = checkpoint["config"]
    bert_config = BertConfig.from_dict(checkpoint["bert_config"])
    bert_config._attn_implementation = cfg.get("attention_implementation", "eager")
    backbone = BertModel(bert_config, add_pooling_layer=True)
    model = HydroBERTMTL(cfg, backbone).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint, restore_scaler(checkpoint["scaler_x"]), restore_scaler(checkpoint["scaler_y"])
