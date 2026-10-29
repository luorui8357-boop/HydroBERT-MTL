"""Seven baselines share outer partitions and standardized-mean structural imputation."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from .data import FEATURES, TARGET, grouped_splits, outer_splits, validate_split, write_json
from .masking import masks
from .engine import set_seed, scaler_state, prediction_table
from .metrics import evaluate, regression, material_weights

MODELS = ["Ridge", "RF", "XGBoost", "FT-Transformer", "TabPFN", "TabR", "CatBoost"]

def transform(frame, scaler, rate=0., seed=42):
    raw = frame[FEATURES].to_numpy(float)
    norm = scaler.transform(raw)
    for i in range(len(raw)):
        flags, _ = masks(raw[i], rate, random.Random(seed + i))
        norm[i, 4:9] = np.where(flags, 0., norm[i, 4:9])
    if not np.isfinite(norm).all():
        raise ValueError("A non-structural value is missing or a training feature is unobserved")
    return norm.astype(np.float32)

def classical(kind, params, cfg):
    if kind == "Ridge":
        from sklearn.linear_model import Ridge
        return Ridge(**params)
    if kind == "RF":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=256, random_state=cfg["seed"], n_jobs=1, **params)
    if kind == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=500, learning_rate=.05, subsample=.9, colsample_bytree=.9,
                            objective="reg:squarederror", tree_method="hist", n_jobs=1, random_state=cfg["seed"], **params)
    if kind == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=600, learning_rate=.05, loss_function="RMSE", random_seed=cfg["seed"],
                                 verbose=False, thread_count=1, allow_writing_files=False, **params)
    raise ValueError(kind)

def neural_fit(kind, train, validation, params, cfg, device, epochs):
    from . import tabular
    set_seed(cfg["seed"])
    sx = StandardScaler().fit(train[FEATURES].to_numpy(float))
    sy = StandardScaler().fit(train[[TARGET]].to_numpy(float))
    x = torch.tensor(transform(train, sx), device=device)
    y = torch.tensor(sy.transform(train[[TARGET]].to_numpy(float)).reshape(-1), dtype=torch.float32, device=device)
    model = tabular.build(kind, params).to(device)
    parameters = model.make_parameter_groups() if kind == "FT-Transformer" else model.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=params["lr"], weight_decay=cfg["neural_weight_decay"], foreach=False, fused=False)
    best, best_epoch, best_state = float("inf"), 1, None
    for epoch in range(1, epochs + 1):
        model.train()
        for index in torch.randperm(len(train), device=device).split(cfg["neural_batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            p = tabular.forward(model, kind, x[index], x, y, index, params.get("context", 32))
            loss = torch.nn.functional.mse_loss(p, y[index])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite baseline loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["neural_gradient_clip"], foreach=False)
            optimizer.step()
        if validation is not None:
            p = neural_predict(model, kind, validation, sx, sy, x, y, params, device)
            mse = np.average((validation[TARGET].to_numpy()-p)**2, weights=material_weights(validation.Material_ID))
            if mse < best:
                best, best_epoch = float(mse), epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch - best_epoch >= cfg["neural_patience"]:
                break
    if validation is not None:
        model.load_state_dict(best_state)
    return model, sx, sy, x, y, best_epoch

def neural_predict(model, kind, frame, sx, sy, bank_x, bank_y, params, device, rate=0., seed=42):
    from . import tabular
    model.eval()
    x = torch.tensor(transform(frame, sx, rate, seed), device=device)
    with torch.no_grad():
        p = torch.cat([tabular.forward(model, kind, b, bank_x, bank_y, context=params.get("context", 32)) for b in x.split(128)]).cpu().numpy()
    return sy.inverse_transform(p.reshape(-1, 1)).reshape(-1)

def run_baselines(df, paper, cfg, protocol, output, models, device="cpu", tabpfn_path=None, selected_outer=None):
    import joblib
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"baseline_config.json", cfg)
    write_json(output/"paper_config.json", paper)
    all_predictions, outer_metadata = [], []
    assignments = []
    splits = outer_splits(df, protocol, paper)
    for label, a, b in splits:
        validate_split(df, a, b, protocol == "lopo")
        role = df[[c for c in ("Record_ID", "Material_ID", "Source_DOI") if c in df]].copy()
        role["Outer_split"], role["Role"] = label, "outer_training"
        role.loc[b, "Role"] = "outer_test"
        assignments.append(role)
    pd.concat(assignments).to_csv(output/"outer_assignments.csv", index=False)
    for label, a, b in splits:
        if selected_outer and label != selected_outer:
            continue
        train, test = df.iloc[a].reset_index(drop=True), df.iloc[b].reset_index(drop=True)
        inner = grouped_splits(train, cfg["inner_splits"])
        inner_roles = []
        for f, (ti, vi) in enumerate(inner, 1):
            role = train[["Record_ID", "Material_ID"]].copy()
            role["Inner_fold"], role["Role"] = f, "inner_training"
            role.loc[vi, "Role"] = "inner_validation"
            inner_roles.append(role)
        for kind in models:
            print("BASELINE", label, kind, flush=True)
            dest = output / label / kind
            dest.mkdir(parents=True)
            pd.concat(inner_roles).to_csv(dest/"inner_assignments.csv", index=False)
            neural = kind in {"FT-Transformer", "TabR"}
            chosen, epochs, search = 0, None, []
            if kind != "TabPFN":
                candidate_scores = []
                for c, params in enumerate(cfg[kind]):
                    squared, weights, best_epochs = [], [], []
                    for fold, (ti, vi) in enumerate(inner, 1):
                        validate_split(train, ti, vi)
                        tr, va = train.iloc[ti], train.iloc[vi]
                        if neural:
                            m, sx, sy, bx, by, ep = neural_fit(kind, tr, va, params, cfg, device, cfg["neural_max_epochs"])
                            p = neural_predict(m, kind, va, sx, sy, bx, by, params, device)
                            best_epochs.append(ep)
                        else:
                            sx = StandardScaler().fit(tr[FEATURES].to_numpy(float))
                            m = classical(kind, params, cfg).fit(transform(tr, sx), tr[TARGET].to_numpy())
                            p = m.predict(transform(va, sx))
                        w = material_weights(va.Material_ID)
                        squared.extend((va[TARGET].to_numpy()-p)**2)
                        weights.extend(w)
                        search.append(dict(Candidate=c, Inner_fold=fold, RMSE=regression(va[TARGET], p, w)["RMSE"],
                                           Selected_epoch=ep if neural else None))
                        del m
                    candidate_scores.append(float(np.average(squared, weights=weights)))
                    print(kind, "candidate", c, candidate_scores[-1], flush=True)
                chosen = int(np.argmin(candidate_scores))
                params = cfg[kind][chosen]
                if neural:
                    epochs = int(np.median([r["Selected_epoch"] for r in search if r["Candidate"] == chosen]))
                pd.DataFrame(search).to_csv(dest/"inner_scores.csv", index=False)
            else:
                params = cfg[kind]
            sx = StandardScaler().fit(train[FEATURES].to_numpy(float))
            if neural:
                m, sx, sy, bx, by, _ = neural_fit(kind, train, None, params, cfg, device, epochs)
                torch.save({"model_state_dict": {k:v.detach().cpu() for k,v in m.state_dict().items()},
                            "scaler_x":scaler_state(sx), "scaler_y":scaler_state(sy),
                            "train_x":bx.cpu(), "train_y":by.cpu(), "parameters":params,
                            "epochs":epochs, "train_record_ids":train.Record_ID.tolist()}, dest/"checkpoint.pt")
            elif kind == "TabPFN":
                from tabpfn import TabPFNRegressor
                kw = dict(n_estimators=params["n_estimators"], random_state=params["random_state"], device=device)
                if tabpfn_path:
                    kw["model_path"] = tabpfn_path
                m = TabPFNRegressor(**kw).fit(transform(train, sx), train[TARGET].to_numpy())
                # Context is the original training data; no fine-tuning or test labels.
                # Keep that private context outside the uploadable source repository.
            else:
                m = classical(kind, params, cfg).fit(transform(train, sx), train[TARGET].to_numpy())
                joblib.dump({"model":m, "scaler_x":sx, "train_record_ids":train.Record_ID.tolist()}, dest/"model.joblib")
            for rate in paper["evaluation_masks"]:
                if neural:
                    raw = neural_predict(m, kind, test, sx, sy, bx, by, params, device, rate, paper["seed"])
                else:
                    raw = np.asarray(m.predict(transform(test, sx, rate, paper["seed"]))).reshape(-1)
                primary = raw.copy() if paper["prediction_clipping"] == "none" else np.maximum(raw, 0.)
                tab = prediction_table(test, raw, primary, kind, label, rate)
                all_predictions.append(tab)
                outer_metadata.append(dict(Model=kind, Outer_split=label, Mask_rate=rate, Metrics=evaluate(tab)))
            write_json(dest/"settings.json", {"parameters":params, "selected_epochs":epochs,
                       "missing_handling":"11 standardized features, masked structure set to zero; no explicit mask flags",
                       "train_record_ids":train.Record_ID.tolist(), "test_record_ids":test.Record_ID.tolist()})
            pd.concat(all_predictions).to_csv(output/"predictions.csv", index=False)
            write_json(output/"fold_metrics.json", outer_metadata)
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
