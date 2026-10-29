# HydroBERT-MTL

Text–tabular multi-task learning for hydrogen-uptake prediction with missing structural descriptors.

## Model and configuration

The full model implements the architecture described in Sections 2.1–2.4:

- SciBERT (`allenai/scibert_scivocab_uncased`), 768-dimensional CLS representation projected to 128 dimensions.
- Eleven standardized numerical descriptors plus five structural-missingness flags, encoded as 64 dimensions.
- Concatenated 192-dimensional representation, a 192→128→1 uptake head, and a 192→64→5 reconstruction head.
- Dynamic 60% masking of observed structural descriptors during training. The text and numerical branches share each mask. Naturally missing descriptors have no reconstruction target.
- AdamW at 2e-5, linear learning-rate decay, auxiliary weight 0.5, batch size 16, maximum 50 epochs, dropout 0.1 and base seed 42.

`configs/paper.json` contains these settings. Features and uptake targets are standardized using training-partition statistics only. Predictions are converted back to wt% before reporting metrics. Prediction clipping is disabled; any different historical clipping rule must be documented and evaluated separately.

## Installation

The core environment is Python 3.10 with `requirements-core.txt`. The baseline environment is Python 3.12 with `requirements-baselines.txt`. Use separate environments because their pinned dependencies differ.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m hydrobert.cli --help
```

## Data and identifiers

Supply CSV/XLSX data following `docs/data_schema.md`. The input requires stable `Record_ID`, `Material_ID`, and, for publication-based evaluation, `Source_DOI` columns. Missing structural descriptors are blank/NaN; an observed zero is retained as an observation. The target is never included in text prompts or input features.

The package includes record/material/publication metadata and frozen single-run five-fold, material-holdout and leave-one-publication-out assignments. Their 2,268 records, 51 materials and 12 publications are internally consistent. Exact correspondence to the original numerical records must be verified by the data custodian; row order alone is not sufficient.

Numerical research data, target values, trained task weights and paper predictions are not included. Code and metadata alone therefore do not reproduce the published numerical results. See `docs/reproducibility.md` for protocol definitions and unresolved historical correspondence.

## Checkpoint selection

The manuscript specifies a maximum of 50 epochs but does not fully define how the reported checkpoint was selected. The release therefore requires an explicit policy instead of assuming a historical procedure:

- `--checkpoint-policy fixed --epochs N`: use a predeclared N between 1 and 50; do not choose it using outer-test performance.
- `--checkpoint-policy nested_refit`: use training-only inner material-grouped five-fold selection, then refit from initialization. This is an optional protocol, not an assertion about the historical paper run.

For fixed training, set `EPOCHS` below to the value established by the experiment records. No numerical value is supplied as a claimed historical default.

```bash
: "${EPOCHS:?Set EPOCHS from the actual experiment records}"
python -m hydrobert.cli train --data /path/input.csv --protocol grouped_cv --checkpoint-policy fixed --epochs "$EPOCHS" --output /path/new_grouped_run
python -m hydrobert.cli train --data /path/input.csv --protocol lopo --checkpoint-policy fixed --epochs "$EPOCHS" --output /path/new_lopo_run
```

All six masking rates use the same fitted checkpoint. Evaluation masks are reproducible for a fixed seed and row order; text and numerical masks agree. All preprocessing is fitted on training data. Entire materials and, in LOPO, entire publications are excluded from the corresponding training partition. Outputs include assignments, configuration, training curves, checkpoints and individual predictions. Output directories must be new.

## Repeated five-fold validation and Fig. S3

```bash
python -m hydrobert.cli repeated-cv --data /path/input.csv --checkpoint-policy fixed --epochs "$EPOCHS" --repetitions 100 --aggregation fold_mean --output /path/new_repeated_run
```

Each repetition shuffles sorted unique material identifiers with seed `42 + repetition - 1`, divides materials into five groups, and keeps all records of each material together. This path bypasses the fixed single-run fold manifest. The same repetition seed controls model initialization; complete inputs and 100% masking use the same fitted models and outer records.

The confirmed R² aggregation is the arithmetic mean of the five fold-specific R² values. The `fold_mean` implementation also averages fold-specific RMSE. The 100 repetition-level results are then summarized as mean and sample SD (`ddof=1`). This explicitly documented split generator must still be checked against any historical repetition assignments before claiming exact reproduction of Fig. S3.

Outputs include `repetition_001/` through `repetition_100/`, their fold assignments and predictions, `repetition_metrics.csv`, `repeated_cv_summary.csv`, and `Fig_S3_repeated_cv` in PNG/PDF/SVG.

## LOPO summary and Fig. S4

Combine predictions for all evaluated models, with identical test record IDs and target values. Select the within-publication R² definition actually used in the experiment:

- `record_weighted`: ordinary R² across records within a publication.
- `pooled_material_balanced`: weighted R² with each record weighted by the inverse number of records for its material.
- `material_macro_R2_defined_only`: mean of material-specific R²; the report rejects undefined material R² rather than silently dropping materials.

```bash
: "${LOPO_R2_DEFINITION:?Set the within-publication R2 definition from the actual evaluation}"
python -m hydrobert.cli lopo-report --predictions /path/all_model_predictions.csv --r2-definition "$LOPO_R2_DEFINITION" --expected-publications 12 --output /path/new_lopo_report
```

Within each publication, RMSE is the arithmetic mean of the material-specific RMSE values. Across the 12 publications, equal publication weights are used for the mean and sample SD. This is not the square root of a material-weighted pooled MSE. All negative R² values are retained.

The report selects complete inputs and 100% masking and generates `publication_metrics.csv`, `table_s4_summary.csv`, `metric_definitions.json`, and `Fig_S4_lopo` in PNG/PDF/SVG. No paper scores or manually chosen winning counts are inserted.

## Baselines and ablations

```bash
python -m hydrobert.cli baselines --data /path/input.csv --protocol grouped_cv --output /path/new_baselines --models Ridge RF XGBoost FT-Transformer TabPFN TabR CatBoost --tabpfn-path /path/tabpfn-v2-regressor.ckpt
```

All seven baselines receive the same 11 standardized numerical features, with masked structures set to the training mean. They do not receive text, reconstruction supervision or mask indicators. This matches the stated missingness-handling profile; it does not prove that earlier baseline tables were generated with this profile. Baseline-specific search grids and optimization budgets are recorded in `configs/baselines.json`; inner training-only selection is used. FT-Transformer and TabR use 11 features in this implementation. TabR retrieval and TabPFN context use only training records.

Supported HydroBERT variants are `full`, `no_reconstruction`, `no_dynamic_mask`, `text_only` and `numerical_only`. The numerical-only variant uses 11 mean-filled inputs without text, mask flags or auxiliary loss. Its input-layer shape differs from the older 16-input numerical-only implementation. The full model's architecture remains unchanged. Variant scores must be computed from actual runs.

## Curves, checkpoints and inference

When an inner validation set is provided, training exports prediction R² and reconstruction R² in addition to the three training losses. Reconstruction validation uses a fixed 60% artificial mask on known structural entries. R² is computed separately for each standardized descriptor and averaged over defined descriptors, with eligible-cell and defined-descriptor counts retained. These are explicit implementation definitions to be checked against the original Fig. S12 evaluation.

New checkpoints save model parameters, scalers, tokenizer, configuration, selected epochs and training identities. The inference command rejects overlap with checkpoint training materials.

```bash
python -m hydrobert.cli predict --data /path/external.csv --checkpoint /path/run/fold_01/checkpoint.pt --mask-rate 1 --output /path/external_predictions.csv
python -m hydrobert.cli metrics --predictions /path/run/predictions.csv --output /path/metrics.json
python -m hydrobert.cli plot --predictions /path/all_models_predictions.csv --output /path/new_masking_figures
```

`plot` produces the controlled masking comparison; `lopo-report` and `repeated-cv` produce their distinct publication/repetition-level summaries. The code does not silently pool repeated predictions as independent observations.

## Source layout

`hydrobert/models/`: network components; `prompts.py`, `masking.py`, `dataset.py`: inputs; `engine.py`: training; `repeated.py`: repeated CV; `metrics.py`, `reporting.py`: metric definitions; `plots.py`: figures; `cli.py`: entry points.

See `THIRD_PARTY_NOTICES.md` for third-party attribution. Keep private numerical data and trained outputs outside the source package unless their distribution has been authorized.
