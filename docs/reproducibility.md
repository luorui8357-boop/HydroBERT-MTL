# Protocol correspondence and data availability

This release implements the stated model architecture and supplies explicit evaluation workflows. It does not certify that newly generated predictions reproduce historical scores.

## Confirmed settings

The architecture, primary hyperparameters, mask handling and auxiliary-loss eligibility follow Sections 2.1–2.4. The author confirmed that each repetition's R² is the arithmetic mean of the five fold-specific R² values. The supplied metadata have 2,268 records, 51 materials and 12 source publications.

## Settings that require experiment records

1. The epoch/checkpoint rule producing R²=0.904 remains unspecified. Training requires an explicit checkpoint policy. A maximum of 50 epochs does not establish that epoch 50 was the reported checkpoint.
2. The historical 100 repeated material partitions are not present. The new repeated-CV implementation shuffles material IDs; its results must be computed afresh or checked against archived assignments. `fold_mean` reports the mean of per-fold R² and per-fold RMSE; confirm the RMSE convention independently if the original analysis differed.
3. Table S4's within-publication R² definition must be selected explicitly. Its RMSE implementation follows the manuscript's per-material arithmetic mean, followed by equal publication aggregation.
4. Original baseline and ablation outputs must be matched to their actual configurations. The 11-input mean-imputation baseline profile and no-mask-indicator numerical ablation cannot be assigned historical scores without verification.
5. Validation reconstruction R² uses a declared masked-known-entry, per-descriptor definition. The original reconstruction-curve scoring convention must be checked before claiming exact Fig. S12 reproduction.

## Isotherm metadata discrepancy

For DOI `10.1016/j.carbon.2009.01.001`, the supplied metadata contain 970 records, 9 material IDs and 15 distinct isotherm IDs. Six materials have two temperature groups; M050 has only T77, and M035/M047 have only T308. The response-letter statement of 18 isotherms is not supported by these metadata alone. Verify source curves and identifiers; no identifiers or research records have been invented to force agreement.

## Public package scope

Available: source code, prompt generation, masking rules, architecture/configuration, identifiers and single-run split manifests, metric/reporting definitions. Not included: numerical research data, target values, historical trained weights and prediction files. A data-availability statement must describe this scope accurately. The metadata cannot substitute for the numerical data when reproducing reported performance.
