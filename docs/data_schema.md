# Input data schema

| Column | Unit | Role |
|---|---|---|
| C(wt%), H(wt%), N(wt%), O(wt%) | wt.% | Elemental descriptors, observed |
| S_BET(m2/g) | m²/g | Structural descriptor |
| V_t(cm3/g), V_m(cm3/g), V_um(cm3/g), V_meso(cm3/g) | cm³/g | Structural descriptors |
| Temp(K) | K | Observed operating temperature |
| Pressure(bar) | bar | Observed equilibrium pressure |
| H2_Uptake(wt%) | wt.% | Training/evaluation label; never a model input |
| Record_ID | identifier | Stable unique adsorption record |
| Material_ID | identifier | Verified material identity shared across all thermodynamic states |
| Source_DOI | DOI | Publication provenance; required for LOPO |
| Isotherm_ID | identifier | Optional isotherm identifier |

Missing structural values must be NaN/blank, not arbitrary zeros. Numeric zero is an observed value. The caller is responsible for confirming uptake conventions and units from source publications. The importer makes no automatic assumptions about excess/absolute uptake or unit conversions.

`record_metadata.csv` retains the original row order with opaque IDs only. Source studies have 970, 60, 40, 200, 20, 250, 150, 84, 24, 140, 90 and 240 records, respectively (2,268 total), and 9, 3, 4, 5, 2, 5, 3, 2, 2, 7, 3 and 6 materials (51 total). These counts are metadata, not independent sample counts. Several records can belong to one material/isotherm.
