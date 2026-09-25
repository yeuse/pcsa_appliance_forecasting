# Archived two-stage transfer results

These records use the earlier PISA two-stage adaptation protocol. They are
separate from the PCSA joint-head transfer comparison in the revised manuscript
and must not be substituted for its transfer table.

Generated from the completed 45-model evaluation (2026-09-21 16:40:38). Source home 7951; target homes 3039, 8386, 8565; seed 42. Units: kW; macro MAE is the unweighted mean across four appliances. Lower is better. All original and joint baselines are retained; no per-home baseline picking.

## Test results

| Target | Support | PISA | Original Aggregate TCN | Original Seq2Seq→TCN | Joint Aggregate TCN | Joint Seq2Seq→TCN |
|---|---|---:|---:|---:|---:|---:|
| 3039 | 1% | 0.259666 | 0.302270 | 0.228838 | 0.301552 | 0.202338 |
| 3039 | 5% | 0.238560 | 0.300225 | 0.198899 | 0.300225 | 0.201696 |
| 3039 | 10% | 0.216938 | 0.295636 | 0.207326 | 0.295636 | 0.216980 |
| 8386 | 1% | 0.033582 | 0.029313 | 0.032694 | 0.029313 | 0.033357 |
| 8386 | 5% | 0.033608 | 0.029483 | 0.033685 | 0.029483 | 0.033779 |
| 8386 | 10% | 0.033467 | 0.029252 | 0.033452 | 0.029252 | 0.033663 |
| 8565 | 1% | 0.291174 | 0.314950 | 0.265655 | 0.314129 | 0.266333 |
| 8565 | 5% | 0.274978 | 0.290888 | 0.228573 | 0.182106 | 0.202984 |
| 8565 | 10% | 0.253228 | 0.199852 | 0.214036 | 0.171820 | 0.183416 |

## Validation results

| Target | Support | PISA | Original Aggregate TCN | Original Seq2Seq→TCN | Joint Aggregate TCN | Joint Seq2Seq→TCN |
|---|---|---:|---:|---:|---:|---:|
| 3039 | 1% | 0.193435 | 0.219463 | 0.151431 | 0.218934 | 0.126419 |
| 3039 | 5% | 0.177496 | 0.218654 | 0.121685 | 0.218654 | 0.116384 |
| 3039 | 10% | 0.156147 | 0.217704 | 0.114557 | 0.217704 | 0.122898 |
| 8386 | 1% | 0.035597 | 0.035717 | 0.036107 | 0.035717 | 0.035260 |
| 8386 | 5% | 0.035693 | 0.035897 | 0.035257 | 0.035897 | 0.035277 |
| 8386 | 10% | 0.035346 | 0.035637 | 0.035157 | 0.035637 | 0.035252 |
| 8565 | 1% | 0.280005 | 0.334965 | 0.276188 | 0.334658 | 0.276753 |
| 8565 | 5% | 0.252339 | 0.309409 | 0.209853 | 0.182256 | 0.155659 |
| 8565 | 10% | 0.223169 | 0.200315 | 0.175024 | 0.169423 | 0.135987 |

## Interpretation and limits

- At 10% support, PISA and joint Seq2Seq are nearly tied on 3039 (0.216938 vs 0.216980) and close on 8386 (0.033467 vs 0.033663). Tiny differences are not evidence of statistical superiority.
- On 8565 at 10%, PISA is worse than joint Seq2Seq (0.253228 vs 0.183416) and joint Aggregate TCN (0.171820).
- Lower overall MAE does not imply good event forecasts: the 8565 audit identifies substantial ON/OFF trade-offs and near-zero internal event detection.
- This is one fixed source to three targets, not four-fold LOHO; one seed provides no estimate of training-seed uncertainty. Do not use overlapping windows as independent replicates for significance claims.
- Adaptation budgets and supervision differ. Original and joint baselines are separate protocol variants, not interchangeable results.
- Checkpoint selection used validation. Test results have now been inspected during development, so subsequent tuning cannot claim a fresh untouched test set.

## Corrected event audit (8565, 10% only)

Internal transitions exclude the first forecast point; both transition endpoints must be valid. Values below are percentages for postprocessed states (minimum ON/OFF 3/2), ±2-minute directional matching.

| Model | Air start precision | Air start recall | Air start F1 | Air stop precision | Air stop recall | Air stop F1 |
|---|---:|---:|---:|---:|---:|---:|
| PISA | 0.1757 | 1.7442 | 0.3193 | 0.5157 | 4.4715 | 0.9248 |
| Joint Seq2Seq→TCN | 0 | 0 | 0 | 0 | 0 | 0 |
| Joint Aggregate TCN | 0 | 0 | 0 | 0 | 0 | 0 |

The three methods also have zero internal start/stop F1 for refrigerator, dishwasher and microwave in this audited setting. This does not excuse PISA's poor event performance. Full-window event metrics use explicitly declared history sources; true-history variants are diagnostics only. Counts refer to overlapping window positions, not unique events. Do not extend this single-setting audit to all three households.

## Provenance

Scalar data, selection labels and SHA256 fingerprints: `transfer_mae.json`.
Corrected events and validation-only fusion diagnostics: `event_audit_8565_10pct.json`.
Private absolute paths and original raw meter data are not included.
