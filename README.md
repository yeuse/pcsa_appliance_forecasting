# PCSA appliance power forecasting

## Revision 3 (R3)

The frozen code and evaluation records for the third manuscript revision are in **[r3/](r3/README.md)**. Use this directory for the current experiments, rather than the legacy code at the repository root.

| Source Home 7951 | Macro MAE, kW (mean ± sample SD; seeds 7, 42, 123) |
|---|---:|
| PCSA | 0.071916 ± 0.003295 |
| Aggregate TCN | 0.082412 ± 0.001770 |
| Seq2Seq-NILM–TCN | 0.075415 ± 0.004827 |

PCSA has the lowest mean MAE in three of twelve target settings, not all homes. Retrained component ablations show a mean benefit from residual correction, but removing future state gating slightly lowers overall MAE. See the complete records and limitations in the R3 documentation.

## Earlier revisions

The root-level implementation, configurations and `results/` are retained for archival comparison. [The earlier README](README_R2_ARCHIVE.md) describes their limitations. They must not be combined with R3 results. Git history preserves the prior public versions.
