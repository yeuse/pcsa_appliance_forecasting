# Candidate verification — 2026-09-21

## Portable source training chain — 2026-09-25

- Added the author-supplied base, historical-refiner and seed-7 residual-TCN
  configurations with repository-relative paths. Their checkpoint references
  form the recorded source initialization chain: two conceptual stages and
  three optimizer runs, including an auxiliary pass in Stage I.
- Added `scripts/train_pcsa_from_config.py` to translate saved configurations
  into the original training CLI. All three dry runs completed and their
  generated arguments parsed successfully. The runner checks input CSV and
  initialization checkpoint paths before a real training run.
- Added official Pecan Street Dataport access links. No raw measurements or
  checkpoint binaries were added to this directory.
- The full synthetic test suite passed: 23 tests in the Python 3.11 CPU
  environment. No household-data training or score reproduction was run.

## PCSA README and Figure 2 update — 2026-09-25

- Aligned the README's method name, source-home results, transfer scope and
  limitations with the R2 manuscript. The archived two-stage transfer results
  are identified separately from the manuscript's joint-head comparison.
- Copied the manuscript's Figure 2 into `figures/Figure_2.png`; its SHA-256
  matches `paper/revision_v2/Figure_2.png` in the author workspace.
- Simplified stale or prescriptive code comments without changing executable
  statements. Python compilation passed, and all 19 synthetic unit tests passed
  in the Python 3.11 environment with torch 2.2.2+cpu, NumPy 1.26.4 and pandas
  2.2.3. No real-data training or inference evaluation was run.

## Earlier candidate checks — 2026-09-21

- Ran from this candidate directory, not the parent source tree.
- Python 3.11 / PyTorch 2.2.2+cpu / numpy 1.26.4 / pandas 2.2.3.
- `python -m unittest discover -s tests -p 'test_*.py' -v`: 19 tests passed.
- Six CLI `--help` checks passed: source PISA training, genuine two-stage source
  training, PISA two-stage transfer, baseline transfer, selected-checkpoint
  evaluation, and four-home data preparation.
- Some synthetic empty-group metric warnings are expected in existing metric
  code and did not fail tests.
- Tests cover strict small-model checkpoint loading, validation-reproduction
  guards, selection-source fallback, masking and directional event boundaries,
  and power-metric agreement. They do not constitute a full household-data run.
- No training or real-data evaluation was started from this candidate.
- Known private path patterns and common private-key markers were scanned;
  this is not a comprehensive security/legal audit.
- Candidate contains no meter CSV files, checkpoint binaries or ZIP archives.
- `MANIFEST.sha256` records candidate source/document/config/result file hashes.

Full source lineage and server environment are still required. The license and
GitHub publication status remain pending author approval.
