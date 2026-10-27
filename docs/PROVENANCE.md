# Source and result provenance

The `paper-results` tag identifies the public package layout. The original evaluated implementation is preserved at commit `e82f38659ddd142f06451e46c36c6e50ab421a8f` in Git history. The current layout promotes that implementation to the repository root and uses `training/` for experiment entry points.

Model, loss, data-loader and preparation logic are unchanged. Changes to experiment files rename imports, source-snapshot paths and documentation. The protocol identifier is renamed; scientific settings are unchanged. A package-layout check prevents sequential training from silently combining source manifests from different layouts.

`records/source_migration.json` lists original and current paths and hashes. `records/package_sha256.json` verifies the current source package. `records/formal_manifest.json` and all original evaluation archives are preserved, not rewritten to pretend that the original runs used the new paths. Historical paths and identifiers inside those records therefore remain exactly as recorded. This package is not byte-for-byte identical to the original experiment source snapshot.

Use fresh output directories with the current commands. To inspect an original run or its exact source bytes, use its archived source snapshot or the original Git commit. Published metrics are unchanged; no new accuracy, training-time or GPU-latency claims follow from this packaging change.
