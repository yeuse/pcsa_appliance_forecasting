# Before GitHub publication

- [ ] Author/coauthor approval and chosen LICENSE.
- [ ] Verify provenance and redistribution permissions for every included file.
- [ ] Add real repository URL only after creation; no publication has occurred.
- [ ] Run tests and CLI help from this directory, not the parent workspace.
- [x] Add the saved base, history-refiner and seed-7 residual-TCN configs with
  a repository-relative training command for each stage.
- [ ] Recover the seed-123 source config and verify all three reported seeds in
  a clean real-data evaluation.
- [ ] Recover server dependency/GPU information.
- [ ] Confirm data-access instructions; do not commit household CSV files.
- [ ] Decide whether selected/source checkpoint weights can be released.
- [ ] Run a clean real-data evaluation and record its manifest.
- [ ] Check manuscript architecture and method names against configs/observed.
- [ ] Audit `git status` / staged files; publish only this candidate directory.
- [ ] Add an immutable version tag after final review.

Do not upload the parent workspace. `.gitignore` is a convenience, not a privacy
guarantee. Do not call a staging candidate a public code release in the response.
