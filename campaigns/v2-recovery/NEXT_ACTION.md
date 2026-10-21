# Next action

1. Add the fresh-test-seed split to the generator and the PDE configs.
2. Regenerate the v2 datasets (dev train/val from the old dev seed,
   test from a FRESH seed never used during development).
3. Screen the lambda grid on the dev validation split only.
4. Freeze the v2 protocol, then run the 18 required matrix rows.
