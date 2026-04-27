# CodeT5-small Fine-Tuning

## How to run
1. Activate the project venv from the repo root: `source .venv/bin/activate`
2. Install requirements: `pip install torch transformers pandas datasets requests`
3. Run: `python training/t5/train_codet5.py`
4. Training will run overnight (~4-6 hours on CPU)
5. Model saves automatically to `training/t5/saved_model/`
6. Repairer.py will automatically load the fine-tuned model on next app start

## Resume if interrupted
Just re-run the same command. It will detect the latest checkpoint
in `training/t5/checkpoints/` and resume from there automatically.

## Dataset
- Source: CTSSB-1M (Richter & Wehrheim, MSR 2022)
- Downloaded automatically from Zenodo (record 6526890) on first run
- 10,000 training samples, 1,000 validation samples
- Filtered for confirmed SStuB pattern bugs only
  - non-empty `sstub_pattern`
  - `likely_bug == True` when present
  - 3-40 line functions
  - <= 400 chars buggy code
  - buggy != fixed
  - de-duplicated by md5(buggy)
- Stratified train mix:
  - `wrong_binary_operator`: 4000
  - `change_identifier_used`: 3000
  - other patterns (proportional): 3000
- Val: 100 each from the top-10 most frequent patterns (no train overlap)

## Expected training time
- CPU (i5-12400F): ~4-6 hours for 3 epochs
- T4 GPU (Colab): ~30-40 minutes for 3 epochs

## Output
- `training/t5/saved_model/` — fine-tuned model (loaded by Repairer.py)
- `training/t5/checkpoints/` — intermediate checkpoints for resume (most recent 3 kept)
- `training/t5/training.log` — full training log
- `training/t5/data/train.csv`, `val.csv` — prepared datasets
- `training/t5/data/raw/` — raw download cache

## Notes
- CPU only by design (no CUDA paths).
- Effective batch size = `batch_size * gradient_accumulation_steps` = 4 * 4 = 16.
- Inference prefix is `"fix: "` to match Repairer.py — do not change.
- The save directory is hardcoded in Repairer.py; do not rename `saved_model/`.
