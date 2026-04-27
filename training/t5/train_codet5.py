"""
Standalone CodeT5-small fine-tuning script for Python bug repair.

Reads CTSSB-1M from training/t5/data/raw/ctssb_data_1M/ (10 x jsonl.gz shards),
filters to high-quality SStuB-pattern bug fixes, and fine-tunes
Salesforce/codet5-small to map "fix: <buggy_code>" -> "<fixed_code>".

Output model is written to training/t5/saved_model/ which Repairer.py loads
automatically on next app start. Do NOT modify Repairer.py.

Run:
    python training/t5/train_codet5.py
"""

# ---------------------------------------------------------------------------
# Requirements check (runs before heavy imports so the error is fast & clear)
# ---------------------------------------------------------------------------
import importlib
import sys

_REQUIRED = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
]
_missing = []
for mod_name, pip_name in _REQUIRED:
    try:
        importlib.import_module(mod_name)
    except ImportError:
        _missing.append(pip_name)
if _missing:
    sys.stderr.write(
        "[train_codet5] Missing required packages: "
        + ", ".join(_missing)
        + "\nActivate your venv from the project root, then install:\n"
        + "    source .venv/bin/activate\n"
        + "    pip install " + " ".join(_missing) + "\n"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Heavy imports
# ---------------------------------------------------------------------------
import hashlib
import json
import logging
import os
import random
import time
from glob import glob

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONFIG = {
    "model_name": "Salesforce/codet5-small",
    "save_dir": "training/t5/saved_model",
    "checkpoint_dir": "training/t5/checkpoints",
    "data_dir": "training/t5/data",
    "train_file": "training/t5/data/train.csv",
    "val_file": "training/t5/data/val.csv",
    "max_input_length": 256,
    "max_target_length": 128,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 3,
    "learning_rate": 5e-5,
    "warmup_steps": 100,
    "save_every_n_steps": 500,
    "num_train_samples": 10000,
    "num_val_samples": 1000,
    "seed": 42,
    "prefix": "fix: ",
}

# Path to the locally extracted CTSSB-1M shards (file-0.jsonl.gz … file-9.jsonl.gz)
LOCAL_CTSSB_DIR = "training/t5/data/raw/ctssb_data_1M"

# Module-level start time so train() can compute ETA without thread state.
start_time = time.time()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging():
    os.makedirs(os.path.dirname(CONFIG["train_file"]), exist_ok=True)
    log_path = os.path.join(os.path.dirname(CONFIG["save_dir"]), "training.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Wipe any existing handlers so reruns don't duplicate output.
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def _iter_jsonl(path: str):
    if path.endswith(".gz"):
        import gzip
        opener = lambda: gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    else:
        opener = lambda: open(path, "r", encoding="utf-8", errors="ignore")
    with opener() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _extract_from_diff(diff: str):
    """Parse unified diff to extract (buggy_line, fixed_line) as actual Python code."""
    before_lines, after_lines = [], []
    for line in diff.splitlines():
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("-"):
            before_lines.append(line[1:])
        elif line.startswith("+"):
            after_lines.append(line[1:])
    before = "\n".join(before_lines).strip()
    after = "\n".join(after_lines).strip()
    return before, after


def _filter_records(records):
    """Apply spec filters in order. Returns list of dicts and logs survivor counts."""
    n0 = len(records)
    logging.info(f"Loaded {n0} raw records")

    # (a) sstub_pattern present, non-empty, and not the catch-all default
    step = []
    for r in records:
        sp = r.get("sstub_pattern")
        if not sp or not isinstance(sp, str) or sp.strip() in ("", "SINGLE_STMT"):
            continue
        step.append(r)
    logging.info(f"After (a) specific sstub_pattern (not SINGLE_STMT): {len(step)}")

    # (b) likely_bug == True
    step = [r for r in step if r.get("likely_bug") is True]
    logging.info(f"After (b) likely_bug=True: {len(step)}")

    # (c) extract actual code from diff; both sides must be non-empty
    rebuilt = []
    for r in step:
        diff = r.get("diff", "")
        if not diff:
            continue
        buggy, fixed = _extract_from_diff(diff)
        if not buggy or not fixed:
            continue
        rebuilt.append((r, buggy, fixed))
    logging.info(f"After (c) parseable diff: {len(rebuilt)}")

    # (d) buggy != fixed
    rebuilt = [(r, b, f) for r, b, f in rebuilt if b.strip() != f.strip()]
    logging.info(f"After (d) buggy != fixed: {len(rebuilt)}")

    # (e) buggy length <= 400 chars
    rebuilt = [(r, b, f) for r, b, f in rebuilt if len(b) <= 400]
    logging.info(f"After (e) buggy <= 400 chars: {len(rebuilt)}")

    # (f) dedupe by md5 of buggy
    seen = set()
    deduped = []
    for r, b, f in rebuilt:
        h = _md5(b)
        if h in seen:
            continue
        seen.add(h)
        deduped.append({
            "input_text": b,
            "target_text": f,
            "sstub_pattern": r.get("sstub_pattern"),
            "_hash": h,
        })
    logging.info(f"After (f) dedup by md5(buggy): {len(deduped)}")
    return deduped


def _stratified_sample(rows, target_total, target_counts, rng):
    """rows: list of dicts with sstub_pattern; returns selected rows."""
    by_pat = {}
    for r in rows:
        by_pat.setdefault(r["sstub_pattern"], []).append(r)
    for pat in by_pat:
        rng.shuffle(by_pat[pat])

    selected = []
    used_pats = set()

    # Take the explicit per-pattern targets first.
    for pat, want in target_counts.items():
        pool = by_pat.get(pat, [])
        take = min(want, len(pool))
        selected.extend(pool[:take])
        by_pat[pat] = pool[take:]
        used_pats.add(pat)

    # Fill remainder proportionally from "everything else".
    remainder = target_total - len(selected)
    if remainder > 0:
        leftover = [(p, lst) for p, lst in by_pat.items() if p not in used_pats and lst]
        total_other = sum(len(lst) for _, lst in leftover)
        if total_other > 0:
            # Proportional allocation, then top up with random fill if rounding leaves a gap.
            allocations = []
            for p, lst in leftover:
                share = int(round(remainder * len(lst) / total_other))
                allocations.append([p, lst, min(share, len(lst))])
            current = sum(a[2] for a in allocations)
            # adjust to hit remainder exactly
            i = 0
            while current < remainder and any(a[2] < len(a[1]) for a in allocations):
                if allocations[i][2] < len(allocations[i][1]):
                    allocations[i][2] += 1
                    current += 1
                i = (i + 1) % len(allocations)
            while current > remainder:
                for a in allocations:
                    if a[2] > 0:
                        a[2] -= 1
                        current -= 1
                        if current == remainder:
                            break
            for p, lst, n in allocations:
                selected.extend(lst[:n])
                by_pat[p] = lst[n:]
    return selected, by_pat


def prepare_dataset():
    os.makedirs(CONFIG["data_dir"], exist_ok=True)

    if os.path.exists(CONFIG["train_file"]) and os.path.exists(CONFIG["val_file"]):
        logging.info("Dataset already prepared — skipping.")
        return

    # --- Step 1: locate local CTSSB-1M shards ---
    shard_paths = sorted(glob(os.path.join(LOCAL_CTSSB_DIR, "*.jsonl.gz")))
    if not shard_paths:
        logging.error(
            f"No *.jsonl.gz shards found in {LOCAL_CTSSB_DIR}. "
            "Download CTSSB-1M from https://zenodo.org/records/10217373 and "
            f"extract the shards to {LOCAL_CTSSB_DIR}/"
        )
        sys.exit(1)
    logging.info(f"Found {len(shard_paths)} shards in {LOCAL_CTSSB_DIR}")

    # --- Step 2: stream + filter (no need to hold all records in RAM at once) ---
    records = []
    for p in shard_paths:
        logging.info(f"Reading {os.path.basename(p)} ...")
        for rec in _iter_jsonl(p):
            records.append(rec)

    rows = _filter_records(records)
    if len(rows) < CONFIG["num_train_samples"] + CONFIG["num_val_samples"]:
        logging.warning(
            f"Only {len(rows)} rows survived filtering; will take what's available."
        )

    # --- Step 3: stratified sampling ---
    rng = random.Random(CONFIG["seed"])
    rng.shuffle(rows)

    # CTSSB-1M uses UPPERCASE pattern names.
    train_targets = {
        "CHANGE_BINARY_OPERATOR": 4000,
        "CHANGE_IDENTIFIER_USED": 3000,
    }
    train_total = CONFIG["num_train_samples"]
    train_rows, leftover_by_pat = _stratified_sample(rows, train_total, train_targets, rng)
    train_hashes = {r["_hash"] for r in train_rows}

    # Val: 100 from each of the top 10 most frequent patterns AFTER train removal.
    val_pool = [r for r in rows if r["_hash"] not in train_hashes]
    pat_counts = {}
    for r in val_pool:
        pat_counts[r["sstub_pattern"]] = pat_counts.get(r["sstub_pattern"], 0) + 1
    top10 = [p for p, _ in sorted(pat_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
    val_rows = []
    by_pat_val = {}
    for r in val_pool:
        by_pat_val.setdefault(r["sstub_pattern"], []).append(r)
    for pat in top10:
        pool = by_pat_val.get(pat, [])
        rng.shuffle(pool)
        take = min(100, len(pool))
        val_rows.extend(pool[:take])

    # If the top10 didn't yield 1000, top up with any remaining val_pool rows.
    if len(val_rows) < CONFIG["num_val_samples"]:
        chosen = {r["_hash"] for r in val_rows}
        extras = [r for r in val_pool if r["_hash"] not in chosen]
        rng.shuffle(extras)
        need = CONFIG["num_val_samples"] - len(val_rows)
        val_rows.extend(extras[:need])

    # Sanity: no overlap.
    val_hashes = {r["_hash"] for r in val_rows}
    assert train_hashes.isdisjoint(val_hashes), "train/val overlap detected"

    # --- Step 4: save ---
    cols = ["input_text", "target_text", "sstub_pattern"]
    train_df = pd.DataFrame([{k: r[k] for k in cols} for r in train_rows])
    val_df = pd.DataFrame([{k: r[k] for k in cols} for r in val_rows])
    train_df.to_csv(CONFIG["train_file"], index=False)
    val_df.to_csv(CONFIG["val_file"], index=False)
    logging.info(f"Wrote {len(train_df)} train rows -> {CONFIG['train_file']}")
    logging.info(f"Wrote {len(val_df)} val rows   -> {CONFIG['val_file']}")

    logging.info("Train class distribution:")
    for pat, c in train_df["sstub_pattern"].value_counts().items():
        logging.info(f"  {pat}: {c}")
    logging.info("Val class distribution:")
    for pat, c in val_df["sstub_pattern"].value_counts().items():
        logging.info(f"  {pat}: {c}")

    logging.info("--- 3 random examples (visual sanity check) ---")
    for i in rng.sample(range(len(train_df)), min(3, len(train_df))):
        row = train_df.iloc[i]
        logging.info(f"[ex {i}] sstub_pattern={row['sstub_pattern']}")
        logging.info(f"  INPUT:\n{row['input_text']}")
        logging.info(f"  TARGET:\n{row['target_text']}")


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------
class BugFixDataset(Dataset):
    def __init__(self, dataframe, tokenizer, config):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.config["prefix"] + str(self.data.iloc[idx]["input_text"])
        target_text = str(self.data.iloc[idx]["target_text"])

        inputs = self.tokenizer(
            input_text,
            max_length=self.config["max_input_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            target_text,
            max_length=self.config["max_target_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = targets["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
_CKPT_GLOB = "checkpoint_epoch*_step*.pt"


def _list_checkpoints(checkpoint_dir):
    paths = glob(os.path.join(checkpoint_dir, _CKPT_GLOB))
    paths.sort(key=lambda p: os.path.getmtime(p))
    return paths


def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, config):
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "config": config,
    }
    path = os.path.join(config["checkpoint_dir"], f"checkpoint_epoch{epoch}_step{step}.pt")
    torch.save(checkpoint, path)
    logging.info(f"Checkpoint saved: {path}")

    existing = _list_checkpoints(config["checkpoint_dir"])
    while len(existing) > 3:
        old = existing.pop(0)
        try:
            os.remove(old)
            logging.info(f"Removed old checkpoint: {old}")
        except OSError as e:
            logging.warning(f"Could not remove {old}: {e}")


def load_latest_checkpoint(model, optimizer, scheduler, config):
    existing = _list_checkpoints(config["checkpoint_dir"])
    if not existing:
        logging.info("No checkpoint found — starting fresh.")
        return 0, 0
    latest = existing[-1]
    logging.info(f"Resuming from checkpoint: {latest}")
    ckpt = torch.load(latest, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["step"]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, val_loader, tokenizer, device, config):
    model.eval()
    total_loss = 0.0
    n_loss_batches = 0
    exact_matches = 0
    total_generated = 0
    GEN_BATCHES = 50

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            n_loss_batches += 1

            if i < GEN_BATCHES:
                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=config["max_target_length"],
                )
                # Restore -100 -> pad before decoding the labels.
                labels_for_decode = labels.clone()
                labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
                preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                tgts = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
                for p, t in zip(preds, tgts):
                    if p.strip() == t.strip():
                        exact_matches += 1
                    total_generated += 1

    avg_loss = total_loss / max(n_loss_batches, 1)
    exact_match_pct = (exact_matches / total_generated * 100) if total_generated else 0.0
    model.train()
    return avg_loss, exact_match_pct


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train():
    global start_time
    start_time = time.time()

    _set_seeds(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    prepare_dataset()
    train_df = pd.read_csv(CONFIG["train_file"])
    val_df = pd.read_csv(CONFIG["val_file"])
    logging.info(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")

    logging.info(f"Loading {CONFIG['model_name']} ...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = T5ForConditionalGeneration.from_pretrained(CONFIG["model_name"])
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    train_dataset = BugFixDataset(train_df, tokenizer, CONFIG)
    val_dataset = BugFixDataset(val_df, tokenizer, CONFIG)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])

    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    optimizer_steps_per_epoch = max(len(train_loader) // CONFIG["gradient_accumulation_steps"], 1)
    total_optimizer_steps = optimizer_steps_per_epoch * CONFIG["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG["warmup_steps"],
        num_training_steps=total_optimizer_steps,
    )

    start_epoch, start_step = load_latest_checkpoint(model, optimizer, scheduler, CONFIG)

    global_step = start_step
    best_val_loss = float("inf")

    for epoch in range(start_epoch, CONFIG["num_epochs"]):
        logging.info(f"--- EPOCH {epoch+1}/{CONFIG['num_epochs']} ---")
        model.train()
        epoch_loss = 0.0
        seen_batches = 0

        # Approximate resume: skip already-done batches inside the resuming epoch.
        skip_until = (start_step % len(train_loader)) if epoch == start_epoch else 0

        for step, batch in enumerate(train_loader):
            if step < skip_until:
                continue

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / CONFIG["gradient_accumulation_steps"]
            loss.backward()

            if (step + 1) % CONFIG["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step > 0 and global_step % CONFIG["save_every_n_steps"] == 0:
                    save_checkpoint(
                        model, optimizer, scheduler,
                        epoch, global_step, loss.item(), CONFIG,
                    )

            epoch_loss += loss.item() * CONFIG["gradient_accumulation_steps"]
            seen_batches += 1

            if step % 50 == 0:
                elapsed = time.time() - start_time
                if global_step > 0:
                    eta_seconds = (elapsed / global_step) * (total_optimizer_steps - global_step)
                    eta_hours = eta_seconds / 3600
                else:
                    eta_hours = float("nan")
                logging.info(
                    f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | "
                    f"Loss: {loss.item()*CONFIG['gradient_accumulation_steps']:.4f} | "
                    f"ETA: {eta_hours:.1f}h"
                )

        avg_train_loss = epoch_loss / max(seen_batches, 1)
        val_loss, exact_match = evaluate(model, val_loader, tokenizer, device, CONFIG)
        logging.info(
            f"Epoch {epoch+1} complete | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Exact Match: {exact_match:.1f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(CONFIG["save_dir"])
            tokenizer.save_pretrained(CONFIG["save_dir"])
            logging.info(f"New best model saved to {CONFIG['save_dir']}")

    logging.info("Training complete!")
    logging.info(f"Best val loss: {best_val_loss:.4f}")
    logging.info(f"Model saved to: {CONFIG['save_dir']}")
    logging.info("You can now run the app — Repairer.py will load the fine-tuned model.")


if __name__ == "__main__":
    setup_logging()
    logging.info("CONFIG: " + json.dumps(CONFIG, indent=2))
    start_time = time.time()
    if "--prepare-only" in sys.argv:
        logging.info("--prepare-only: running data preparation then exiting.")
        prepare_dataset()
        logging.info("Data ready. Check training/t5/data/train.csv and val.csv.")
        sys.exit(0)
    train()
