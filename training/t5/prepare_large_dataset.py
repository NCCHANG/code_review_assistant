"""
Generate train.csv and val.csv from the raw CTSSB-1M shards,
keeping ONLY the three structurally-deterministic bug patterns:

  CHANGE_BOOLEAN_LITERAL   True <-> False swaps         (was 100% EM)
  CHANGE_UNARY_OPERATOR    add/remove/change unary op    (was  67% EM)
  CHANGE_BINARY_OPERATOR   +/-/*/< etc. swaps            (was  50% EM)

These are the only patterns whose correct fix can be inferred from code
structure alone — T5 achieves 74%+ EM on them combined, versus 18-20%
on the full 22-pattern mix (where most patterns require guessing
project-specific identifier names).

Run from the project root:
    python training/t5/prepare_large_dataset.py

Outputs:
    training/t5/data/train.csv
    training/t5/data/val.csv
"""
import gzip, hashlib, json, logging, os, random, sys
from glob import glob
from collections import Counter

TARGET_PATTERNS = {
    "CHANGE_BOOLEAN_LITERAL",
    "CHANGE_UNARY_OPERATOR",
    "CHANGE_BINARY_OPERATOR",
}

VAL_FRACTION = 0.15   # 15% of data goes to val
SEED         = 42
DATA_DIR     = "training/t5/data"
RAW_DIR      = os.path.join(DATA_DIR, "raw/ctssb_data_1M")
OUT_TRAIN    = os.path.join(DATA_DIR, "train.csv")
OUT_VAL      = os.path.join(DATA_DIR, "val.csv")
MAX_CHARS    = 400    # tighter limit: these patterns have short, clean snippets

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler(sys.stdout)])


def md5(s):
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def extract_from_diff(diff):
    input_parts, before, after = [], [], []
    for line in diff.splitlines():
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("-"):
            before.append(line[1:].rstrip())
            input_parts.append(line[1:].rstrip())
        elif line.startswith("+"):
            after.append(line[1:].rstrip())
        else:
            input_parts.append(line[1:].rstrip() if line.startswith(" ") else line.rstrip())
    return "\n".join(input_parts).strip(), "\n".join(before).strip(), "\n".join(after).strip()


def load_and_filter():
    shards = sorted(glob(os.path.join(RAW_DIR, "*.jsonl.gz")))
    if not shards:
        sys.exit(f"No shards found in {RAW_DIR}")

    rows, seen = [], set()
    for shard in shards:
        logging.info(f"Reading {os.path.basename(shard)} ...")
        with gzip.open(shard, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sp = obj.get("sstub_pattern", "")
                if sp not in TARGET_PATTERNS:
                    continue
                if not obj.get("likely_bug"):
                    continue

                input_with_context, buggy, fixed = extract_from_diff(obj.get("diff", ""))
                if not buggy or not fixed or buggy == fixed:
                    continue
                if len(input_with_context) > MAX_CHARS:
                    continue

                h = md5(buggy)
                if h in seen:
                    continue
                seen.add(h)
                rows.append({
                    "input_text":    input_with_context,
                    "target_text":   fixed,
                    "sstub_pattern": sp,
                    "_hash":         h,
                })

    logging.info(f"Total usable rows after filtering: {len(rows):,}")
    return rows


def write_csv(rows, path):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["input_text", "target_text", "sstub_pattern"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "input_text":    r["input_text"],
                "target_text":   r["target_text"],
                "sstub_pattern": r["sstub_pattern"],
            })


def main():
    rows = load_and_filter()
    if not rows:
        sys.exit("No rows produced — check RAW_DIR path.")

    rng = random.Random(SEED)
    rng.shuffle(rows)

    n_val   = max(500, int(len(rows) * VAL_FRACTION))
    n_train = len(rows) - n_val

    train_rows = rows[:n_train]
    val_rows   = rows[n_train:]

    write_csv(train_rows, OUT_TRAIN)
    write_csv(val_rows,   OUT_VAL)

    logging.info(f"Wrote {len(train_rows):,} train rows -> {OUT_TRAIN}")
    logging.info(f"Wrote {len(val_rows):,}  val rows  -> {OUT_VAL}")

    dist = Counter(r["sstub_pattern"] for r in train_rows)
    logging.info("Train pattern distribution:")
    for pat, n in dist.most_common():
        logging.info(f"  {pat}: {n:,}")

    logging.info("Done. Upload train.csv and val.csv to Colab.")


if __name__ == "__main__":
    main()
