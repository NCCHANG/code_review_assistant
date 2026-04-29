"""
Generate train_50k.csv and val_5k.csv from the raw CTSSB-1M shards.

Run from the project root:
    python training/t5/prepare_large_dataset.py

Output files (rename to train.csv / val.csv before uploading to Colab):
    training/t5/data/train_50k.csv
    training/t5/data/val_5k.csv
"""
import gzip, hashlib, json, logging, os, random, sys
from glob import glob

NUM_TRAIN = 50_000
NUM_VAL   = 5_000
SEED      = 42
DATA_DIR  = "training/t5/data"
RAW_DIR   = os.path.join(DATA_DIR, "raw/ctssb_data_1M")
OUT_TRAIN = os.path.join(DATA_DIR, "train_50k.csv")
OUT_VAL   = os.path.join(DATA_DIR, "val_5k.csv")
MAX_CHARS = 400

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler(sys.stdout)])


def md5(s):
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def extract_from_diff(diff):
    before, after = [], []
    for line in diff.splitlines():
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("-"):
            before.append(line[1:])
        elif line.startswith("+"):
            after.append(line[1:])
    return "\n".join(before).strip(), "\n".join(after).strip()


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
                if not sp or sp == "SINGLE_STMT":
                    continue
                if not obj.get("likely_bug"):
                    continue

                buggy, fixed = extract_from_diff(obj.get("diff", ""))
                if not buggy or not fixed or buggy == fixed:
                    continue
                if len(buggy) > MAX_CHARS:
                    continue

                h = md5(buggy)
                if h in seen:
                    continue
                seen.add(h)
                rows.append({"input_text": buggy, "target_text": fixed,
                             "sstub_pattern": sp, "_hash": h})

    logging.info(f"Total usable rows after filtering: {len(rows)}")
    return rows


def write_csv(rows, path):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["input_text", "target_text", "sstub_pattern"])
        w.writeheader()
        for r in rows:
            w.writerow({"input_text": r["input_text"],
                        "target_text": r["target_text"],
                        "sstub_pattern": r["sstub_pattern"]})


def main():
    rows = load_and_filter()

    rng = random.Random(SEED)
    rng.shuffle(rows)

    if len(rows) < NUM_TRAIN + NUM_VAL:
        logging.warning(f"Only {len(rows)} rows available — taking all.")

    train_rows = rows[:NUM_TRAIN]
    train_hashes = {r["_hash"] for r in train_rows}
    val_pool = [r for r in rows[NUM_TRAIN:] if r["_hash"] not in train_hashes]
    val_rows = val_pool[:NUM_VAL]

    write_csv(train_rows, OUT_TRAIN)
    write_csv(val_rows, OUT_VAL)

    logging.info(f"Wrote {len(train_rows)} rows -> {OUT_TRAIN}")
    logging.info(f"Wrote {len(val_rows)} rows  -> {OUT_VAL}")

    from collections import Counter
    dist = Counter(r["sstub_pattern"] for r in train_rows)
    logging.info("Train distribution (top 10):")
    for pat, n in dist.most_common(10):
        logging.info(f"  {pat}: {n}")

    logging.info("Done. Rename files to train.csv / val.csv before uploading to Colab.")


if __name__ == "__main__":
    main()
