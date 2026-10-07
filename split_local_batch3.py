#!/usr/bin/env python3
# split_local_batch3.py — 本地模型批 1103 条切分 + manifest（round3 续训用）
# 切分方案：75/10/15，seed 42，保留 id/dim/source manifest
import json, hashlib, os, random

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "data", "local_batch3.jsonl")
OUT_DIR = os.path.join(BASE, "data", "train_local_batch3")
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(42)

rows = []
with open(SRC, encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        rows.append({
            "id": o.get("id") or f"local_batch3:{line_no:04d}",
            "dim": o.get("dim") or "not_available",
            "source": o.get("source") or "local_batch3",
            "prompt": o["prompt"],
            "completion": o["completion"],
        })

# 打乱后切分
idx = list(range(len(rows)))
random.shuffle(idx)
n = len(rows)
n_train = int(n * 0.75)
n_valid = int(n * 0.10)
# 剩余给 test
train = [rows[i] for i in idx[:n_train]]
valid = [rows[i] for i in idx[n_train:n_train+n_valid]]
test  = [rows[i] for i in idx[n_train+n_valid:]]

def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

for name, data in [("train", train), ("valid", valid), ("test", test)]:
    with open(f"{OUT_DIR}/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}, ensure_ascii=False) + "\n")
    with open(f"{OUT_DIR}/{name}_manifest.jsonl", "w", encoding="utf-8") as mf:
        for r in data:
            mf.write(json.dumps({
                "id": r["id"],
                "dim": r["dim"],
                "source": r["source"],
                "split": name,
                "prompt_sha256": sha256_text(r["prompt"]),
                "completion_sha256": sha256_text(r["completion"]),
            }, ensure_ascii=False) + "\n")

print(f"total={n} train={len(train)} valid={len(valid)} test={len(test)}")
print(f"out_dir={OUT_DIR}")
