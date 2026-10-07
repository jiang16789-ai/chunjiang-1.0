#!/usr/bin/env python3
"""chunjiang1.0 数据准备：合并 flash 压缩语料 → 转 mlx-lm completions 格式 → dim 分层切分 75/10/15。
用法：python3 prepare_train.py [随机种子]
"""
import json, random, sys
from collections import defaultdict

ROOT = '/path/to/chunjiang1.0'
SRC = f'{ROOT}/data/corpus_root_5000.jsonl'
# 第二批：flash(短语料) + span(超长文) 两个压缩源合并，id 不重叠
COMP_FILES = [
    f'{ROOT}/data/compressed/flash_compressed.jsonl',
    f'{ROOT}/data/compressed/span_compressed.jsonl',
]
OUT_DIR = f'{ROOT}/data/train'
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
random.seed(seed)

# 1. 读压缩产物 {id: compressed}（合并 flash + span 两个源）
comp = {}
for cf in COMP_FILES:
    for line in open(cf, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        comp[o['id']] = o['compressed']

# 2. 读黄金根 {id: {dim, prompt, content}}
root = {}
for line in open(SRC, encoding='utf-8'):
    line = line.strip()
    if not line:
        continue
    o = json.loads(line)
    root[o['id']] = o

# 3. 有效语料：content 非空 且 已压缩（prompt 用原文，completion 用压缩版）
rows = []
skipped = 0
for oid, o in root.items():
    if not o.get('content'):
        skipped += 1  # 空题（13 条，后续补）
        continue
    if oid not in comp:
        skipped += 1  # 超长文还没压完（后台继续）
        continue
    rows.append({
        'id': oid,
        'dim': o.get('dim', '?'),
        'prompt': o['prompt'],
        'completion': comp[oid],
    })

print(f'有效语料 {len(rows)} 条（跳过 {skipped} 条：空题+未压完的超长文）')

# 4. 转 mlx-lm completions 格式
items = [{'prompt': r['prompt'], 'completion': r['completion']} for r in rows]

# 5. dim 分层切分 75/10/15（每个 dim 都覆盖三份）
groups = defaultdict(list)
for r in rows:
    groups[r['dim']].append(r)

train, valid, test = [], [], []
for dim, g in groups.items():
    random.shuffle(g)
    n = len(g)
    n_test = max(1, round(n * 0.15))
    n_valid = max(1, round(n * 0.10))
    n_train = n - n_test - n_valid
    if n_train < 1:  # 极小 dim 兜底
        n_test, n_valid, n_train = max(1, n - 2), 1, max(0, n - 2)
        if n == 1:
            n_test, n_valid, n_train = 0, 0, 1
    test += g[:n_test]
    valid += g[n_test:n_test + n_valid]
    train += g[n_test + n_valid:]

# 6. 落盘
import os
os.makedirs(OUT_DIR, exist_ok=True)
for name, data in [('train', train), ('valid', valid), ('test', test)]:
    p = f'{OUT_DIR}/{name}.jsonl'
    with open(p, 'w', encoding='utf-8') as f:
        for r in data:
            f.write(json.dumps({'prompt': r['prompt'], 'completion': r['completion']}, ensure_ascii=False) + '\n')
    print(f'{name}: {len(data)} 条 -> {p}')

# 7. 校验：切分后 dim 覆盖
print(f'\n切分统计：train {len(train)} / valid {len(valid)} / test {len(test)} = {len(train)+len(valid)+len(test)} 条')
for name, data in [('train', train), ('valid', valid), ('test', test)]:
    ds = len(set(r['dim'] for r in data))
    print(f'  {name} 覆盖 dim 数: {ds}/56')
