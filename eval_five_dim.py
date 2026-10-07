#!/usr/bin/env python3
"""chunjiang1.0 五维评测（可实测部分）。

论文第 4 章设计的五维：
  D1 结论类别 macro-F1      -> 本 test 集无「买/卖/持」标注体系，设计未实现（见报告）
  D2 关键因子加权召回        -> 无因子权重表，设计未实现
  D3 无依据因子率            -> 实测：生成答案中「不在题面出现」的数字占比
  D4 证据忠实度              -> 实测：生成答案数字中可追溯到题面的比例
  D5 LLM-as-a-Judge         -> 未做人工双标校准子集，设计未实现

D3/D4 的口径：数字 token 正则 r'-?\\d+(?:\\.\\d+)?%?'；
  题面数字集合 P（来自 prompt），生成数字集合 G，参考答案数字集合 R。
  证据忠实度 = |G∩P| / |G|；无依据率 = 1 - 忠实度；
  参考答案数字召回 = |G∩R| / |R|（作为「是否有依据接近参考答案」的代理，非金标）。

只读 data/train_local_batch3/test.jsonl；不写任何训练/评测数据。
输出：eval/five_dim_gen.jsonl（逐条）+ eval/five_dim_summary.json
"""
import argparse, json, os, re, time

import mlx.core as mx
from mlx_lm import batch_generate, load

ROOT = os.path.dirname(os.path.abspath(__file__))
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?%?")


def nums(s):
    return set(NUM_RE.findall(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="~/.cache/huggingface/hub/models--mlx-community--Qwen3-8B-4bit/snapshots/545dc4251c05440727734bcd94334791f6ab0192")
    ap.add_argument("--adapter", default=os.path.join(ROOT, "adapters/chunjiang1.0-batch3-resume-r8-lr2e5"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data/train_local_batch3/test.jsonl"))
    ap.add_argument("--limit", type=int, default=166)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/five_dim_gen.jsonl"))
    ap.add_argument("--summary", default=os.path.join(ROOT, "eval/five_dim_summary.json"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()][: args.limit]
    print(f"loaded {len(rows)} test items", flush=True)

    model, tok = load(args.model, adapter_path=args.adapter)
    model.eval()

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out, encoding="utf-8"):
            if l.strip():
                done.add(json.loads(l)["i"])
    print(f"resume: {len(done)} already done", flush=True)

    t0 = time.time()
    fh = open(args.out, "a", encoding="utf-8")
    per = []
    for start in range(0, len(rows), args.chunk):
        idxs = [i for i in range(start, min(start + args.chunk, len(rows))) if i not in done]
        if not idxs:
            continue
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": rows[i]["prompt"]}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
            for i in idxs
        ]
        resp = batch_generate(model, tok, prompts, max_tokens=args.max_tokens, verbose=False)
        for j, i in enumerate(idxs):
            gen = resp.texts[j]
            ref = rows[i]["completion"]
            pr = rows[i]["prompt"]
            P, G, R = nums(pr), nums(gen), nums(ref)
            faithful = len(G & P) / len(G) if G else None
            ref_recall = len(G & R) / len(R) if R else None
            ref_faithful = len(R & P) / len(R) if R else None
            rec = {
                "i": i, "n_prompt_nums": len(P), "n_gen_nums": len(G), "n_ref_nums": len(R),
                "faithfulness_D4": faithful, "fabrication_rate_D3": (1 - faithful) if faithful is not None else None,
                "ref_num_recall_proxy": ref_recall, "ref_faithfulness_baseline": ref_faithful,
                "gen_chars": len(gen), "gen_head": gen[:160],
            }
            per.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        print(f"chunk {start}-{idxs[-1]} done, elapsed {time.time()-t0:.0f}s", flush=True)
    fh.close()

    # summary over all rows present in file
    allrec = [json.loads(l) for l in open(args.out, encoding="utf-8") if l.strip()]
    n = len(allrec)
    def mean(key):
        vals = [r[key] for r in allrec if r[key] is not None]
        return (sum(vals) / len(vals)) if vals else None
    summary = {
        "n_items": n,
        "D3_fabrication_rate_mean": mean("fabrication_rate_D3"),
        "D4_faithfulness_mean": mean("faithfulness_D4"),
        "ref_num_recall_proxy_mean": mean("ref_num_recall_proxy"),
        "ref_faithfulness_baseline_mean": mean("ref_faithfulness_baseline"),
        "D1_conclusion_macro_F1": "NOT_IMPLEMENTED (test 集无买/卖/持标签体系)",
        "D2_factor_weighted_recall": "NOT_IMPLEMENTED (无因子权重表)",
        "D5_llm_as_judge": "NOT_IMPLEMENTED (未做人工双标校准子集)",
        "max_tokens": args.max_tokens, "adapter": args.adapter, "data": args.data, "limit": args.limit,
    }
    json.dump(summary, open(args.summary, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
