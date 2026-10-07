#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""chunjiang1.0 技术报告配图生成脚本（全程脚本生成，数字只从数据源取）。"""
import os
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)  # 项目根（脚本所在 release/ 的上级）
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# ---- 中文字体：优先 STHeiti，回退 Hiragino Sans GB ----
available = {f.name for f in fm.fontManager.ttflist}
for cand in ["STHeiti", "Heiti SC", "Hiragino Sans GB", "PingFang SC"]:
    if cand in available:
        plt.rcParams["font.family"] = cand
        print("FONT_USED:", cand)
        break
plt.rcParams["axes.unicode_minus"] = False

DPI = 200

# ================= 解析训练日志 =================
TRAIN_RE = re.compile(r"^Iter (\d+): Train loss ([\d.]+),")
VAL_RE = re.compile(r"^Iter (\d+): Val loss ([\d.]+),")


def parse(path):
    tr, va = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = TRAIN_RE.match(line)
            if m:
                tr.append((int(m.group(1)), float(m.group(2))))
                continue
            m = VAL_RE.match(line)
            if m:
                va.append((int(m.group(1)), float(m.group(2))))
    return tr, va


b2_tr, b2_va = parse(os.path.join(BASE, "train_batch2.log"))
b3_tr, b3_va = parse(os.path.join(BASE, "train_batch3.log"))

print("PARSE batch2 Train:", len(b2_tr), "Val:", len(b2_va))
print("PARSE batch3 Train:", len(b3_tr), "Val:", len(b3_va))
print("b2 train x-range:", b2_tr[0][0], "-", b2_tr[-1][0])
print("b3 train x-range:", b3_tr[0][0], "-", b3_tr[-1][0])
print("b2 last val:", b2_va[-1])
print("b3 last val:", b3_va[-1])

# ================= fig1 训练损失曲线（双子图，独立坐标轴） =================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), dpi=DPI)

# 左：batch2
x2 = [p[0] for p in b2_tr]
y2 = [p[1] for p in b2_tr]
axL.plot(x2, y2, color="#1f77b4", lw=1.6, marker="o", ms=2.5, label="Train loss")
axL.scatter([p[0] for p in b2_va], [p[1] for p in b2_va],
            color="#d62728", s=26, zorder=5, label="Val loss")
axL.set_title("batch2 首训 · Train / Val loss (seq_len=3072)", fontsize=12)
axL.set_xlabel("Iter")
axL.set_ylabel("Loss")
axL.set_xlim(0, 720)
axL.grid(alpha=0.3, ls="--")
axL.legend(loc="upper right", fontsize=9)

# 右：batch3
x3 = [p[0] for p in b3_tr]
y3 = [p[1] for p in b3_tr]
axR.plot(x3, y3, color="#2ca02c", lw=1.6, marker="o", ms=3.5, label="Train loss")
axR.scatter([p[0] for p in b3_va], [p[1] for p in b3_va],
            color="#d62728", s=26, zorder=5, label="Val loss")
axR.set_title("batch3 续训 · Train / Val loss (seq_len=4096)", fontsize=12)
axR.set_xlabel("Iter")
axR.set_ylabel("Loss")
axR.set_xlim(0, 152)
axR.grid(alpha=0.3, ls="--")
axR.legend(loc="upper right", fontsize=9)

fig.suptitle("fig1 训练损失曲线（两组 val 集口径不同，不可直接横比）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
p1 = os.path.join(OUT, "fig1_train_loss.png")
fig.savefig(p1, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("WROTE", p1, os.path.getsize(p1))

# ================= fig2 test loss / ppl 三模型对比（双 Y 轴） =================
models = ["基座\nQwen3-8B", "batch2", "batch3"]
loss = [2.653, 2.118, 1.899]
ppl = [14.196, 8.312, 6.681]

fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=DPI)
xpos = range(len(models))
w = 0.38
b1 = ax1.bar([x - w / 2 for x in xpos], loss, w, color="#4c72b0", label="test loss（左轴）")
ax2 = ax1.twinx()
b2 = ax2.bar([x + w / 2 for x in xpos], ppl, w, color="#dd8452", label="test ppl（右轴）")

ax1.set_ylabel("test loss")
ax2.set_ylabel("test ppl")
ax1.set_xticks(list(xpos))
ax1.set_xticklabels(models)
ax1.set_title("fig2 独立 test 集（166 条）三模型对比：test loss / test ppl", fontsize=12)
ax1.set_ylim(0, max(loss) * 1.25)
ax2.set_ylim(0, max(ppl) * 1.25)

for r, v in zip(b1, loss):
    ax1.text(r.get_x() + r.get_width() / 2, v + 0.03, f"{v:.3f}",
             ha="center", va="bottom", fontsize=9)
for r, v in zip(b2, ppl):
    ax2.text(r.get_x() + r.get_width() / 2, v + 0.25, f"{v:.3f}",
             ha="center", va="bottom", fontsize=9)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
fig.tight_layout()
p2 = os.path.join(OUT, "fig2_test_loss.png")
fig.savefig(p2, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("WROTE", p2, os.path.getsize(p2))

# ================= fig3 语料构成（黄金根 5000 两块） =================
labels3 = ["模型生成语料\n4710 条 (94.2%)", "合成数据补题\n290 条 (5.8%)"]
sizes3 = [4710, 290]
colors3 = ["#4c72b0", "#dd8452"]
fig, ax = plt.subplots(figsize=(7.5, 6), dpi=DPI)
wedges, texts = ax.pie(sizes3, labels=labels3, colors=colors3,
                       startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.45, edgecolor="white"))
ax.set_title("fig3 黄金根 5000 条构成", fontsize=13)
ax.text(0, -1.35, "另 batch3 续训使用 1103 条本地语料（底座 Apache 2.0）",
        ha="center", va="center", fontsize=9, color="#555555")
ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.5, 1.25)
fig.tight_layout()
p3 = os.path.join(OUT, "fig3_corpus.png")
fig.savefig(p3, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("WROTE", p3, os.path.getsize(p3))

# ================= fig4 数据切分比例 =================
labels4 = ["train 3733\n(75%)", "valid 508\n(10%)", "test 746\n(15%)"]
sizes4 = [3733, 508, 746]
colors4 = ["#4c72b0", "#dd8452", "#55a868"]
fig, ax = plt.subplots(figsize=(7.5, 6), dpi=DPI)
ax.pie(sizes4, labels=labels4, colors=colors4,
       startangle=90, counterclock=False, autopct=None,
       wedgeprops=dict(width=0.45, edgecolor="white"))
ax.set_title("fig4 黄金根 5000 条数据切分 75/10/15", fontsize=13)
fig.tight_layout()
p4 = os.path.join(OUT, "fig4_split.png")
fig.savefig(p4, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("WROTE", p4, os.path.getsize(p4))

print("ALL_DONE")
