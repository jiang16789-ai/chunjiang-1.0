# chunjiang1.0

面向 A 股量化复盘的「投资决策审查与纠错」LoRA 微调模型：不预测行情、不给投资观点，只对给定的投资方案与复盘材料做四类裁决——挑硬伤、核数字、判是否违反账户硬约束、输出唯一决策。

## 开源边界（2026-10-07 定版）

- **开源**：训练 / 评估脚本、方法论（技术报告）、模型卡。许可证 Apache-2.0。
- **暂不公开**：训练语料（含原始根与全部派生产物）、评测集（含 train / valid / test 切分）、模型权重（LoRA adapter）。
- **原因**：训练语料主体为模型生成语料，来源与授权信息有限；另有少量合成数据。基于当前许可状态，语料与权重均不对外发布，仅公开可复现的方法论与脚本。

> 因此本仓库**不含** `data/`、`adapters/` 及任何语料/权重文件。按本仓库脚本可复现**流程**，但因语料与权重不公开，无法复现论文中的损失数值。

## 仓库内容

| 路径 | 说明 |
| --- | --- |
| `paper/paper-chunjiang1.0.md` | 技术报告（方法论、语料构成、评测体系、结果、开源边界） |
| `MODEL_CARD.md` | 模型卡（模型概述、数据来源与许可、开发者署名、许可） |
| `prepare_train.py` | 数据准备：合并压缩语料 → 转 mlx-lm completions 格式 → 按维度分层切分 75/10/15 |
| `split_local_batch3.py` | 第三轮续训数据切分 + manifest（id/dim/source + prompt/completion sha256） |
| `eval_five_dim.py` | 五维评测中可实测部分（D3 无依据因子率 / D4 证据忠实度代理口径） |
| `train_config.yaml` | 首训配置（8B · MLX LoRA · 4bit 基座） |
| `train_config_batch2.yaml` | 第二批微调配置 |
| `train_config_batch3.yaml` | 第三轮续训配置（batch2 adapter 续训） |
| `train_config_batch3_smoke.yaml` | 冒烟测试配置（验证 resume 加载 + 序列长度内存） |

## 复现方法

**环境**：Apple Silicon（MLX）+ `mlx-lm`；基座 `mlx-community/Qwen3-8B-4bit`（Apache-2.0，4-bit 量化）。

1. **准备数据**：把语料整理成 `data/corpus_root_5000.jsonl`（每行 `{id, dim, prompt, content}`）与压缩产物，然后
   `python3 prepare_train.py 42` → 生成 `data/train/{train,valid,test}.jsonl`。
2. **切分（续训语料）**：`python3 split_local_batch3.py` → 生成 `data/train_local_batch3/` 与 manifest。
3. **训练**：`python3 -m mlx_lm lora --config train_config_batch2.yaml`（首训），再 `--config train_config_batch3.yaml`（续训）。
4. **评估**：`python3 -m mlx_lm ... test` 或 `python3 eval_five_dim.py`；五维评测的 D1/D2/D5 为设计态、尚未实现（见技术报告 §4）。

**注意**：脚本与配置中的 `/path/to/chunjiang1.0` 与 `~/.cache/huggingface/...` 为**占位路径**，需按本地实际情况替换；语料与权重不公开，训练数值不可复现。

## 许可

开源部分（脚本 / 方法论 / 模型卡）采用 **Apache License 2.0**（见 `LICENSE`）。基座 Qwen3-8B 同为 Apache-2.0。训练语料、评测集与模型权重不对外发布，无对外许可。

## 免责声明

本项目与其输出均为**投资决策审查工具**的研究产物，**不构成任何投资建议**；不提供行情预测，也不对任何交易的盈亏负责。使用者须自行承担全部风险。
