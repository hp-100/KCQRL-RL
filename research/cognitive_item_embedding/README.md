# Cognitive Item Embedding: Conic10K Data Audit

研究范围是高中数学圆锥曲线题目。当前阶段只做 Conic10K 数据下载、数据体检和固定随机种子的 100 题人工审核样本；明确不做模型训练、LLM 自动标注、Q 矩阵、NCDM 或 CAT。

## 安装

```bash
python -m pip install -r research/cognitive_item_embedding/requirements.txt
```

`datasets` 仅用于优先从 Hugging Face `WenyangHui/Conic10K` 加载；脚本保留官方 GitHub 仓库 fallback。

## 命令

```bash
python research/cognitive_item_embedding/scripts/download_conic10k.py
python research/cognitive_item_embedding/scripts/inspect_dataset.py
python research/cognitive_item_embedding/scripts/sample_items.py --seed 20260623 --n 100
pytest research/cognitive_item_embedding/tests
```

## 输出

- `data/raw/conic10k/`: 本地原始 JSONL 与 metadata，已被 `.gitignore` 排除。
- `data/interim/conic10k_audit.{json,md}` 与 `conic10k_missing.csv`: 小型统计结果。
- `docs/DATA_AUDIT_REPORT.md`: 自动更新的数据体检报告。
- `data/samples/conic10k_sample_100.{csv,jsonl,html}`: 100 题人工审核样本，审核列留空。

## 数据许可与引用

优先数据源：Hugging Face `WenyangHui/Conic10K`。官方仓库 `whyNLP/Conic10K` 标注 MIT License。若本地下载路径无法确认许可或版本，应只提交脚本和统计说明，不提交完整原始数据。
