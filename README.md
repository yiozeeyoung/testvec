# testvec （kag 项目）

一个将分子 SMILES 的子结构分解结果，映射/合并成 ID 序列并构建指示矩阵（dense / indicator / 稀疏归一化）并可视化的实用仓库。数据示例来自 QM9。

主要能力：
- 将 JSONL 分解结果抽取为 CSV（smiles, smi1..smi6）并筛选（仅保留 smi5、smi6 为空的样本）。
- 将子结构 SMILES 用词表映射到离散 ID（vocab: id ↔ smi）。
- 合并 ID 列为单列序列（newid），并进行数据清洗与扩展（占位掩码）。
- 依据扩展结果构建 0/1 指示矩阵与稀疏 Lp 归一化矩阵，并输出多种制品（CSV/NPZ/PT）。
- 生成行计数分布、稀疏模式、相似度热力图等可视化图。

## 目录结构（节选）
- `clean_data/` 各阶段中间件与样例数据（不建议推送到远程）。
- `dataset/qm9/` 通过 PyTorch Geometric 加载 QM9 并抽取 SMILES 的脚本。
- `src/data/dataload.py` 数据抽取与转换（JSONL→CSV、筛选、映射、合并）。
- `src/data/matrix.py` 基于合并结果的过滤、扩展、矩阵构建（dense/indicator/sparse）。
- `src/figure/data.py` 稀疏矩阵制品可视化（直方图、Top-K 柱状、稀疏图、相似度）。

## 环境依赖
- Python 3.10+
- 必需：numpy, matplotlib
- 可选：
  - torch（保存稀疏张量 .pt）
  - torch_geometric（下载/读取 QM9）
  - rdkit（从 SDF 兜底解析 SMILES）
  - seaborn、pandas（聚类热图）

可选安装示例（按需）：
```bash
pip install numpy matplotlib
pip install torch
pip install torch_geometric
pip install rdkit-pypi
pip install seaborn pandas
```

## 数据准备：获取 QM9 SMILES（可选）
使用脚本 `dataset/qm9/load.py`（首次运行会下载 QM9，耗时较长）：
```bash
python -m dataset.qm9.load
```
输出：`dataset/qm9_smi.csv`（单列 header: smiles）。

## 常用数据流程
假设已有 JSONL 分解结果（每行包含 `smiles` 与若干 `smi`），以及词表 CSV（两列：`id,smi`）。

1) JSONL → CSV（smi1..smi6）
```bash
python - <<'PY'
from src.data.dataload import jsonl_to_smi_csv
out, n = jsonl_to_smi_csv(
    jsonl_path='clean_data/qm9/decomp.jsonl',
    out_csv_path='clean_data/qm9/decomp.smi6.csv'
)
print(out, n)
PY
```
仅保留 `smi` 出现次数 ≤ 4（smi5、smi6 为 '0'）：
```bash
python - <<'PY'
from src.data.dataload import jsonl_to_smi_csv_only_smi56_zero
out, n = jsonl_to_smi_csv_only_smi56_zero(
    jsonl_path='clean_data/qm9/decomp.jsonl',
    out_csv_path='clean_data/qm9/decomp.smi6.smi56eq0.csv'
)
print(out, n)
PY
```
或先通用转换再过滤：
```bash
python - <<'PY'
from src.data.dataload import filter_smi56_zero
out, n = filter_smi56_zero(
    csv_path='clean_data/qm9/decomp.smi6.csv'
)
print(out, n)
PY
```

2) 将 smi1..smi6 映射到 ID（词表 `id,smi`）
```bash
python - <<'PY'
from src.data.dataload import map_smi_columns_to_id
out, n = map_smi_columns_to_id(
    vocab_csv_path='clean_data/qm9_ke/decomp.vocab100.kek.csv',
    in_csv_path='clean_data/qm9/decomp.smi6.smi56eq0.csv',
    out_csv_path='clean_data/qm9/decomp.smi6.smi56eq0.mapped.csv'
)
print(out, n)
PY
```

3) 合并为单列 `newid`
```bash
python - <<'PY'
from src.data.dataload import merge_smi_ids_to_newid
out, n = merge_smi_ids_to_newid(
    in_csv_path='clean_data/qm9/decomp.smi6.smi56eq0.mapped.csv',
    out_csv_path='clean_data/matrix/base.csv'  # 两列：smiles,newid
)
print(out, n)
PY
```

4) 过滤、扩展与矩阵构建
- 过滤掉 `newid` 不含分号的样本：
```bash
python -m src.data.matrix \
  --input clean_data/matrix/base.csv \
  --output clean_data/matrix/base.filtered.csv
```
- 扩展 `newid`（占位为全角“（）”），产生 `newid_masked,od`：
```bash
python -m src.data.matrix \
  --input clean_data/matrix/base.filtered.csv \
  --expand \
  --output clean_data/matrix/base.filtered.expanded.csv
```
- 构建 0/1 指示矩阵：
```bash
python - <<'PY'
from src.data.dataload import add_id_column_to_csv
path = add_id_column_to_csv('clean_data/qm9/decomp.smi6.smi56eq0.mapped.csv', out_csv_path='clean_data/matrix/ids.csv')
print(path)
PY
```
```bash
python -m src.data.matrix \
  --make-indicator \
  --ids-csv clean_data/matrix/ids.csv \
  --expanded-csv clean_data/matrix/base.filtered.expanded.csv \
  --output clean_data/matrix/qm9_matrix_indicator.csv
```
- 生成稀疏 Lp 归一化指示矩阵（写出 `.coo.npz`/`.row_counts.csv`，可选 `.columns.txt`/`.pt`）：
```bash
python -m src.data.matrix \
  --make-sparse \
  --ids-csv clean_data/matrix/ids.csv \
  --expanded-csv clean_data/matrix/base.filtered.expanded.csv \
  --out-prefix clean_data/matrix/qm9_matrix_indicator_norm_p2 \
  --p 2 \
  --save-columns
```

## 可视化
将稀疏指标矩阵制品可视化到 `figure/`：
```bash
python -m src.figure.data \
  --prefix clean_data/matrix/qm9_matrix_indicator_norm_p2 \
  --out-dir figure \
  --similarity
```
输出：
- `row_counts_topk.png`、`row_counts_hist.png`
- `row_values_samples.png`、`sparse_pattern_sample.png`
- `sim_heatmap.png`（和可选 `sim_clustermap.png`）

## 注意
- 大体量数据与图片请不要提交到 Git（本仓库提供 `.gitignore` 示例）。
- 无需 GPU 也可跑大部分数据管线；`torch_geometric`/`rdkit` 仅在特定步骤需要。

## 许可与致谢
- 许可：MIT（如需更改，请在根目录替换 License）
- 数据与工具：QM9、PyTorch Geometric、RDKit、NumPy、Matplotlib、Seaborn、Pandas 等
