#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 PyTorch Geometric 加载 QM9 数据集，提取 SMILES 并保存到 kag/dataset/qm9_smi.csv。

优先顺序：
1) data.smiles（逐条遍历）
2) dataset.smiles（部分版本提供）
3) 读取原始 raw 路径中的 gdb9.sdf.csv（列名 smiles/SMILES 均支持）

输出 CSV 仅包含一列 header: smiles。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import List, Optional


def _info(msg: str) -> None:
	print(f"[QM9] {msg}")


def load_qm9_smiles(qm9_root: Path) -> List[str]:
	try:
		from torch_geometric.datasets import QM9
	except Exception as e:
		raise RuntimeError(
			"导入 torch_geometric 失败，请先安装：pip install torch_geometric"
		) from e

	_info(f"准备在 {qm9_root} 下载/加载 QM9 数据集（首次运行将自动下载，可能较慢）")
	ds = QM9(root=str(qm9_root))
	n = len(ds)
	_info(f"QM9 加载完成，共 {n} 条分子")

	smiles: List[str] = []

	# 1) 尝试逐条读取 data.smiles（部分版本支持）
	try:
		cnt = 0
		for data in ds:  # type: ignore[assignment]
			s = None
			# torch_geometric.data.Data 支持 dict 风格访问
			if hasattr(data, "smiles"):
				s = getattr(data, "smiles")
			else:
				try:
					s = data.get("smiles")  # type: ignore[attr-defined]
				except Exception:
					s = None
			if s is not None:
				smiles.append(str(s))
			cnt += 1
		if len(smiles) == n and n > 0:
			_info("使用 data.smiles 成功提取 SMILES")
			return smiles
		else:
			smiles.clear()
	except Exception:
		smiles.clear()

	# 2) 尝试 dataset.smiles（列表）
	try:
		if hasattr(ds, "smiles"):
			s_list = getattr(ds, "smiles")
			if isinstance(s_list, (list, tuple)) and len(s_list) > 0:
				smiles = [str(s) for s in s_list]
				_info("使用 dataset.smiles 成功提取 SMILES")
				return smiles
	except Exception:
		pass

	# 3) 回退到 raw 的 CSV 文件（通常为 gdb9.sdf.csv）
	try:
		raw_dir = Path(ds.raw_dir)
		csv_candidates = list(raw_dir.glob("*.csv"))
		if not csv_candidates:
			raise FileNotFoundError(f"未在 {raw_dir} 找到 *.csv")
		# 选第一个（一般是 gdb9.sdf.csv）
		csv_path = csv_candidates[0]
		_info(f"从原始文件读取：{csv_path}")
		with csv_path.open("r", newline="") as f:
			reader = csv.DictReader(f)
			key = None
			# 兼容 'smiles' 或 'SMILES'
			if reader.fieldnames:
				for k in reader.fieldnames:
					if k.lower() == "smiles":
						key = k
						break
			if key is None:
				raise KeyError("原始 CSV 未包含 smiles 列")
			for row in reader:
				s = row.get(key)
				if s is not None:
					smiles.append(s)
		if smiles:
			_info("已从原始 CSV 提取 SMILES")
			return smiles
	except Exception:
		pass

	# 4) 最后回退：使用 RDKit 直接从 SDF 生成 SMILES（无需 CSV 中存在 smiles 列）
	try:
		from rdkit import Chem  # type: ignore
		raw_dir = Path(ds.raw_dir)
		sdf_candidates = list(raw_dir.glob("*.sdf"))
		if not sdf_candidates:
			raise FileNotFoundError(f"未在 {raw_dir} 找到 *.sdf")
		sdf_path = sdf_candidates[0]
		_info(f"使用 RDKit 从 SDF 生成 SMILES：{sdf_path}")
		suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
		for mol in suppl:
			if mol is None:
				continue
			try:
				s = Chem.MolToSmiles(mol)
				smiles.append(s)
			except Exception:
				# 个别分子失败则跳过
				continue
		if smiles:
			_info("已从 SDF 直接生成 SMILES")
			return smiles
	except Exception as e:
		raise RuntimeError("无法从 PyG QM9 提取 SMILES，请检查环境或 PyG 版本/依赖 (RDKit)") from e

	raise RuntimeError("未能提取到任何 SMILES")


def save_smiles_csv(smiles: List[str], out_csv: Path) -> None:
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	with out_csv.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["smiles"])  # header
		for s in smiles:
			writer.writerow([s])
	_info(f"已写入 {len(smiles)} 条 SMILES 至 {out_csv}")


def main(argv: List[str] | None = None) -> int:
	# 将数据根目录放在当前文件夹下的 'pyg'，输出到项目根 dataset/qm9_smi.csv
	this_dir = Path(__file__).resolve().parent
	kag_root = this_dir.parent  # kag/
	qm9_root = this_dir / "pyg"  # kag/qm9/pyg
	out_csv = kag_root / "dataset" / "qm9_smi.csv"

	_info(f"输出目标：{out_csv}")

	smiles = load_qm9_smiles(qm9_root)
	save_smiles_csv(smiles, out_csv)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())


# ----------------------------
# 附加功能：为 CSV 添加自定义 id 列
# 规则：
#   - 个位数（0-9） -> a-j
#   - 十位数（0-9） -> A-J
#   - 百位数及以上 -> 用 k 的重复次数表示百位（例如 0-99 无 k，100-199 前缀 'k'，200-299 前缀 'kk'，以此类推），
#     然后拼接十位和个位的映射字符。
#   - 例如：
#       0 -> aa
#       9 -> aj
#       10 -> Ba
#       19 -> Bj
#       99 -> Jj
#       100 -> kaa
#       101 -> kab
#       110 -> kBa
#       999 -> kkkkkkkkJj （9 个 k + J + j）

_ONES = "abcdefghij"  # 0..9
_TENS = "ABCDEFGHIJ"  # 0..9


def _index_to_id(idx: int) -> str:
	if idx < 0:
		raise ValueError("index must be non-negative")
	hundreds = idx // 100
	tens = (idx // 10) % 10
	ones = idx % 10
	prefix = "k" * hundreds if hundreds > 0 else ""
	return f"{prefix}{_TENS[tens]}{_ONES[ones]}"


def add_id_column_to_csv(
	csv_path: str | Path,
	out_csv_path: Optional[str | Path] = None,
	id_col_name: str = "id",
) -> Path:
	"""
	为给定 CSV 添加一列名为 `id`（可自定义）的列。id 生成规则：
	- 十位数用 A-J 表示（0->A, 9->J）
	- 个位数用 a-j 表示（0->a, 9->j）
	- 若有百位数（及以上），用 'k' 重复次数表示百位数，然后接上十位、个位映射字符

	如果 out_csv_path 未提供，则在原文件同目录生成一个带后缀的文件名（例如 foo.id.csv）。
	返回实际写入的 CSV 路径。
	"""
	in_path = Path(csv_path)
	if not in_path.exists():
		raise FileNotFoundError(f"CSV not found: {in_path}")

	if out_csv_path is None:
		out_path = in_path.with_suffix("")
		out_path = out_path.with_name(out_path.name + ".id").with_suffix(".csv")
	else:
		out_path = Path(out_csv_path)

	# 读取全部行并检测 header
	rows: List[List[str]] = []
	with in_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.reader(f)
		try:
			header = next(reader)
		except StopIteration:
			raise ValueError("输入 CSV 为空")
		rows = [r for r in reader]

	# 写出：将 id 列加在最前或最后，这里加在最前
	out_path.parent.mkdir(parents=True, exist_ok=True)
	with out_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow([id_col_name] + header)
		for i, r in enumerate(rows):
			idv = _index_to_id(i)
			writer.writerow([idv] + r)

	_info(f"已写入带 id 的 CSV：{out_path}（共 {len(rows)} 行）")
	return out_path

