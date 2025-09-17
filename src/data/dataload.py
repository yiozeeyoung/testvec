from __future__ import annotations

from pathlib import Path
import csv
from typing import Optional, Tuple
import json
from typing import Any, Iterable, List


def save_first_column_as_csv(
	txt_path: str | Path,
	out_csv_path: Optional[str | Path] = None,
	column_name: str = "smi",
) -> Tuple[Path, int]:
	"""
	读取一个以空白（制表符或空格）分隔的文本文件，只提取第一列，并保存为单列 CSV。

	参数:
		txt_path: 输入 txt 文件路径。
		out_csv_path: 输出 CSV 路径；默认与输入同目录、同名但扩展名为 .csv。
		column_name: CSV 的列名（默认 "smi"）。

	返回:
		(输出 CSV 路径, 写入的行数)

	说明:
		- 行内以任意空白字符分隔（包含制表符和多个空格）。
		- 会跳过空行；若某行只有一列之外的内容将被忽略（仅取第一列）。
	"""
	in_path = Path(txt_path)
	if not in_path.exists():
		raise FileNotFoundError(f"Input file not found: {in_path}")

	if out_csv_path is None:
		out_path = in_path.with_suffix(".csv")
	else:
		out_path = Path(out_csv_path)

	count = 0
	out_path.parent.mkdir(parents=True, exist_ok=True)

	with in_path.open("r", encoding="utf-8") as fin, out_path.open(
		"w", encoding="utf-8", newline=""
	) as fout:
		writer = csv.writer(fout)
		writer.writerow([column_name])

		for line in fin:
			# 去除两端空白并跳过空行
			s = line.strip()
			if not s:
				continue
			# 以任意空白分隔，取第一列
			first = s.split()[0]
			writer.writerow([first])
			count += 1

	return out_path, count


if __name__ == "__main__":
	# 便捷用法：直接运行本文件会将指定路径的第一列导出为 CSV。
	default_in = "/Users/ayang/code/kag/clean_data/qm9_ke/qm9_ps_vocab_100_kek.txt"
	out_csv, n = save_first_column_as_csv(default_in, column_name="smi")
	print(f"Saved {n} rows to {out_csv}")


def _collect_smi_values(obj: Any, max_items: int = 4) -> List[str]:
	"""深度优先遍历字典/列表，按遇到顺序收集键名为 'smi' 的值。

	仅收集至多 max_items 个条目（可配置，典型为 6 个，对应 smi1..smi6）。
	"""
	found: List[str] = []

	def dfs(x: Any) -> None:
		if len(found) >= max_items:
			return
		if isinstance(x, dict):
			# 先收集当前层级的 'smi'
			if "smi" in x:
				val = x["smi"]
				found.append(str(val))
				if len(found) >= max_items:
					return
			# 再递归其余键值
			for k, v in x.items():
				if k == "smi":
					continue
				dfs(v)
		elif isinstance(x, list):
			for it in x:
				if len(found) >= max_items:
					break
				dfs(it)
		# 其他类型忽略

	dfs(obj)
	return found


def jsonl_to_smi_csv(
	jsonl_path: str | Path,
	out_csv_path: Optional[str | Path] = None,
	limit_lines: Optional[int] = None,
) -> Tuple[Path, int]:
	"""
	将 JSONL 转换为 CSV，列为：smiles, smi1, smi2, smi3, smi4, smi5, smi6。
	- smiles: 每行 JSON 中键 'smiles' 的值（若缺失则为 '0'）
	- smi1..smi6: 按遇到顺序收集该行中键名为 'smi' 的值（最多 6 个）；不足则以 '0' 补齐。

	参数:
		jsonl_path: 输入 JSONL 路径
		out_csv_path: 输出 CSV 路径；默认与输入同名但扩展名为 .csv
		limit_lines: 若提供，仅处理前 N 行（用于快速测试）

	返回: (输出路径, 写入行数)
	"""
	in_path = Path(jsonl_path)
	if not in_path.exists():
		raise FileNotFoundError(f"Input JSONL not found: {in_path}")

	if out_csv_path is None:
		out_path = in_path.with_suffix(".csv")
	else:
		out_path = Path(out_csv_path)

	out_path.parent.mkdir(parents=True, exist_ok=True)

	written = 0
	with in_path.open("r", encoding="utf-8") as fin, out_path.open(
		"w", encoding="utf-8", newline=""
	) as fout:
		writer = csv.writer(fout)
		writer.writerow(["smiles", "smi1", "smi2", "smi3", "smi4", "smi5", "smi6"])

		for i, line in enumerate(fin):
			if limit_lines is not None and i >= limit_lines:
				break
			s = line.strip()
			if not s:
				continue
			try:
				obj = json.loads(s)
			except Exception:
				# 跳过无法解析的行
				continue

			smiles = str(obj.get("smiles", "0"))
			smi_vals = _collect_smi_values(obj, max_items=6)
			# 填充到 6 个
			while len(smi_vals) < 6:
				smi_vals.append("0")

			writer.writerow([smiles] + smi_vals[:6])
			written += 1

	return out_path, written


def filter_smi56_zero(
	csv_path: str | Path,
	out_csv_path: Optional[str | Path] = None,
) -> Tuple[Path, int]:
	"""
	过滤 CSV：仅保留 smi5 和 smi6 均为字符串 '0' 的行。

	参数:
		csv_path: 输入 CSV 路径（需包含列：smiles, smi1..smi6）
		out_csv_path: 输出路径；默认在原文件名后添加后缀 .smi56eq0.csv

	返回: (输出路径, 保留的行数，不含表头)
	"""
	in_path = Path(csv_path)
	if not in_path.exists():
		raise FileNotFoundError(f"CSV not found: {in_path}")

	if out_csv_path is None:
		out_path = in_path.with_suffix("")
		out_path = out_path.with_name(out_path.name + ".smi56eq0").with_suffix(".csv")
	else:
		out_path = Path(out_csv_path)

	kept = 0
	with in_path.open("r", encoding="utf-8", newline="") as fin, out_path.open(
		"w", encoding="utf-8", newline=""
	) as fout:
		reader = csv.reader(fin)
		writer = csv.writer(fout)
		try:
			header = next(reader)
		except StopIteration:
			raise ValueError("输入 CSV 为空")
		writer.writerow(header)

		# 获取列索引
		try:
			idx_smi5 = header.index("smi5")
			idx_smi6 = header.index("smi6")
		except ValueError as e:
			raise ValueError("输入 CSV 缺少列 'smi5' 或 'smi6'") from e

		for row in reader:
			if not row:
				continue
			v5 = row[idx_smi5].strip() if idx_smi5 < len(row) else ""
			v6 = row[idx_smi6].strip() if idx_smi6 < len(row) else ""
			if v5 == "0" and v6 == "0":
				writer.writerow(row)
				kept += 1

	return out_path, kept


def jsonl_to_smi_csv_only_smi56_zero(
	jsonl_path: str | Path,
	out_csv_path: Optional[str | Path] = None,
	limit_lines: Optional[int] = None,
) -> Tuple[Path, int]:
	"""
	从 JSONL 直接生成 CSV（smiles + smi1..smi6），并且仅保留那些收集到的 'smi' 个数 <= 4 的行
	（即 smi5 与 smi6 必为 '0'）。可用于绕过读取既有 CSV 时的异常。
	"""
	in_path = Path(jsonl_path)
	if not in_path.exists():
		raise FileNotFoundError(f"Input JSONL not found: {in_path}")

	if out_csv_path is None:
		out_path = in_path.with_suffix("")
		out_path = out_path.with_name(out_path.name + ".smi6.smi56eq0").with_suffix(".csv")
	else:
		out_path = Path(out_csv_path)

	out_path.parent.mkdir(parents=True, exist_ok=True)

	written = 0
	with in_path.open("r", encoding="utf-8") as fin, out_path.open(
		"w", encoding="utf-8", newline=""
	) as fout:
		writer = csv.writer(fout)
		writer.writerow(["smiles", "smi1", "smi2", "smi3", "smi4", "smi5", "smi6"])

		for i, line in enumerate(fin):
			if limit_lines is not None and i >= limit_lines:
				break
			s = line.strip()
			if not s:
				continue
			try:
				obj = json.loads(s)
			except Exception:
				continue

			smiles = str(obj.get("smiles", "0"))
			smi_vals = _collect_smi_values(obj, max_items=6)
			# 仅保留 smi 出现次数 <= 4 的样本
			if len(smi_vals) > 4:
				continue
			while len(smi_vals) < 6:
				smi_vals.append("0")
			writer.writerow([smiles] + smi_vals[:6])
			written += 1

	return out_path, written


def map_smi_columns_to_id(
	vocab_csv_path: str | Path,
	in_csv_path: str | Path,
	out_csv_path: Optional[str | Path] = None,
	smi_cols: Optional[List[str]] = None,
	id_col_name: str = "id",
) -> Tuple[Path, int]:
	"""
	使用 vocab CSV（列含 id 与 smi 对应关系）将 in_csv 中 smi 列（默认 smi1..smi6）逐项替换为 id。

	要求：
	  - vocab_csv_path: 形如由 add_id_column_to_csv 生成的 CSV，第一列为 id，第二列为 smi。
	  - in_csv_path: 包含 smi1..smi6 的 CSV（例如由 jsonl_to_smi_csv 生成）

	返回: (输出 CSV 路径, 写入行数，不含表头)
	"""
	vocab_path = Path(vocab_csv_path)
	in_path = Path(in_csv_path)
	if not vocab_path.exists():
		raise FileNotFoundError(f"vocab CSV not found: {vocab_path}")
	if not in_path.exists():
		raise FileNotFoundError(f"input CSV not found: {in_path}")

	if smi_cols is None:
		smi_cols = ["smi1", "smi2", "smi3", "smi4", "smi5", "smi6"]

	# 构建 smi->id 映射
	smi2id: dict[str, str] = {}
	with vocab_path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.reader(f)
		try:
			header = next(reader)
		except StopIteration:
			raise ValueError("vocab CSV 为空")
		# 允许两种布局：
		# 1) id,smi
		# 2) smi 单列（这时无法映射到 id；因此要求是包含 id 列）
		try:
			idx_id = header.index(id_col_name)
			idx_smi = header.index("smi")
		except ValueError:
			# 尝试处理没有 header 的情况（两列）
			idx_id, idx_smi = 0, 1 if len(header) > 1 else (None, None)  # type: ignore[assignment]
		if idx_id is None or idx_smi is None:
			raise ValueError("vocab CSV 必须包含 'id' 与 'smi' 两列")

		for row in reader:
			if not row:
				continue
			if idx_id >= len(row) or idx_smi >= len(row):
				continue
			sid = row[idx_id].strip()
			ssmi = row[idx_smi].strip()
			if ssmi:
				smi2id[ssmi] = sid

	# 读取 in_csv 并替换
	if out_csv_path is None:
		out_path = in_path.with_suffix("")
		out_path = out_path.with_name(out_path.name + ".mapped").with_suffix(".csv")
	else:
		out_path = Path(out_csv_path)

	written = 0
	with in_path.open("r", encoding="utf-8", newline="") as fin, out_path.open(
		"w", encoding="utf-8", newline=""
	) as fout:
		reader = csv.reader(fin)
		writer = csv.writer(fout)
		try:
			header = next(reader)
		except StopIteration:
			raise ValueError("输入 CSV 为空")
		writer.writerow(header)

		# 找到 smi 列索引
		col_idx = []
		for c in smi_cols:
			try:
				col_idx.append(header.index(c))
			except ValueError:
				col_idx.append(None)  # type: ignore[arg-type]

		for row in reader:
			if not row:
				continue
			new_row = row[:]
			for idx in col_idx:
				if idx is None or idx >= len(new_row):
					continue
				val = new_row[idx].strip()
				if val and val != "0":
					new_row[idx] = smi2id.get(val, val)  # 若找不到映射，保留原值
			writer.writerow(new_row)
			written += 1

	return out_path, written


def merge_smi_ids_to_newid(
	in_csv_path: str | Path,
	out_csv_path: Optional[str | Path] = None,
	smi_cols: Optional[List[str]] = None,
	newid_col: str = "newid",
	delimiter: str = ";",
) -> Tuple[Path, int]:
	"""
	将输入 CSV 中的 smi1..smi6 列（默认为这六列）取非 "0" 的值合并为单列 `newid`（用分号连接），
	并输出仅包含两列：smiles,newid。
	"""
	in_path = Path(in_csv_path)
	if not in_path.exists():
		raise FileNotFoundError(f"CSV not found: {in_path}")

	if smi_cols is None:
		smi_cols = ["smi1", "smi2", "smi3", "smi4", "smi5", "smi6"]

	if out_csv_path is None:
		out_path = in_path.with_suffix("")
		out_path = out_path.with_name(out_path.name + f".{newid_col}").with_suffix(".csv")
	else:
		out_path = Path(out_csv_path)

	written = 0
	with in_path.open("r", encoding="utf-8", newline="") as fin, out_path.open(
		"w", encoding="utf-8", newline=""
	) as fout:
		reader = csv.reader(fin)
		writer = csv.writer(fout)
		try:
			header = next(reader)
		except StopIteration:
			raise ValueError("输入 CSV 为空")

		# 获取列索引
		try:
			idx_smiles = header.index("smiles")
		except ValueError as e:
			raise ValueError("输入 CSV 缺少列 'smiles'") from e
		idx_cols = []
		for c in smi_cols:
			try:
				idx_cols.append(header.index(c))
			except ValueError:
				idx_cols.append(None)  # type: ignore[arg-type]

		# 写出头
		writer.writerow(["smiles", newid_col])

		for row in reader:
			if not row:
				continue
			smi_vals: List[str] = []
			for idx in idx_cols:
				if idx is None or idx >= len(row):
					continue
				v = row[idx].strip()
				if v and v != "0":
					smi_vals.append(v)
			newid = delimiter.join(smi_vals)
			writer.writerow([row[idx_smiles], newid])
			written += 1

	return out_path, written

