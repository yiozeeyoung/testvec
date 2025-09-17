"""
Tools for processing matrix CSV files.

Features:
- Filter out rows whose 'newid' column has no separator (default ';').
- Expand each 'newid' into multiple masked variants, replacing each token in turn
  with Chinese parentheses '（）' and output pairs of (masked_string, od).

Usage examples:
- Dry run (show counts only):
	python -m src.data.matrix --input /Users/ayang/code/kag/clean_data/matrix/base.csv --delimiter ';'

- Write filtered CSV to a new file:
	python -m src.data.matrix --input /Users/ayang/code/kag/clean_data/matrix/base.csv \
		--output /Users/ayang/code/kag/clean_data/matrix/base.filtered.csv

- In-place filtering (with backup):
	python -m src.data.matrix --input /Users/ayang/code/kag/clean_data/matrix/base.csv --in-place

- Expand 'newid' into masked variants (two columns: 'newid_masked', 'od'):
	python -m src.data.matrix --input /Users/ayang/code/kag/clean_data/matrix/base.filtered.csv \
		--expand --output /Users/ayang/code/kag/clean_data/matrix/base.filtered.expanded.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import itertools
import math
from typing import Iterable, Tuple, Iterator, Dict
try:
	import numpy as np  # type: ignore
except Exception:  # numpy is optional for non-sparse paths
	np = None  # type: ignore
try:
	import torch  # type: ignore
except Exception:
	torch = None  # type: ignore


def _iter_filtered_rows(
	rows: Iterable[dict], *, delimiter: str = ";", column: str = "newid"
) -> Tuple[Iterable[dict], int, int]:
	"""Filter rows keeping only those whose `column` contains `delimiter`.

	Returns a tuple of (filtered_rows_iterable, kept_count, removed_count).
	"""

	kept = 0
	removed = 0

	def generator():
		nonlocal kept, removed
		for row in rows:
			value = (row.get(column) or "").strip()
			if delimiter in value:
				kept += 1
				yield row
			else:
				removed += 1

	return generator(), kept, removed  # counts will be finalized after iteration


def filter_csv(
	input_csv: str,
	output_csv: str | None = None,
	*,
	delimiter: str = ";",
	column: str = "newid",
	encoding: str = "utf-8",
) -> Tuple[int, int, int]:
	"""Filter CSV rows by keeping only those with delimiter in the specified column.

	Args:
		input_csv: Path to the input CSV file.
		output_csv: Path to write filtered CSV. If None, no file is written.
		delimiter: The substring that must appear in the target column to keep the row.
		column: Column name to check, default 'newid'.
		encoding: File encoding to use.

	Returns:
		total_rows, kept_rows, removed_rows
	"""

	with open(input_csv, "r", encoding=encoding, newline="") as f:
		reader = csv.DictReader(f)
		fieldnames = reader.fieldnames or []
		if column not in fieldnames:
			raise ValueError(
				f"Column '{column}' not found in {input_csv}. Available: {fieldnames}"
			)

		filtered_iter, kept_ref, removed_ref = _iter_filtered_rows(reader, delimiter=delimiter, column=column)

		total = 0
		# If output path provided, write filtered rows
		if output_csv is not None:
			# Ensure directory exists
			os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
			with open(output_csv, "w", encoding=encoding, newline="") as out_f:
				writer = csv.DictWriter(out_f, fieldnames=fieldnames)
				writer.writeheader()
				for row in filtered_iter:
					writer.writerow(row)
					total += 1  # count kept rows as we stream
			kept = total
			# removed is unknown until generator is fully consumed; but since we only
			# iterated kept rows, we can compute removed by scanning input once more.
			# Instead, more efficient: compute total by re-reading quickly.
			# Simpler approach: re-open input to count total rows (excluding header).
			with open(input_csv, "r", encoding=encoding, newline="") as f2:
				total_rows = sum(1 for _ in f2) - 1
			removed = total_rows - kept
			return total_rows, kept, removed

		# No output: just count by iterating all rows
		kept = 0
		for _ in filtered_iter:
			kept += 1
		# To compute total, re-open file and count lines excluding header
		with open(input_csv, "r", encoding=encoding, newline="") as f2:
			total_rows = sum(1 for _ in f2) - 1
		removed = total_rows - kept
		return total_rows, kept, removed


def _detect_delimiter(value: str, default: str = ";") -> str:
	"""Detect whether a value uses ASCII ';' or full-width '；' as delimiter.

	If neither is present, returns `default`.
	"""
	if "；" in value:
		return "；"
	if ";" in value:
		return ";"
	return default


def iter_newid_mask_variants(
	rows: Iterable[dict], *, column: str = "newid", placeholder: str = "（）"
) -> Iterator[Dict[str, str]]:
	"""Yield masked variants for each row's `column` value.

	For input like "de；bc；ac" (or with ASCII ';'), yields per-token masked strings
	where the i-th token is replaced by `placeholder`, and also yields the replaced
	token as 'od'. The original delimiter style is preserved per row.

	Yields dicts with keys:
	- 'newid_masked'
	- 'od'
	"""
	for row in rows:
		value = (row.get(column) or "").strip()
		if not value:
			continue
		delim = _detect_delimiter(value)
		tokens = [t.strip() for t in value.split(delim)]
		if not tokens:
			continue
		for i, tok in enumerate(tokens):
			masked = list(tokens)
			masked[i] = placeholder
			yield {"newid_masked": delim.join(masked), "od": tok}


def expand_newid_to_csv(
	input_csv: str,
	output_csv: str,
	*,
	column: str = "newid",
	encoding: str = "utf-8",
	placeholder: str = "（）",
) -> Tuple[int, int]:
	"""Expand `newid` column per row into masked variants and write to CSV.

	Writes header ['newid_masked', 'od'] and appends one row per variant.

	Returns a tuple: (input_rows_count, output_rows_count)
	"""
	# Ensure directory exists
	os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
	with open(input_csv, "r", encoding=encoding, newline="") as f_in, open(
		output_csv, "w", encoding=encoding, newline=""
	) as f_out:
		reader = csv.DictReader(f_in)
		writer = csv.DictWriter(f_out, fieldnames=["newid_masked", "od"])
		writer.writeheader()
		in_rows = 0
		out_rows = 0
		for row in reader:
			in_rows += 1
			for item in iter_newid_mask_variants([row], column=column, placeholder=placeholder):
				writer.writerow(item)
				out_rows += 1
	return in_rows, out_rows


def _make_backup(path: str, suffix: str = ".bak") -> str:
	backup_path = path + suffix
	shutil.copy2(path, backup_path)
	return backup_path


def _read_column_values(csv_path: str, column: str, *, encoding: str = "utf-8") -> list:
	"""Read a single column from CSV into a list (skips header)."""
	values = []
	with open(csv_path, "r", encoding=encoding, newline="") as f:
		reader = csv.DictReader(f)
		if column not in (reader.fieldnames or []):
			raise ValueError(f"Column '{column}' not found in {csv_path}. Available: {reader.fieldnames}")
		for row in reader:
			values.append((row.get(column) or "").strip())
	return values


def build_large_matrix(
	*,
	ids_csv: str,
	expanded_csv: str,
	output_csv: str,
	id_column: str = "id",
	expanded_column: str = "newid_masked",
	fill_value: str = "0",
	encoding: str = "utf-8",
) -> Tuple[int, int]:
	"""Build a large matrix CSV.

	- First column header: 'id'; rows populated from ids_csv[id_column].
	- Remaining column headers: values from expanded_csv[expanded_column].
	- Cell values are filled with `fill_value` as strings.

	Returns:
		(n_rows, n_cols) where n_cols includes the 'id' column.
	"""
	ids = _read_column_values(ids_csv, id_column, encoding=encoding)
	col_names = _read_column_values(expanded_csv, expanded_column, encoding=encoding)

	# Prepare output dir
	os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

	header = ["id"] + col_names
	with open(output_csv, "w", encoding=encoding, newline="") as f_out:
		writer = csv.writer(f_out)
		writer.writerow(header)
		row_template = [fill_value] * len(col_names)
		for _id in ids:
			writer.writerow([_id] + row_template)

	return len(ids), len(header)


def build_indicator_matrix_from_expanded(
	*,
	ids_csv: str,
	expanded_csv: str,
	output_csv: str,
	id_column: str = "id",
	expanded_column: str = "newid_masked",
	od_column: str = "od",
	encoding: str = "utf-8",
	one_value: str = "1",
	zero_value: str = "0",
) -> Tuple[int, int]:
	"""Build a 0/1 indicator matrix using expanded CSV.

	Columns correspond to expanded[newid_masked] in order; rows correspond to ids[id].
	Cell(i, j) = 1 iff ids[i] == expanded[od][j], else 0.

	Returns (n_rows, n_cols including 'id').
	"""
	ids = _read_column_values(ids_csv, id_column, encoding=encoding)
	# Read expanded columns and od as two parallel lists preserving order
	col_names: list[str] = []
	od_list: list[str] = []
	with open(expanded_csv, "r", encoding=encoding, newline="") as f:
		reader = csv.DictReader(f)
		fns = reader.fieldnames or []
		if expanded_column not in fns or od_column not in fns:
			raise ValueError(
				f"Required columns '{expanded_column}' and '{od_column}' not found in {expanded_csv}. Available: {fns}"
			)
		for row in reader:
			col_names.append((row.get(expanded_column) or "").strip())
			od_list.append((row.get(od_column) or "").strip())

	os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
	header = ["id"] + col_names
	with open(output_csv, "w", encoding=encoding, newline="") as f_out:
		writer = csv.writer(f_out)
		writer.writerow(header)
		# For each id, write a row where positions matching od_list are one_value
		for _id in ids:
			row = [_id]
			# Compute per-column values on the fly to avoid large memory
			row.extend(one_value if od == _id else zero_value for od in od_list)
			writer.writerow(row)

	return len(ids), len(header)


def build_indicator_sparse_and_normalized(
	*,
	ids_csv: str,
	expanded_csv: str,
	out_prefix: str,
	id_column: str = "id",
	expanded_column: str = "newid_masked",
	od_column: str = "od",
	p: float = 2.0,
	save_torch: bool = False,
	save_columns: bool = False,
	encoding: str = "utf-8",
) -> Tuple[int, int]:
	"""Build a sparse COO normalized indicator matrix and save artifacts.

	Artifacts written:
	- <out_prefix>.coo.npz: dict with keys {rows, cols, data, shape}
	- <out_prefix>.row_counts.csv: two columns {id, count}
	- <out_prefix>.columns.txt: (optional) list of column names in order
	- <out_prefix>.pt: (optional) torch sparse COO tensor with normalized values
	"""
	if np is None:
		raise RuntimeError("NumPy required for sparse build")

	# Read ids and build index map
	ids = _read_column_values(ids_csv, id_column, encoding=encoding)
	row_index = {v: i for i, v in enumerate(ids)}

	rows: list[int] = []
	cols: list[int] = []
	col_names: list[str] = []
	# We'll first record raw ones (value=1), then normalize per row
	with open(expanded_csv, "r", encoding=encoding, newline="") as f:
		reader = csv.DictReader(f)
		fns = reader.fieldnames or []
		if expanded_column not in fns or od_column not in fns:
			raise ValueError(
				f"Required columns '{expanded_column}' and '{od_column}' not found in {expanded_csv}. Available: {fns}"
			)
		for j, row in enumerate(reader):
			col_names.append((row.get(expanded_column) or "").strip())
			od = (row.get(od_column) or "").strip()
			i = row_index.get(od)
			if i is not None:
				rows.append(i)
				cols.append(j)

	n_rows = len(ids)
	n_cols = len(col_names)

	# Compute row counts (non-zeros per row)
	counts = [0] * n_rows
	for i in rows:
		counts[i] += 1

	# Save row counts CSV
	rc_path = out_prefix + ".row_counts.csv"
	os.makedirs(os.path.dirname(rc_path) or ".", exist_ok=True)
	with open(rc_path, "w", encoding=encoding, newline="") as f:
		w = csv.writer(f)
		w.writerow(["id", "count"])
		for _id, c in zip(ids, counts):
			w.writerow([_id, c])

	# Lp normalization per row: each non-zero value becomes 1 / (count**(1/p))
	data = np.empty(len(rows), dtype=np.float32)
	# Precompute row scales; avoid division by zero
	scales = np.ones(n_rows, dtype=np.float32)
	if p <= 0:
		raise ValueError("p must be > 0 for Lp normalization")
	for i in range(n_rows):
		c = counts[i]
		if c > 0:
			scales[i] = float(c) ** (-1.0 / float(p))
		else:
			scales[i] = 1.0
	for k, i in enumerate(rows):
		data[k] = scales[i]

	# Save COO as NPZ
	coo_path = out_prefix + ".coo.npz"
	np.savez_compressed(coo_path, rows=np.array(rows, dtype=np.int32), cols=np.array(cols, dtype=np.int32), data=data, shape=np.array([n_rows, n_cols], dtype=np.int64))

	# Optionally save columns list
	if save_columns:
		with open(out_prefix + ".columns.txt", "w", encoding=encoding) as f:
			for name in col_names:
				f.write(name + "\n")

	# Optionally save torch sparse tensor
	if save_torch:
		if torch is None:
			raise RuntimeError("PyTorch not available; install torch or omit --to-torch")
		if len(rows) == 0:
			i_tensor = torch.zeros((2, 0), dtype=torch.long)
			v_tensor = torch.zeros((0,), dtype=torch.float32)
		else:
			i_tensor = torch.tensor([rows, cols], dtype=torch.long)
			v_tensor = torch.tensor(data, dtype=torch.float32)
		sp = torch.sparse_coo_tensor(i_tensor, v_tensor, size=(n_rows, n_cols))
		torch.save(sp.coalesce(), out_prefix + ".pt")

	return n_rows, n_cols


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Filter rows in CSV where a column lacks a delimiter")
	parser.add_argument("--input", help="Input CSV path")
	parser.add_argument("--output", help="Output CSV path; if omitted with --in-place, input will be overwritten with backup")
	parser.add_argument("--column", default="newid", help="Column to check (default: newid)")
	parser.add_argument("--delimiter", default=";", help="Delimiter that must appear in the column (default: ';')")
	parser.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
	parser.add_argument("--in-place", action="store_true", help="Write results back to the input file (makes a .bak backup)")
	# New: expansion options
	parser.add_argument("--expand", action="store_true", help="Expand 'newid' into masked variants; output has columns 'newid_masked' and 'od'")
	parser.add_argument("--placeholder", default="（）", help="Placeholder used to replace a token when expanding (default: Chinese parentheses '（）')")
	# New: build large matrix options
	parser.add_argument("--make-matrix", action="store_true", help="Build a large matrix: first column from ids CSV 'id', other columns from expanded CSV 'newid_masked'")
	parser.add_argument("--ids-csv", help="CSV containing an 'id' column to populate the first column of the matrix")
	parser.add_argument("--expanded-csv", help="CSV containing 'newid_masked' column to define matrix column names")
	parser.add_argument("--fill", default="0", help="Cell fill value for the matrix (default: '0')")
	# New: indicator matrix options
	parser.add_argument("--make-indicator", action="store_true", help="Build 0/1 indicator matrix where cell(i,j)=1 iff id equals expanded od for column j")
	# New: sparse/normalized matrix options
	parser.add_argument("--make-sparse", action="store_true", help="Build sparse COO indicator matrix normalized by row Lp norm (default p=2)")
	parser.add_argument("--out-prefix", help="Output prefix for sparse artifacts (writes <prefix>.coo.npz, <prefix>.row_counts.csv, optional columns file)")
	parser.add_argument("--p", type=float, default=2.0, help="Row-wise Lp normalization p (default: 2.0)")
	parser.add_argument("--to-torch", action="store_true", help="Additionally save PyTorch COO tensor to <prefix>.pt (requires torch)")
	parser.add_argument("--save-columns", action="store_true", help="Also save column names to <prefix>.columns.txt (one per line)")

	args = parser.parse_args(argv)

	# Sparse normalized indicator matrix mode
	if args.make_sparse:
		if np is None:
			raise SystemExit("NumPy is required for --make-sparse; please install numpy")
		ids_csv = args.ids_csv
		expanded_csv = args.expanded_csv
		if not ids_csv or not expanded_csv:
			raise SystemExit("--make-sparse requires both --ids-csv and --expanded-csv")
		if not args.out_prefix:
			raise SystemExit("--make-sparse requires --out-prefix to specify output base path")
		n_rows, n_cols = build_indicator_sparse_and_normalized(
			ids_csv=ids_csv,
			expanded_csv=expanded_csv,
			out_prefix=args.out_prefix,
			id_column="id",
			expanded_column="newid_masked",
			od_column="od",
			p=args.p,
			save_torch=args.to_torch,
			save_columns=args.save_columns,
			encoding=args.encoding,
		)
		print(f"Sparse normalized matrix saved with prefix: {args.out_prefix}")
		print(f"Shape: {n_rows} x {n_cols} (rows x columns)")
		return 0

	# Indicator matrix mode
	if args.make_indicator:
		ids_csv = args.ids_csv
		expanded_csv = args.expanded_csv
		if not ids_csv or not expanded_csv:
			raise SystemExit("--make-indicator requires both --ids-csv and --expanded-csv")
		out_path = args.output or (expanded_csv + ".indicator.csv")
		n_ids, n_cols = build_indicator_matrix_from_expanded(
			ids_csv=ids_csv,
			expanded_csv=expanded_csv,
			output_csv=out_path,
			id_column="id",
			expanded_column="newid_masked",
			od_column="od",
			encoding=args.encoding,
		)
		print(f"Indicator matrix written to: {out_path}")
		print(f"Shape: {n_ids} x {n_cols} (rows x columns including 'id')")
		return 0

	# If make-matrix mode is selected
	if args.make_matrix:
		ids_csv = args.ids_csv
		expanded_csv = args.expanded_csv
		if not ids_csv or not expanded_csv:
			raise SystemExit("--make-matrix requires both --ids-csv and --expanded-csv")
		out_path = args.output or (expanded_csv + ".matrix.csv")
		n_ids, n_cols = build_large_matrix(
			ids_csv=ids_csv,
			expanded_csv=expanded_csv,
			output_csv=out_path,
			id_column="id",
			expanded_column="newid_masked",
			fill_value=args.fill,
			encoding=args.encoding,
		)
		print(f"Matrix written to: {out_path}")
		print(f"Shape: {n_ids} x {n_cols} (rows x columns including 'id')")
		return 0

	# If expand mode selected, run expansion first and return
	if args.expand:
		if not args.input:
			parser.error("--expand requires --input to be provided")
		out_path = args.output or (args.input + ".expanded.csv")
		in_rows, out_rows = expand_newid_to_csv(
			args.input,
			out_path,
			column=args.column,
			encoding=args.encoding,
			placeholder=args.placeholder,
		)
		print(f"Expanded variants written to: {out_path}")
		print(f"Input rows: {in_rows}\nOutput rows (masked variants): {out_rows}")
		return 0

	if args.in_place and args.output:
		parser.error("--output and --in-place are mutually exclusive")

	# Dry run or output to a new file
	if not args.in_place:
		if not args.input:
			parser.error("filter mode requires --input")
		total, kept, removed = filter_csv(
			args.input,
			args.output,
			delimiter=args.delimiter,
			column=args.column,
			encoding=args.encoding,
		)
		print(f"Total rows: {total}\nKept (with '{args.delimiter}'): {kept}\nRemoved: {removed}")
		if args.output:
			print(f"Written filtered CSV to: {args.output}")
		return 0

	# In-place: backup then write to a temp file and move over
	if not args.input:
		parser.error("--in-place requires --input")
	backup = _make_backup(args.input)
	tmp_out = args.input + ".tmp"
	total, kept, removed = filter_csv(
		args.input,
		tmp_out,
		delimiter=args.delimiter,
		column=args.column,
		encoding=args.encoding,
	)
	os.replace(tmp_out, args.input)
	print(f"In-place filtering completed. Backup at: {backup}")
	print(f"Total rows: {total}\nKept (with '{args.delimiter}'): {kept}\nRemoved: {removed}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

