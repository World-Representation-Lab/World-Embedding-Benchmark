from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render similarity matrix CSVs as PNG heatmaps.")
    parser.add_argument("matrix_dir", help="Directory containing overall.csv and family_*.csv files.")
    parser.add_argument("--output-dir", help="Defaults to matrix_dir.")
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--figsize-scale", type=float, default=0.45)
    parser.add_argument("--min-figsize", type=float, default=6.0)
    parser.add_argument("--max-figsize", type=float, default=28.0)
    parser.add_argument("--annotate", action="store_true", help="Draw values inside cells. Best for small matrices.")
    parser.add_argument("--center-diagonal", action="store_true", help="Subtract each row's diagonal score before plotting.")
    parser.add_argument("--global-scale", action="store_true", help="Use one color scale across all CSVs. By default each PNG uses its own min/max for better contrast.")
    parser.add_argument("--cmap", default="viridis_r")
    return parser.parse_args()


def read_matrix(path: Path) -> tuple[list[str], list[str], list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 3 or header[0] != "query_id" or header[1] != "query_family":
            raise ValueError(f"Unexpected matrix header in {path}: {header[:3]}")
        col_ids = header[2:]
        row_ids: list[str] = []
        row_families: list[str] = []
        values: list[list[float]] = []
        for row in reader:
            row_ids.append(row[0])
            row_families.append(row[1])
            values.append([float(x) for x in row[2:]])
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (len(row_ids), len(col_ids)):
        raise ValueError(f"Bad matrix shape in {path}: {matrix.shape}")
    return row_ids, row_families, col_ids, matrix


def short_labels(labels: list[str]) -> list[str]:
    return [label if len(label) <= 28 else label[:25] + "..." for label in labels]


def maybe_center_diagonal(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("--center-diagonal requires a square matrix")
    return matrix - np.diag(matrix)[:, None]


def plot_matrix(
    *,
    csv_path: Path,
    output_path: Path,
    row_ids: list[str],
    col_ids: list[str],
    matrix: np.ndarray,
    args: argparse.Namespace,
    vmin: float | None,
    vmax: float | None,
) -> None:
    import matplotlib.pyplot as plt

    matrix = matrix * 100.0
    if vmin is not None:
        vmin = vmin * 100.0
    if vmax is not None:
        vmax = vmax * 100.0
    if vmin is None:
        vmin = float(np.nanmin(matrix))
    if vmax is None:
        vmax = float(np.nanmax(matrix))

    n_rows, n_cols = matrix.shape
    width = min(args.max_figsize, max(args.min_figsize, n_cols * args.figsize_scale + 3.0))
    height = min(args.max_figsize, max(args.min_figsize, n_rows * args.figsize_scale + 2.5))
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap=args.cmap, vmin=vmin, vmax=vmax)
    ax.set_title(csv_path.stem)
    ax.set_xlabel("video")
    ax.set_ylabel("text query")
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(short_labels(col_ids), rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(short_labels(row_ids), fontsize=7)

    if args.annotate and n_rows * n_cols <= 400:
        threshold = (float(np.nanmin(matrix)) + float(np.nanmax(matrix))) / 2.0
        for i in range(n_rows):
            for j in range(n_cols):
                color = "white" if matrix[i, j] < threshold else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=5, color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("cosine similarity x100")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    matrix_dir = Path(args.matrix_dir)
    output_dir = Path(args.output_dir) if args.output_dir else matrix_dir
    csv_paths = sorted(matrix_dir.glob(args.pattern))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matched {args.pattern!r} in {matrix_dir}")

    loaded = []
    for path in csv_paths:
        row_ids, row_families, col_ids, matrix = read_matrix(path)
        if args.center_diagonal:
            matrix = maybe_center_diagonal(matrix)
        loaded.append((path, row_ids, row_families, col_ids, matrix))

    vmin = vmax = None
    if args.global_scale:
        vmin = min(float(np.nanmin(item[4])) for item in loaded)
        vmax = max(float(np.nanmax(item[4])) for item in loaded)

    for path, row_ids, _, col_ids, matrix in loaded:
        output_path = output_dir / f"{path.stem}.png"
        plot_matrix(
            csv_path=path,
            output_path=output_path,
            row_ids=row_ids,
            col_ids=col_ids,
            matrix=matrix,
            args=args,
            vmin=vmin,
            vmax=vmax,
        )
        print(output_path)


if __name__ == "__main__":
    main()
