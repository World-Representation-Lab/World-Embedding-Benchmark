from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


TEXT_COLUMNS = ("raw_text", "parsed_text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve the ordered fluid evaluation rows/videos while replacing "
            "captions from the ID-matched correct-caption dataset."
        )
    )
    parser.add_argument("--ordered-source", required=True)
    parser.add_argument("--caption-source", required=True)
    parser.add_argument("--output", default="datasets/World-Embedding-Fluid-Retrieval")
    return parser.parse_args()


def parquet_files(root: Path) -> list[Path]:
    files = sorted(root.glob("*/test-*.parquet"))
    if not files:
        files = sorted(root.glob("*/train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No family Parquet files found under {root}")
    return files


def load_caption_map(root: Path) -> dict[str, tuple[str, str, str]]:
    by_case_id: dict[str, tuple[str, str, str]] = {}
    query_ids: set[str] = set()
    for path in parquet_files(root):
        table = pq.read_table(path, columns=["query_id", "case_id", *TEXT_COLUMNS])
        for row in table.to_pylist():
            query_id = str(row["query_id"])
            case_id = str(row["case_id"])
            if query_id in query_ids:
                raise ValueError(f"Duplicate query_id in caption source: {query_id}")
            if case_id in by_case_id:
                raise ValueError(f"Duplicate case_id in caption source: {case_id}")
            query_ids.add(query_id)
            by_case_id[case_id] = (
                query_id,
                str(row["raw_text"]),
                str(row["parsed_text"]),
            )
    return by_case_id


def merge(ordered_root: Path, caption_root: Path, output_root: Path) -> dict[str, object]:
    if output_root.resolve() in {ordered_root.resolve(), caption_root.resolve()}:
        raise ValueError("Output must not overwrite either source dataset")
    caption_by_case = load_caption_map(caption_root)
    ordered_files = parquet_files(ordered_root)
    ordered_case_ids: set[str] = set()
    ordered_query_ids: set[str] = set()
    family_rows: dict[str, int] = {}

    for source_path in ordered_files:
        relative = source_path.relative_to(ordered_root)
        output_path = output_root / relative
        table = pq.read_table(source_path)
        rows = table.select(["query_id", "case_id"]).to_pylist()
        raw_text: list[str] = []
        parsed_text: list[str] = []
        for row in rows:
            query_id = str(row["query_id"])
            case_id = str(row["case_id"])
            if query_id in ordered_query_ids:
                raise ValueError(f"Duplicate ordered query_id: {query_id}")
            if case_id in ordered_case_ids:
                raise ValueError(f"Duplicate ordered case_id: {case_id}")
            ordered_query_ids.add(query_id)
            ordered_case_ids.add(case_id)
            try:
                caption_query_id, raw, parsed = caption_by_case[case_id]
            except KeyError as exc:
                raise ValueError(f"No caption row for case_id {case_id}") from exc
            if caption_query_id != query_id:
                raise ValueError(
                    f"query_id mismatch for {case_id}: {query_id} != {caption_query_id}"
                )
            raw_text.append(raw)
            parsed_text.append(parsed)

        for column_name, values in zip(TEXT_COLUMNS, (raw_text, parsed_text), strict=True):
            index = table.schema.get_field_index(column_name)
            if index < 0:
                raise ValueError(f"Missing {column_name} in {source_path}")
            field = table.schema.field(index)
            table = table.set_column(index, field, pa.array(values, type=field.type))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        pq.write_table(table, temporary, compression="snappy")
        os.replace(temporary, output_path)
        family_rows[relative.parent.as_posix()] = table.num_rows
        print(f"Wrote {output_path} ({table.num_rows} rows)")

    caption_case_ids = set(caption_by_case)
    if ordered_case_ids != caption_case_ids:
        raise ValueError(
            "Source case_id sets differ after merge: "
            f"ordered-only={len(ordered_case_ids - caption_case_ids)}, "
            f"caption-only={len(caption_case_ids - ordered_case_ids)}"
        )

    for name in ("README.md", ".gitattributes"):
        source = ordered_root / name
        if source.exists():
            output_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output_root / name)

    manifest = {
        "ordered_source": str(ordered_root),
        "caption_source": str(caption_root),
        "mapping_key": "case_id (with query_id equality validation)",
        "preserved_columns": ["query_id", "case_id", "video"],
        "replaced_columns": list(TEXT_COLUMNS),
        "num_rows": len(ordered_case_ids),
        "family_rows": family_rows,
    }
    manifest_path = output_root / "merge_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest_path)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = merge(Path(args.ordered_source), Path(args.caption_source), Path(args.output))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
