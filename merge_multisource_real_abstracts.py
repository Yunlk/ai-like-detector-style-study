"""
Merge and de-duplicate multi-source real abstract CSV files.

Deduplication priority:
    1. DOI, when available
    2. source_platform + id

The script also normalizes period labels from year:
    - year <= 2021: pre_chatgpt
    - year == 2022: transition_2022
    - year >= 2023: post_chatgpt
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_doi(doi: str) -> str:
    value = (doi or "").strip().lower()
    value = value.replace("https://doi.org/", "").replace("http://doi.org/", "")
    value = value.replace("doi:", "").strip()
    return value


def parse_year(row: dict[str, str]) -> int:
    raw = row.get("year", "")
    match = re.search(r"\d{4}", raw or "")
    return int(match.group(0)) if match else 0


def period_for_year(year: int) -> str:
    if year and year <= 2021:
        return "pre_chatgpt"
    if year == 2022:
        return "transition_2022"
    if year >= 2023:
        return "post_chatgpt"
    return "unknown"


def dedupe_key(row: dict[str, str]) -> tuple[str, str]:
    doi = clean_doi(row.get("doi", ""))
    if doi:
        return ("doi", doi)
    return (row.get("source_platform", ""), row.get("id", ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="Comma-separated input CSV paths")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dropped-out", default="", help="Optional CSV of duplicate dropped rows")
    args = parser.parse_args()

    input_paths = [Path(part.strip()) for part in args.inputs.split(",") if part.strip()]
    if not input_paths:
        raise SystemExit("No input files.")

    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for path in input_paths:
        current = read_csv(path)
        rows.extend(current)
        for field in current[0].keys() if current else []:
            if field not in fieldnames:
                fieldnames.append(field)

    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    for row in rows:
        year = parse_year(row)
        row["year"] = str(year) if year else row.get("year", "")
        row["period"] = period_for_year(year)
        row["group"] = f"real_{row['period']}" if row["period"] != "unknown" else row.get("group", "")

        key = dedupe_key(row)
        if key in seen:
            dropped.append(row)
            continue
        seen.add(key)
        merged.append(row)

    write_csv(Path(args.out), merged, fieldnames)
    if args.dropped_out:
        write_csv(Path(args.dropped_out), dropped, fieldnames)

    counts: dict[str, int] = {}
    platform_counts: dict[str, int] = {}
    year_counts: dict[str, int] = {}
    for row in merged:
        counts[row.get("period", "")] = counts.get(row.get("period", ""), 0) + 1
        platform_counts[row.get("source_platform", "")] = platform_counts.get(row.get("source_platform", ""), 0) + 1
        year_counts[row.get("year", "")] = year_counts.get(row.get("year", ""), 0) + 1

    print(f"Input rows: {len(rows)}")
    print(f"Merged rows: {len(merged)}")
    print(f"Dropped duplicates: {len(dropped)}")
    print(f"Period counts: {counts}")
    print(f"Platform counts: {platform_counts}")
    print(f"Year counts: {dict(sorted(year_counts.items()))}")
    print(f"Wrote {args.out}")
    if args.dropped_out:
        print(f"Wrote dropped rows to {args.dropped_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
