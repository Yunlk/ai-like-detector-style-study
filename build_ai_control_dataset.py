"""
Build the modeling dataset from real published abstracts and AI-control outputs.

Inputs:
    1. real_published_abstracts_clean.csv
    2. real_published_abstracts_ai_controls_generated.csv

Output:
    all_abstract_variants.csv

The output contains one published human/unknown-assistance row per source text,
plus AI-generated and AI-polished control rows for successful generations.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


AI_LABELS = {
    "ai_generate_from_metadata": {
        "text_variant": "ai_generate_from_metadata",
        "known_ai_status": "ai_generated_control",
        "target_is_ai_generated": "1",
        "target_is_ai_assisted": "1",
    },
    "ai_polish_original_abstract": {
        "text_variant": "ai_polish_original_abstract",
        "known_ai_status": "ai_polished_control",
        "target_is_ai_generated": "1",
        "target_is_ai_assisted": "1",
    },
}


OUTPUT_FIELDS = [
    "source_id",
    "doi",
    "year",
    "period",
    "language",
    "discipline",
    "journal",
    "title",
    "abstract",
    "text_variant",
    "known_ai_status",
    "target_is_ai_generated",
    "target_is_ai_assisted",
    "source_type",
    "word_count",
    "generation_model",
    "generation_status",
    "generation_error",
    "generated_at",
    "raw_response_id",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_word_count(text: str) -> str:
    return str(len(text.split()))


def real_row_to_output(row: dict[str, str]) -> dict[str, str]:
    abstract = row.get("abstract", "")
    return {
        "source_id": row.get("id", ""),
        "doi": row.get("doi", ""),
        "year": row.get("year", ""),
        "period": row.get("period", ""),
        "language": row.get("language", ""),
        "discipline": row.get("discipline", ""),
        "journal": row.get("journal", ""),
        "title": row.get("title", ""),
        "abstract": abstract,
        "text_variant": row.get("text_variant", "published_abstract") or "published_abstract",
        "known_ai_status": row.get("known_ai_status", ""),
        "target_is_ai_generated": row.get("target_is_ai_generated", "0") or "0",
        "target_is_ai_assisted": "0",
        "source_type": row.get("source_type", ""),
        "word_count": row.get("word_count", "") or normalize_word_count(abstract),
        "generation_model": "",
        "generation_status": "",
        "generation_error": "",
        "generated_at": "",
        "raw_response_id": "",
    }


def ai_row_to_output(ai_row: dict[str, str], real_by_id: dict[str, dict[str, str]]) -> dict[str, str] | None:
    source_id = ai_row.get("source_id", "")
    prompt_type = ai_row.get("prompt_type", "")
    label = AI_LABELS.get(prompt_type)
    real = real_by_id.get(source_id)
    if label is None or real is None:
        return None

    abstract = ai_row.get("generated_text", "").strip()
    return {
        "source_id": source_id,
        "doi": real.get("doi", ""),
        "year": real.get("year", ai_row.get("year", "")),
        "period": real.get("period", ai_row.get("period", "")),
        "language": real.get("language", ""),
        "discipline": real.get("discipline", ""),
        "journal": real.get("journal", ""),
        "title": real.get("title", ai_row.get("title", "")),
        "abstract": abstract,
        "text_variant": label["text_variant"],
        "known_ai_status": label["known_ai_status"],
        "target_is_ai_generated": label["target_is_ai_generated"],
        "target_is_ai_assisted": label["target_is_ai_assisted"],
        "source_type": "siliconflow_ai_control",
        "word_count": normalize_word_count(abstract),
        "generation_model": ai_row.get("model", ""),
        "generation_status": ai_row.get("status", ""),
        "generation_error": ai_row.get("error", ""),
        "generated_at": ai_row.get("generated_at", ""),
        "raw_response_id": ai_row.get("raw_response_id", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, help="Clean real published abstracts CSV")
    parser.add_argument("--generated", required=True, help="AI-control generated CSV")
    parser.add_argument("--out", required=True, help="Output merged CSV")
    parser.add_argument("--allow-errors", action="store_true", help="Skip failed AI rows instead of exiting")
    args = parser.parse_args()

    real_rows = read_csv(Path(args.real))
    generated_rows = read_csv(Path(args.generated))
    real_by_id = {row.get("id", ""): row for row in real_rows}

    status_counts: dict[str, int] = {}
    for row in generated_rows:
        status = row.get("status", "")
        status_counts[status] = status_counts.get(status, 0) + 1

    failed = [row for row in generated_rows if row.get("status") != "ok" or not row.get("generated_text", "").strip()]
    if failed and not args.allow_errors:
        print("Generated control file is not complete.", file=sys.stderr)
        print(f"Status counts: {status_counts}", file=sys.stderr)
        for row in failed[:10]:
            print(
                f"- {row.get('status')} | {row.get('prompt_type')} | "
                f"{row.get('title', '')[:80]} | {row.get('error', '')[:200]}",
                file=sys.stderr,
            )
        return 2

    output_rows: list[dict[str, str]] = [real_row_to_output(row) for row in real_rows]
    skipped_missing_real = 0
    skipped_failed = 0

    for row in generated_rows:
        if row.get("status") != "ok" or not row.get("generated_text", "").strip():
            skipped_failed += 1
            continue
        converted = ai_row_to_output(row, real_by_id)
        if converted is None:
            skipped_missing_real += 1
            continue
        output_rows.append(converted)

    write_csv(Path(args.out), output_rows, OUTPUT_FIELDS)

    variant_counts: dict[str, int] = {}
    for row in output_rows:
        variant = row.get("text_variant", "")
        variant_counts[variant] = variant_counts.get(variant, 0) + 1

    print(f"Real rows: {len(real_rows)}")
    print(f"Generated rows: {len(generated_rows)}")
    print(f"Generated status counts: {status_counts}")
    print(f"Output rows: {len(output_rows)}")
    print(f"Variant counts: {variant_counts}")
    print(f"Skipped failed generated rows: {skipped_failed}")
    print(f"Skipped rows without matching real source_id: {skipped_missing_real}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
