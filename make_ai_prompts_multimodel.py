"""
Create AI-control prompts with per-row model assignment.

For each real abstract, this script creates:
    1. ai_generate_from_metadata
    2. ai_polish_original_abstract

Models are assigned by round-robin by default, using the provided model list.
The output is compatible with call_siliconflow_controls_multimodel.py.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


GENERATE_TEMPLATE = """Write a scholarly abstract in English for a research paper with the following metadata.

Title: {title}
Field or topic: {discipline}
Journal or source: {journal}
Source platform: {source_platform}

Constraints:
- Write 120 to 220 words.
- Use a conventional academic abstract style.
- Do not invent author names, affiliations, citations, funding, DOI, exact sample size, or exact numerical results unless they are implied by the metadata.
- Do not mention that you are an AI model.
- Output only the abstract.
"""


POLISH_TEMPLATE = """Polish the following published abstract into fluent, conventional academic English.

Constraints:
- Preserve the original meaning, claims, methods, and results.
- Do not add new findings, citations, author names, affiliations, funding, DOI, exact sample size, or exact numerical results.
- Keep roughly the same length.
- Do not mention that you are an AI model.
- Output only the polished abstract.

Original abstract:
{abstract}
"""


FIELDNAMES = [
    "source_id",
    "doi",
    "year",
    "period",
    "language",
    "discipline",
    "journal",
    "title",
    "source_platform",
    "platform_type",
    "publisher",
    "is_preprint",
    "source_type",
    "prompt_type",
    "prompt",
    "model",
    "model_assignment_method",
    "generated_text",
]


def parse_models(models: str) -> list[str]:
    parsed = [model.strip() for model in models.split(",") if model.strip()]
    if not parsed:
        raise SystemExit("No models provided. Use --models model1,model2,...")
    return parsed


def choose_model(models: list[str], index: int, method: str) -> str:
    if method == "round_robin":
        return models[index % len(models)]
    if method == "random":
        return random.choice(models)
    raise SystemExit(f"Unknown assignment method: {method}")


def base_prompt_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "source_id": row.get("id") or row.get("doi") or row.get("title", ""),
        "doi": row.get("doi", ""),
        "year": row.get("year", ""),
        "period": row.get("period", ""),
        "language": row.get("language", ""),
        "discipline": row.get("discipline", ""),
        "journal": row.get("journal", ""),
        "title": row.get("title", ""),
        "source_platform": row.get("source_platform", ""),
        "platform_type": row.get("platform_type", ""),
        "publisher": row.get("publisher", ""),
        "is_preprint": row.get("is_preprint", ""),
        "source_type": row.get("source_type", ""),
        "generated_text": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Real abstract CSV")
    parser.add_argument("--out", required=True, help="Prompt CSV")
    parser.add_argument("--models", required=True, help="Comma-separated SiliconFlow model names")
    parser.add_argument("--assignment", choices=["round_robin", "random"], default="round_robin")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=0, help="Optional number of real abstracts to use")
    args = parser.parse_args()

    random.seed(args.seed)
    models = parse_models(args.models)

    src = Path(args.input)
    dst = Path(args.out)
    with src.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if args.limit:
        rows = rows[: args.limit]

    prompt_rows: list[dict[str, str]] = []
    assignment_index = 0
    for row in rows:
        base = base_prompt_row(row)

        generate_model = choose_model(models, assignment_index, args.assignment)
        assignment_index += 1
        generate_row = dict(base)
        generate_row.update(
            {
                "prompt_type": "ai_generate_from_metadata",
                "prompt": GENERATE_TEMPLATE.format(
                    title=row.get("title", ""),
                    discipline=row.get("discipline", ""),
                    journal=row.get("journal", ""),
                    source_platform=row.get("source_platform", ""),
                ),
                "model": generate_model,
                "model_assignment_method": args.assignment,
            }
        )
        prompt_rows.append(generate_row)

        polish_model = choose_model(models, assignment_index, args.assignment)
        assignment_index += 1
        polish_row = dict(base)
        polish_row.update(
            {
                "prompt_type": "ai_polish_original_abstract",
                "prompt": POLISH_TEMPLATE.format(abstract=row.get("abstract", "")),
                "model": polish_model,
                "model_assignment_method": args.assignment,
            }
        )
        prompt_rows.append(polish_row)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(prompt_rows)

    counts: dict[str, int] = {}
    prompt_type_counts: dict[str, int] = {}
    for row in prompt_rows:
        counts[row["model"]] = counts.get(row["model"], 0) + 1
        prompt_type_counts[row["prompt_type"]] = prompt_type_counts.get(row["prompt_type"], 0) + 1

    print(f"Real input rows: {len(rows)}")
    print(f"Prompt rows: {len(prompt_rows)}")
    print(f"Models: {counts}")
    print(f"Prompt types: {prompt_type_counts}")
    print(f"Wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
