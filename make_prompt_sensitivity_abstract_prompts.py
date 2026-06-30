"""
Create prompt-style sensitivity tasks on the same batch of real abstracts.

Output is compatible with call_siliconflow_controls_multimodel.py.
Recommended small experiment:

    50 source abstracts x 4 prompt styles x 2 models = 400 generations
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


PROMPT_STYLES: dict[str, str] = {
    "neutral_metadata": (
        "Write a scholarly abstract in English for a research paper using only the metadata below. "
        "Do not claim specific results beyond what can be reasonably inferred from the title and field. "
        "Use a concise academic style."
    ),
    "standard_academic": (
        "Write a polished journal-style English abstract for a research paper using only the metadata below. "
        "Use a conventional academic structure with background, objective, methods, results, and conclusion. "
        "Make the writing clear, formal, and publication-ready."
    ),
    "over_template": (
        "Write a highly standardized and formulaic English academic abstract using only the metadata below. "
        "Use explicit template phrases such as 'This study aims', 'The results suggest', and 'These findings indicate'. "
        "Keep the wording cautious, balanced, and typical of a generic scientific abstract."
    ),
    "anti_template": (
        "Write an English academic abstract using only the metadata below, but avoid generic AI-like academic phrasing. "
        "Do not use stock phrases such as 'This study aims', 'The results suggest', or 'These findings indicate'. "
        "Vary sentence length, avoid overly symmetrical structure, and keep the abstract specific and restrained. "
        "Do not become informal."
    ),
}


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
    "prompt_style",
    "prompt",
    "model",
    "model_assignment_method",
    "generated_text",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_models(value: str) -> list[str]:
    models = [item.strip() for item in value.split(",") if item.strip()]
    if not models:
        raise SystemExit("Provide at least one model via --models.")
    return models


def build_prompt(row: dict[str, str], style_name: str, style_instruction: str, max_words: int) -> str:
    return f"""{style_instruction}

Metadata:
Title: {row.get("title", "").strip()}
Year: {row.get("year", "").strip()}
Discipline / topic: {row.get("discipline", "").strip()}
Journal or source: {row.get("journal", "").strip()}
Source platform: {row.get("source_platform", "").strip()}

Constraints:
- Output only the abstract.
- Do not include a title, references, bullet points, or section labels.
- Keep the abstract between 120 and {max_words} words.
- Do not copy or paraphrase the original published abstract, because it is not provided to you.
- Prompt style label for the experiment: {style_name}
"""


def choose_rows(rows: list[dict[str, str]], sample_size: int, seed: int) -> list[dict[str, str]]:
    candidates = [
        row
        for row in rows
        if row.get("language", "").lower() == "en"
        and row.get("title", "").strip()
        and row.get("abstract", "").strip()
        and 60 <= int(row.get("word_count") or len(row.get("abstract", "").split())) <= 260
    ]
    if len(candidates) < sample_size:
        raise SystemExit(f"Only {len(candidates)} usable rows found, fewer than --sample-size {sample_size}.")

    rng = random.Random(seed)
    by_period: dict[str, list[dict[str, str]]] = {}
    for row in candidates:
        by_period.setdefault(row.get("period", "unknown"), []).append(row)

    selected: list[dict[str, str]] = []
    periods = [period for period in ["pre_chatgpt", "post_chatgpt"] if period in by_period]
    if len(periods) >= 2:
        first_n = sample_size // 2
        second_n = sample_size - first_n
        for period, count in zip(periods[:2], [first_n, second_n]):
            period_rows = by_period[period][:]
            rng.shuffle(period_rows)
            selected.extend(period_rows[:count])
    else:
        rng.shuffle(candidates)
        selected = candidates[:sample_size]

    rng.shuffle(selected)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--models", default="deepseek-ai/DeepSeek-V4-Pro,tencent/Hunyuan-MT-7B")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--max-words", type=int, default=180)
    args = parser.parse_args()

    rows = read_csv(Path(args.input))
    source_rows = choose_rows(rows, args.sample_size, args.seed)
    models = parse_models(args.models)

    output_rows: list[dict[str, str]] = []
    for source in source_rows:
        for style_name, style_instruction in PROMPT_STYLES.items():
            prompt = build_prompt(source, style_name, style_instruction, args.max_words)
            for model in models:
                output_rows.append(
                    {
                        "source_id": source.get("id", source.get("source_id", "")),
                        "doi": source.get("doi", ""),
                        "year": source.get("year", ""),
                        "period": source.get("period", ""),
                        "language": source.get("language", ""),
                        "discipline": source.get("discipline", ""),
                        "journal": source.get("journal", ""),
                        "title": source.get("title", ""),
                        "source_platform": source.get("source_platform", ""),
                        "platform_type": source.get("platform_type", ""),
                        "publisher": source.get("publisher", ""),
                        "is_preprint": source.get("is_preprint", ""),
                        "source_type": "published_abstract",
                        "prompt_type": "prompt_sensitivity_generate_from_metadata",
                        "prompt_style": style_name,
                        "prompt": prompt,
                        "model": model,
                        "model_assignment_method": "full_factorial_prompt_style_x_model",
                        "generated_text": "",
                    }
                )

    write_csv(Path(args.out), output_rows)
    print(
        f"Wrote {len(output_rows)} prompt rows "
        f"({len(source_rows)} sources x {len(PROMPT_STYLES)} styles x {len(models)} models) to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
