"""
Relabel the pilot dataset with a stricter three-way text-origin framework.

The v1 dataset treated AI-polished text as target_is_ai_generated=1. For the
paper's refined definition, AI polishing preserves human semantic authorship but
changes surface style. This script keeps the text unchanged and writes v2 labels.

Text-origin definitions:
    published_abstract:
        text_origin = human_original
        target_is_ai_generated = 0
        target_is_ai_assisted = 0

    ai_generate_from_metadata:
        text_origin = ai_generated
        target_is_ai_generated = 1
        target_is_ai_assisted = 1

    ai_polish_original_abstract:
        text_origin = human_semantics_ai_polished
        target_is_ai_generated = 0
        target_is_ai_assisted = 1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


TEXT_ORIGIN_BY_VARIANT = {
    "published_abstract": {
        "text_origin": "human_original",
        "text_origin_cn": "人类原始/真实发表摘要",
        "target_is_ai_generated": "0",
        "target_is_ai_assisted": "0",
        "semantic_author": "human",
        "surface_editor": "human_or_unknown",
        "training_role": "train_negative_for_ai_generated_detector",
    },
    "ai_generate_from_metadata": {
        "text_origin": "ai_generated",
        "text_origin_cn": "AI根据元数据生成摘要",
        "target_is_ai_generated": "1",
        "target_is_ai_assisted": "1",
        "semantic_author": "ai",
        "surface_editor": "ai",
        "training_role": "train_positive_for_ai_generated_detector",
    },
    "ai_polish_original_abstract": {
        "text_origin": "human_semantics_ai_polished",
        "text_origin_cn": "人类语义主导+AI语言润色",
        "target_is_ai_generated": "0",
        "target_is_ai_assisted": "1",
        "semantic_author": "human",
        "surface_editor": "ai",
        "training_role": "heldout_ai_assisted_human_text",
    },
}


EXTRA_FIELDS = [
    "text_origin",
    "text_origin_cn",
    "semantic_author",
    "surface_editor",
    "training_role",
    "label_version",
    "label_note_cn",
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


def relabel_row(row: dict[str, str]) -> dict[str, str]:
    variant = row.get("text_variant", "")
    mapping = TEXT_ORIGIN_BY_VARIANT.get(variant)
    if mapping is None:
        raise ValueError(f"Unknown text_variant: {variant}")

    output = dict(row)
    output.update(mapping)
    output["label_version"] = "v2_human_semantics_ai_polish_not_ai_generated"

    if variant == "ai_polish_original_abstract":
        output["known_ai_status"] = "ai_assisted_human_authored_control"
        output["label_note_cn"] = (
            "该文本由AI润色，但语义来源为原始人类摘要；因此不作为严格AI生成文本，"
            "而作为AI辅助人类文本单独分析。"
        )
    elif variant == "ai_generate_from_metadata":
        output["known_ai_status"] = "ai_generated_control"
        output["label_note_cn"] = "该文本由AI根据标题/年份/时期等元数据生成，作为AI生成正例。"
    else:
        output["label_note_cn"] = (
            "该文本为真实发表摘要；pre_chatgpt可作为历史阴性对照，"
            "post_chatgpt的AI辅助情况未知。"
        )

    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input all_abstract_variants.csv")
    parser.add_argument("--out", required=True, help="Output all_abstract_variants_v2.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.out)
    rows = read_csv(input_path)
    if not rows:
        raise SystemExit("Input CSV has no rows.")

    relabeled = [relabel_row(row) for row in rows]

    fieldnames = list(rows[0].keys())
    for field in EXTRA_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    write_csv(output_path, relabeled, fieldnames)

    counts: dict[str, int] = {}
    origin_counts: dict[str, int] = {}
    generated_counts: dict[str, int] = {}
    assisted_counts: dict[str, int] = {}
    for row in relabeled:
        counts[row.get("text_variant", "")] = counts.get(row.get("text_variant", ""), 0) + 1
        origin_counts[row.get("text_origin", "")] = origin_counts.get(row.get("text_origin", ""), 0) + 1
        generated_counts[row.get("target_is_ai_generated", "")] = (
            generated_counts.get(row.get("target_is_ai_generated", ""), 0) + 1
        )
        assisted_counts[row.get("target_is_ai_assisted", "")] = (
            assisted_counts.get(row.get("target_is_ai_assisted", ""), 0) + 1
        )

    print(f"Read rows: {len(rows)}")
    print(f"Wrote rows: {len(relabeled)}")
    print(f"text_variant counts: {counts}")
    print(f"text_origin counts: {origin_counts}")
    print(f"target_is_ai_generated counts: {generated_counts}")
    print(f"target_is_ai_assisted counts: {assisted_counts}")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
