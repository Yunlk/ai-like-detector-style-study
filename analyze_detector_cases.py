"""
Export misclassification and edge-case tables from detector predictions.

Input:
    local_detector_results/predictions.csv

Outputs:
    - false_positive_pre_chatgpt.csv
    - false_positive_post_chatgpt.csv
    - top_ai_like_published_cases.csv
    - ai_polish_low_score_cases.csv
    - ai_polish_high_score_cases.csv
    - ai_generate_low_score_cases.csv
    - all_case_flags.csv
    - case_summary.csv

The output keeps full abstracts for close reading and adds Chinese notes that
describe why a case is worth inspecting.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


STYLE_FEATURES = [
    "computed_word_count",
    "sentence_count",
    "sentence_length_mean",
    "sentence_length_std",
    "lexical_diversity",
    "hapax_ratio",
    "bigram_repetition",
    "trigram_repetition",
    "connector_ratio",
    "template_phrase_count",
    "punctuation_ratio",
    "token_entropy",
]

CASE_FIELDS = [
    "case_type",
    "case_type_cn",
    "score_column",
    "ai_like_score",
    "score_band",
    "inspection_note_cn",
    "style_reason_cn",
    "source_id",
    "doi",
    "year",
    "period",
    "language",
    "discipline",
    "journal",
    "title",
    "text_variant",
    "known_ai_status",
    "target_is_ai_generated",
    "target_is_ai_assisted",
    "generation_model",
    "word_count",
    "computed_word_count",
    "sentence_count",
    "sentence_length_mean",
    "sentence_length_std",
    "lexical_diversity",
    "hapax_ratio",
    "bigram_repetition",
    "trigram_repetition",
    "connector_ratio",
    "template_phrase_count",
    "punctuation_ratio",
    "token_entropy",
    "abstract_excerpt",
    "abstract",
]

SUMMARY_FIELDS = [
    "score_column",
    "group",
    "group_cn",
    "n",
    "mean_score",
    "median_score",
    "min_score",
    "max_score",
    "share_ge_high_threshold",
    "share_lt_low_threshold",
]

GROUP_CN = {
    "published_pre_chatgpt": "ChatGPT普及前真实摘要",
    "published_post_chatgpt": "ChatGPT普及后真实摘要",
    "published_all": "全部真实发表摘要",
    "ai_generate_from_metadata": "AI根据元数据生成摘要",
    "ai_polish_original_abstract": "AI润色原始摘要",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: str) -> float | None:
    try:
        if value == "":
            return None
        number = float(value)
        if math.isnan(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def score_of(row: dict[str, str], score_col: str) -> float:
    return to_float(row.get(score_col, "")) or 0.0


def short_text(text: str, limit: int = 360) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def median_feature(rows: list[dict[str, str]], feature: str) -> float:
    values = [to_float(row.get(feature, "")) for row in rows]
    clean = [value for value in values if value is not None]
    return statistics.median(clean) if clean else 0.0


def score_band(score: float, high_threshold: float, very_high_threshold: float) -> str:
    if score >= very_high_threshold:
        return "very_high"
    if score >= high_threshold:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"


def group_rows(rows: list[dict[str, str]], group: str) -> list[dict[str, str]]:
    if group == "published_pre_chatgpt":
        return [
            row
            for row in rows
            if row.get("text_variant") == "published_abstract" and row.get("period") == "pre_chatgpt"
        ]
    if group == "published_post_chatgpt":
        return [
            row
            for row in rows
            if row.get("text_variant") == "published_abstract" and row.get("period") == "post_chatgpt"
        ]
    if group == "published_all":
        return [row for row in rows if row.get("text_variant") == "published_abstract"]
    return [row for row in rows if row.get("text_variant") == group]


def build_reference(rows: list[dict[str, str]]) -> dict[str, float]:
    published = group_rows(rows, "published_all")
    return {feature: median_feature(published, feature) for feature in STYLE_FEATURES}


def style_reason(row: dict[str, str], reference: dict[str, float], ai_like: bool) -> str:
    """Generate a compact Chinese explanation from the known style-feature directions."""
    reasons: list[str] = []

    token_entropy = to_float(row.get("token_entropy", "")) or 0.0
    hapax_ratio = to_float(row.get("hapax_ratio", "")) or 0.0
    connector_ratio = to_float(row.get("connector_ratio", "")) or 0.0
    sentence_mean = to_float(row.get("sentence_length_mean", "")) or 0.0
    sentence_std = to_float(row.get("sentence_length_std", "")) or 0.0
    trigram_rep = to_float(row.get("trigram_repetition", "")) or 0.0
    bigram_rep = to_float(row.get("bigram_repetition", "")) or 0.0
    punctuation = to_float(row.get("punctuation_ratio", "")) or 0.0
    word_count = to_float(row.get("computed_word_count", "")) or 0.0

    if ai_like:
        if token_entropy > reference.get("token_entropy", 0.0):
            reasons.append("词汇分布熵高于真实摘要中位数")
        if hapax_ratio > reference.get("hapax_ratio", 0.0):
            reasons.append("低频/一次性词比例较高")
        if sentence_mean < reference.get("sentence_length_mean", 0.0):
            reasons.append("平均句长偏短")
        if trigram_rep < reference.get("trigram_repetition", 0.0):
            reasons.append("三元词组重复率偏低")
        if punctuation < reference.get("punctuation_ratio", 0.0):
            reasons.append("标点密度偏低")
        if sentence_std < reference.get("sentence_length_std", 0.0):
            reasons.append("句长波动偏小")
        if word_count < reference.get("computed_word_count", 0.0):
            reasons.append("摘要长度偏短")
        if connector_ratio > reference.get("connector_ratio", 0.0):
            reasons.append("连接词比例偏高")
    else:
        if token_entropy <= reference.get("token_entropy", 0.0):
            reasons.append("词汇分布熵不高")
        if hapax_ratio <= reference.get("hapax_ratio", 0.0):
            reasons.append("一次性词比例接近真实摘要")
        if sentence_mean >= reference.get("sentence_length_mean", 0.0):
            reasons.append("平均句长较长")
        if trigram_rep >= reference.get("trigram_repetition", 0.0) or bigram_rep >= reference.get("bigram_repetition", 0.0):
            reasons.append("n-gram重复模式更接近真实摘要")
        if punctuation >= reference.get("punctuation_ratio", 0.0):
            reasons.append("标点密度较高")
        if sentence_std >= reference.get("sentence_length_std", 0.0):
            reasons.append("句长波动较大")

    return "；".join(reasons[:5]) if reasons else "需人工细读；该样本未呈现明显单一风格原因"


def make_case(
    row: dict[str, str],
    *,
    case_type: str,
    case_type_cn: str,
    note: str,
    score_col: str,
    high_threshold: float,
    very_high_threshold: float,
    reference: dict[str, float],
    ai_like_reason: bool,
) -> dict[str, str]:
    score = score_of(row, score_col)
    output = {
        "case_type": case_type,
        "case_type_cn": case_type_cn,
        "score_column": score_col,
        "ai_like_score": f"{score:.6f}",
        "score_band": score_band(score, high_threshold, very_high_threshold),
        "inspection_note_cn": note,
        "style_reason_cn": style_reason(row, reference, ai_like_reason),
        "abstract_excerpt": short_text(row.get("abstract", "")),
    }
    for field in CASE_FIELDS:
        if field not in output:
            output[field] = row.get(field, "")
    return output


def summarize_group(
    rows: list[dict[str, str]],
    *,
    group: str,
    score_col: str,
    high_threshold: float,
    low_threshold: float,
) -> dict[str, str]:
    values = [score_of(row, score_col) for row in group_rows(rows, group)]
    if not values:
        return {
            "score_column": score_col,
            "group": group,
            "group_cn": GROUP_CN.get(group, group),
            "n": "0",
            "mean_score": "",
            "median_score": "",
            "min_score": "",
            "max_score": "",
            "share_ge_high_threshold": "",
            "share_lt_low_threshold": "",
        }
    return {
        "score_column": score_col,
        "group": group,
        "group_cn": GROUP_CN.get(group, group),
        "n": str(len(values)),
        "mean_score": f"{statistics.mean(values):.6f}",
        "median_score": f"{statistics.median(values):.6f}",
        "min_score": f"{min(values):.6f}",
        "max_score": f"{max(values):.6f}",
        "share_ge_high_threshold": f"{sum(value >= high_threshold for value in values) / len(values):.6f}",
        "share_lt_low_threshold": f"{sum(value < low_threshold for value in values) / len(values):.6f}",
    }


def sorted_by_score(rows: list[dict[str, str]], score_col: str, reverse: bool = True) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: score_of(row, score_col), reverse=reverse)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="local_detector_results/predictions.csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--score-column", default="combined_logreg_ai_like_score")
    parser.add_argument("--high-threshold", type=float, default=0.50)
    parser.add_argument("--very-high-threshold", type=float, default=0.80)
    parser.add_argument("--low-threshold", type=float, default=0.50)
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()

    rows = read_csv(Path(args.input))
    if not rows:
        raise SystemExit("Input CSV has no rows.")
    if args.score_column not in rows[0]:
        raise SystemExit(f"Score column not found: {args.score_column}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = build_reference(rows)

    pre_fp_raw = [
        row
        for row in group_rows(rows, "published_pre_chatgpt")
        if score_of(row, args.score_column) >= args.high_threshold
    ]
    post_fp_raw = [
        row
        for row in group_rows(rows, "published_post_chatgpt")
        if score_of(row, args.score_column) >= args.high_threshold
    ]
    top_published_raw = sorted_by_score(group_rows(rows, "published_all"), args.score_column)[: args.top_n]
    ai_polish_low_raw = [
        row
        for row in sorted_by_score(group_rows(rows, "ai_polish_original_abstract"), args.score_column, reverse=False)
        if score_of(row, args.score_column) < args.low_threshold
    ]
    ai_polish_high_raw = [
        row
        for row in sorted_by_score(group_rows(rows, "ai_polish_original_abstract"), args.score_column)
        if score_of(row, args.score_column) >= args.high_threshold
    ]
    ai_generate_low_raw = [
        row
        for row in sorted_by_score(group_rows(rows, "ai_generate_from_metadata"), args.score_column, reverse=False)
        if score_of(row, args.score_column) < args.low_threshold
    ]

    outputs = {
        "false_positive_pre_chatgpt.csv": [
            make_case(
                row,
                case_type="false_positive_pre_chatgpt",
                case_type_cn="ChatGPT普及前真实摘要被判为AI-like",
                note="历史阴性对照中出现高AI-like分数，可作为假阳性案例审读。",
                score_col=args.score_column,
                high_threshold=args.high_threshold,
                very_high_threshold=args.very_high_threshold,
                reference=reference,
                ai_like_reason=True,
            )
            for row in sorted_by_score(pre_fp_raw, args.score_column)
        ],
        "false_positive_post_chatgpt.csv": [
            make_case(
                row,
                case_type="false_positive_post_chatgpt",
                case_type_cn="ChatGPT普及后真实发表摘要被判为AI-like",
                note="AI辅助情况未知；高分可能来自真实AI参与、检测规训后的写作风格，或模型假阳性。",
                score_col=args.score_column,
                high_threshold=args.high_threshold,
                very_high_threshold=args.very_high_threshold,
                reference=reference,
                ai_like_reason=True,
            )
            for row in sorted_by_score(post_fp_raw, args.score_column)
        ],
        "top_ai_like_published_cases.csv": [
            make_case(
                row,
                case_type="top_ai_like_published",
                case_type_cn="真实发表摘要AI-like分数最高样本",
                note="真实发表摘要中的最高分样本，适合逐篇人工审查。",
                score_col=args.score_column,
                high_threshold=args.high_threshold,
                very_high_threshold=args.very_high_threshold,
                reference=reference,
                ai_like_reason=True,
            )
            for row in top_published_raw
        ],
        "ai_polish_low_score_cases.csv": [
            make_case(
                row,
                case_type="ai_polish_low_score",
                case_type_cn="AI润色摘要低分样本",
                note="AI参与但分数较低，适合分析为什么AI润色更难被检测。",
                score_col=args.score_column,
                high_threshold=args.high_threshold,
                very_high_threshold=args.very_high_threshold,
                reference=reference,
                ai_like_reason=False,
            )
            for row in ai_polish_low_raw
        ],
        "ai_polish_high_score_cases.csv": [
            make_case(
                row,
                case_type="ai_polish_high_score",
                case_type_cn="AI润色摘要高分样本",
                note="AI润色后仍被识别为AI-like，可与低分润色样本对照。",
                score_col=args.score_column,
                high_threshold=args.high_threshold,
                very_high_threshold=args.very_high_threshold,
                reference=reference,
                ai_like_reason=True,
            )
            for row in ai_polish_high_raw
        ],
        "ai_generate_low_score_cases.csv": [
            make_case(
                row,
                case_type="ai_generate_low_score",
                case_type_cn="AI生成摘要低分样本",
                note="完整AI生成但分数低，适合分析检测器漏检边界。",
                score_col=args.score_column,
                high_threshold=args.high_threshold,
                very_high_threshold=args.very_high_threshold,
                reference=reference,
                ai_like_reason=False,
            )
            for row in ai_generate_low_raw
        ],
    }

    for filename, case_rows in outputs.items():
        write_csv(out_dir / filename, case_rows, CASE_FIELDS)

    all_cases: list[dict[str, str]] = []
    for case_rows in outputs.values():
        all_cases.extend(case_rows)
    write_csv(out_dir / "all_case_flags.csv", all_cases, CASE_FIELDS)

    summary_rows = [
        summarize_group(
            rows,
            group=group,
            score_col=args.score_column,
            high_threshold=args.high_threshold,
            low_threshold=args.low_threshold,
        )
        for group in [
            "published_pre_chatgpt",
            "published_post_chatgpt",
            "published_all",
            "ai_generate_from_metadata",
            "ai_polish_original_abstract",
        ]
    ]
    write_csv(out_dir / "case_summary.csv", summary_rows, SUMMARY_FIELDS)

    print(f"Rows read: {len(rows)}")
    print(f"Score column: {args.score_column}")
    print(f"High threshold: {args.high_threshold}")
    print(f"Low threshold: {args.low_threshold}")
    for filename, case_rows in outputs.items():
        print(f"{filename}: {len(case_rows)}")
    print(f"all_case_flags.csv: {len(all_cases)}")
    print(f"Wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
