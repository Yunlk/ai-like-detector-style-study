"""
Score prompt-style sensitivity abstract generations with local detectors.

Inputs:
- real abstracts CSV used to create prompts
- generated CSV produced by call_siliconflow_controls_multimodel.py
- saved detector model directory

Outputs:
- row-level scores
- style/model summary Markdown and CSV
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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

TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
ACADEMIC_CONNECTORS = {
    "therefore",
    "however",
    "moreover",
    "furthermore",
    "nevertheless",
    "consequently",
    "thus",
    "hence",
    "whereas",
    "although",
    "because",
    "since",
    "while",
    "overall",
    "finally",
}
TEMPLATE_PHRASES = [
    "the purpose of this study",
    "this study aims",
    "this paper presents",
    "the results show",
    "the results showed",
    "in this paper",
    "in this study",
    "we propose",
    "we investigated",
    "we examined",
    "it is concluded",
    "this article examines",
    "the results suggest",
    "these findings indicate",
]


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def style_matrix(rows: list[dict[str, str]]) -> list[list[float]]:
    return [[to_float(row.get(name, "")) for name in STYLE_FEATURES] for row in rows]


def abstract_list(rows: list[dict[str, str]]) -> list[str]:
    return [row.get("abstract", "") for row in rows]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_RE.split(text.strip()) if part.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def ngram_repetition(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(len(ngrams), 1)


def shannon_entropy(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def features_for_text(text: str) -> dict[str, str]:
    tokens = tokenize(text)
    sentences = split_sentences(text)
    sentence_lengths = [len(tokenize(sentence)) for sentence in sentences]
    lower_text = text.lower()

    mean_len = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
    if len(sentence_lengths) > 1:
        variance = sum((length - mean_len) ** 2 for length in sentence_lengths) / (len(sentence_lengths) - 1)
        std_len = math.sqrt(variance)
    else:
        std_len = 0.0

    connector_count = sum(1 for token in tokens if token in ACADEMIC_CONNECTORS)
    template_count = sum(1 for phrase in TEMPLATE_PHRASES if phrase in lower_text)
    punctuation_count = sum(1 for char in text if char in ",;:()[]")

    return {
        "computed_word_count": str(len(tokens)),
        "sentence_count": str(len(sentences)),
        "sentence_length_mean": f"{mean_len:.4f}",
        "sentence_length_std": f"{std_len:.4f}",
        "lexical_diversity": f"{(len(set(tokens)) / len(tokens)) if tokens else 0.0:.4f}",
        "hapax_ratio": f"{(sum(1 for count in Counter(tokens).values() if count == 1) / len(tokens)) if tokens else 0.0:.4f}",
        "bigram_repetition": f"{ngram_repetition(tokens, 2):.4f}",
        "trigram_repetition": f"{ngram_repetition(tokens, 3):.4f}",
        "connector_ratio": f"{(connector_count / len(tokens)) if tokens else 0.0:.4f}",
        "template_phrase_count": str(template_count),
        "punctuation_ratio": f"{(punctuation_count / len(text)) if text else 0.0:.4f}",
        "token_entropy": f"{shannon_entropy(tokens):.4f}",
    }


def load_models(model_dir: Path) -> dict[str, Any]:
    try:
        import joblib
    except ImportError as exc:
        raise SystemExit("Install dependencies first: python -m pip install scikit-learn joblib") from exc

    names = ["text_tfidf_logreg", "style_logreg", "combined_logreg"]
    models = {}
    for name in names:
        path = model_dir / f"{name}.joblib"
        if not path.exists():
            raise SystemExit(f"Missing model file: {path}")
        models[name] = joblib.load(path)
    return models


def build_rows(real_rows: list[dict[str, str]], generated_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    generated_source_ids = {row.get("source_id", "") for row in generated_rows}
    rows: list[dict[str, str]] = []

    for row in real_rows:
        source_id = row.get("id", row.get("source_id", ""))
        if source_id not in generated_source_ids:
            continue
        abstract = row.get("abstract", "").strip()
        if not abstract:
            continue
        out = {
            "source_id": source_id,
            "year": row.get("year", ""),
            "period": row.get("period", ""),
            "title": row.get("title", ""),
            "source_platform": row.get("source_platform", ""),
            "model": "none",
            "prompt_style": "published_baseline",
            "text_variant": "published_abstract",
            "abstract": abstract,
            "status": "ok",
        }
        out.update(features_for_text(abstract))
        rows.append(out)

    for row in generated_rows:
        if row.get("status") != "ok" or not row.get("generated_text", "").strip():
            continue
        abstract = row.get("generated_text", "").strip()
        out = {
            "source_id": row.get("source_id", ""),
            "year": row.get("year", ""),
            "period": row.get("period", ""),
            "title": row.get("title", ""),
            "source_platform": row.get("source_platform", ""),
            "model": row.get("model", ""),
            "prompt_style": row.get("prompt_style", ""),
            "text_variant": "prompt_sensitivity_generated",
            "abstract": abstract,
            "status": row.get("status", ""),
        }
        out.update(features_for_text(abstract))
        rows.append(out)

    return rows


def score_rows(rows: list[dict[str, str]], models: dict[str, Any]) -> list[dict[str, str]]:
    for name, model in models.items():
        X = [row["abstract"] for row in rows] if name == "text_tfidf_logreg" else rows
        scores = model.predict_proba(X)[:, 1]
        for row, score in zip(rows, scores):
            row[f"{name}_ai_like_score"] = f"{float(score):.6f}"
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("text_variant", ""), row.get("prompt_style", ""), row.get("model", ""))
        groups[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (variant, style, model), group_rows in sorted(groups.items()):
        combined = [float(row["combined_logreg_ai_like_score"]) for row in group_rows]
        text_scores = [float(row["text_tfidf_logreg_ai_like_score"]) for row in group_rows]
        style_scores = [float(row["style_logreg_ai_like_score"]) for row in group_rows]
        summary_rows.append(
            {
                "text_variant": variant,
                "prompt_style": style,
                "model": model,
                "n": len(group_rows),
                "combined_mean": f"{mean(combined):.6f}",
                "combined_ge_0_5": f"{sum(score >= 0.5 for score in combined) / len(combined):.3f}",
                "combined_ge_0_8": f"{sum(score >= 0.8 for score in combined) / len(combined):.3f}",
                "text_tfidf_mean": f"{mean(text_scores):.6f}",
                "style_mean": f"{mean(style_scores):.6f}",
                "word_count_mean": f"{mean([float(row['computed_word_count']) for row in group_rows]):.2f}",
                "template_phrase_mean": f"{mean([float(row['template_phrase_count']) for row in group_rows]):.2f}",
            }
        )
    return summary_rows


def write_markdown(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 摘要提示词敏感性实验汇总",
        "",
        "| 文本类型 | 提示词风格 | 模型 | n | Combined均值 | >=0.5 | >=0.8 | Text均值 | Style均值 | 模板短语均值 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['text_variant']} | {row['prompt_style']} | {row['model']} | {row['n']} "
            f"| {float(row['combined_mean']):.3f} "
            f"| {float(row['combined_ge_0_5']):.1%} "
            f"| {float(row['combined_ge_0_8']):.1%} "
            f"| {float(row['text_tfidf_mean']):.3f} "
            f"| {float(row['style_mean']):.3f} "
            f"| {float(row['template_phrase_mean']):.2f} |"
        )

    generated = [row for row in summary_rows if row["text_variant"] == "prompt_sensitivity_generated"]
    if generated:
        high = max(generated, key=lambda row: float(row["combined_mean"]))
        low = min(generated, key=lambda row: float(row["combined_mean"]))
        spread = float(high["combined_mean"]) - float(low["combined_mean"])
        lines.extend(
            [
                "",
                "## 可写入论文的结果句",
                "",
                (
                    f"在摘要级提示词敏感性实验中，不同提示词风格与模型组合的 combined AI-like 平均分存在差异。"
                    f"最高组合为 `{high['prompt_style']}` / `{high['model']}`（均值 {float(high['combined_mean']):.3f}），"
                    f"最低组合为 `{low['prompt_style']}` / `{low['model']}`（均值 {float(low['combined_mean']):.3f}），"
                    f"二者相差 {spread:.3f}。"
                    "该结果说明，在同一批题名和元数据条件下，提示词风格仍可能影响本地检测器输出的 AI-like 分数。"
                ),
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--model-dir", default="local_detector_results_budget_grouped_v2")
    parser.add_argument("--out-dir", default="prompt_sensitivity_abstracts")
    args = parser.parse_args()

    real_rows = read_csv(Path(args.real))
    generated_rows = read_csv(Path(args.generated))
    rows = build_rows(real_rows, generated_rows)
    if not rows:
        raise SystemExit("No rows to score. Check generated status/generation text.")

    models = load_models(Path(args.model_dir))
    scored_rows = score_rows(rows, models)
    summary_rows = summarize(scored_rows)

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "prompt_sensitivity_abstract_scores.csv", scored_rows)
    write_csv(out_dir / "prompt_sensitivity_abstract_summary.csv", summary_rows)
    write_markdown(out_dir / "prompt_sensitivity_abstract_summary.md", summary_rows)

    print(f"Scored {len(scored_rows)} rows.")
    print(f"Wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
