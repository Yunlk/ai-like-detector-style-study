"""
Extract simple, dependency-free style features from abstract texts.

Usage:
    python work/pilot_tools/extract_style_features.py --input real_published_abstracts_clean.csv --out style_features.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path


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
]


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.out)

    with src.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise SystemExit("Input CSV has no rows.")

    output_rows = []
    feature_names: list[str] | None = None
    for row in rows:
        feats = features_for_text(row.get("abstract", ""))
        feature_names = feature_names or list(feats.keys())
        output_rows.append({**row, **feats})

    fieldnames = list(rows[0].keys()) + [name for name in (feature_names or []) if name not in rows[0]]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
