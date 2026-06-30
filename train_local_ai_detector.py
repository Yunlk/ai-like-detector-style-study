"""
Train a transparent local AI-like detector for academic abstracts.

Recommended input:
    all_abstract_variants_style_features.csv

Training design:
    - Train on published_abstract vs ai_generate_from_metadata only.
    - Keep ai_polish_original_abstract as an out-of-training scoring group.
    - Split by source_id so variants from the same source paper cannot leak
      across train/test.
    - Audit similarity between published abstracts and AI-control variants.
    - Report pre/post ChatGPT-period scores separately for published abstracts.

Models:
    1. text_tfidf_logreg: TF-IDF word/character features + logistic regression
    2. style_logreg: hand-engineered style features + logistic regression
    3. combined_logreg: TF-IDF + style features + logistic regression

Outputs:
    - predictions CSV with ai_like_score columns
    - metrics JSON
    - top feature CSVs
    - saved sklearn pipelines
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
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


TRAIN_VARIANTS = {"published_abstract": 0, "ai_generate_from_metadata": 1}
SCORE_VARIANTS = {
    "published_abstract",
    "ai_generate_from_metadata",
    "ai_polish_original_abstract",
}


def require_sklearn() -> None:
    try:
        import joblib  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: scikit-learn/joblib.\n"
            "Install once in PowerShell with:\n"
            "  python -m pip install scikit-learn joblib\n"
            "Then rerun this script."
        ) from exc


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def style_matrix(rows: list[dict[str, str]]) -> list[list[float]]:
    return [[to_float(row.get(name, "")) for name in STYLE_FEATURES] for row in rows]


def abstract_list(rows: list[dict[str, str]]) -> list[str]:
    return [row.get("abstract", "") for row in rows]


def group_values(rows: list[dict[str, str]], group_col: str) -> list[str]:
    groups: list[str] = []
    missing = 0
    for index, row in enumerate(rows):
        value = row.get(group_col, "").strip()
        if not value:
            missing += 1
            value = f"__missing_{group_col}_{index}"
        groups.append(value)
    if missing:
        raise SystemExit(
            f"{missing} training rows have empty {group_col!r}. "
            "Use a dataset with stable source IDs before grouped evaluation."
        )
    return groups


def grouped_train_test_split(
    rows: list[dict[str, str]],
    y: list[int],
    group_col: str,
    test_size: float,
    seed: int,
) -> tuple[list[int], list[int], dict[str, Any]]:
    from sklearn.model_selection import GroupShuffleSplit

    indices = list(range(len(rows)))
    groups = group_values(rows, group_col)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx_array, test_idx_array = next(splitter.split(indices, y, groups))
    train_idx = list(train_idx_array)
    test_idx = list(test_idx_array)

    train_groups = {groups[i] for i in train_idx}
    test_groups = {groups[i] for i in test_idx}
    overlap = sorted(train_groups & test_groups)
    if overlap:
        raise SystemExit(f"Grouped split failed: {len(overlap)} groups appear in both train and test.")

    summary = {
        "split_method": "GroupShuffleSplit",
        "group_col": group_col,
        "n_training_rows": len(rows),
        "n_groups_total": len(set(groups)),
        "n_train_rows": len(train_idx),
        "n_test_rows": len(test_idx),
        "n_train_groups": len(train_groups),
        "n_test_groups": len(test_groups),
        "n_group_overlap": len(overlap),
        "train_label_counts": dict(Counter(y[i] for i in train_idx)),
        "test_label_counts": dict(Counter(y[i] for i in test_idx)),
    }
    return train_idx, test_idx, summary


def build_similarity_audit(
    rows: list[dict[str, str]],
    group_col: str,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], set[str]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    row_id_key = "__row_id"
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group = row.get(group_col, "").strip()
        if group:
            by_group[group].append(row)

    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    for group_rows in by_group.values():
        published = [
            row
            for row in group_rows
            if row.get("text_variant") == "published_abstract" and row.get("abstract", "").strip()
        ]
        variants = [
            row
            for row in group_rows
            if row.get("text_variant") in {"ai_generate_from_metadata", "ai_polish_original_abstract"}
            and row.get("abstract", "").strip()
        ]
        if not published or not variants:
            continue
        source = published[0]
        for variant in variants:
            pairs.append((source, variant))

    if not pairs:
        return [], {}, set()

    texts: list[str] = []
    for source, variant in pairs:
        texts.append(source.get("abstract", ""))
        texts.append(variant.get("abstract", ""))

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)

    audit_rows: list[dict[str, Any]] = []
    similarity_by_row_id: dict[str, dict[str, str]] = {}
    high_similarity_generated_row_ids: set[str] = set()

    for pair_index, (source, variant) in enumerate(pairs):
        source_matrix_index = pair_index * 2
        variant_matrix_index = source_matrix_index + 1
        similarity = float(cosine_similarity(matrix[source_matrix_index], matrix[variant_matrix_index])[0][0])
        high_flag = similarity >= threshold
        variant_id = variant.get(row_id_key, "")
        text_variant = variant.get("text_variant", "")
        if high_flag and text_variant == "ai_generate_from_metadata" and variant_id:
            high_similarity_generated_row_ids.add(variant_id)

        similarity_info = {
            "similarity_to_published_abstract": f"{similarity:.6f}",
            "high_similarity_to_published": "1" if high_flag else "0",
        }
        if variant_id:
            similarity_by_row_id[variant_id] = similarity_info

        audit_rows.append(
            {
                group_col: variant.get(group_col, ""),
                "text_variant": text_variant,
                "generation_model": variant.get("generation_model", ""),
                "year": variant.get("year", ""),
                "period": variant.get("period", ""),
                "title": variant.get("title", ""),
                "published_word_count": len(source.get("abstract", "").split()),
                "variant_word_count": len(variant.get("abstract", "").split()),
                "cosine_similarity_to_published": f"{similarity:.6f}",
                "high_similarity_threshold": f"{threshold:.6f}",
                "high_similarity_flag": "1" if high_flag else "0",
            }
        )

    return audit_rows, similarity_by_row_id, high_similarity_generated_row_ids


def summarize_scores(rows: list[dict[str, Any]], score_key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        variant = str(row.get("text_variant", ""))
        period = str(row.get("period", ""))
        key = variant if variant != "published_abstract" else f"{variant}:{period or 'unknown_period'}"
        score = row.get(score_key)
        if score in ("", None):
            continue
        groups.setdefault(key, []).append(float(score))

    summary: dict[str, dict[str, float]] = {}
    for key, values in sorted(groups.items()):
        values_sorted = sorted(values)
        n = len(values_sorted)
        high_50 = sum(1 for value in values_sorted if value >= 0.50) / n if n else 0.0
        high_80 = sum(1 for value in values_sorted if value >= 0.80) / n if n else 0.0
        summary[key] = {
            "n": n,
            "mean": statistics.mean(values_sorted),
            "median": statistics.median(values_sorted),
            "min": min(values_sorted),
            "max": max(values_sorted),
            "share_score_ge_0_50": high_50,
            "share_score_ge_0_80": high_80,
        }
    return summary


def classification_metrics(y_true: list[int], y_score: list[float], threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    y_pred = [1 if score >= threshold else 0 for score in y_score]
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "average_precision": average_precision_score(y_true, y_score),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def top_linear_features(vectorizer: Any, classifier: Any, top_n: int) -> list[dict[str, Any]]:
    feature_names = list(vectorizer.get_feature_names_out())
    coefs = classifier.coef_[0]
    ranked = sorted(zip(feature_names, coefs), key=lambda item: abs(item[1]), reverse=True)
    return [
        {
            "feature": feature,
            "coefficient": f"{coef:.8f}",
            "direction": "ai_like" if coef > 0 else "published_like",
        }
        for feature, coef in ranked[:top_n]
    ]


def top_combined_features(preprocessor: Any, classifier: Any, top_n: int) -> list[dict[str, Any]]:
    word_names = [
        f"word_tfidf__{name}"
        for name in preprocessor.transformer_list[0][1].named_steps["tfidf"].get_feature_names_out()
    ]
    char_names = [
        f"char_tfidf__{name}"
        for name in preprocessor.transformer_list[1][1].named_steps["tfidf"].get_feature_names_out()
    ]
    style_names = [f"style__{name}" for name in STYLE_FEATURES]
    feature_names = word_names + char_names + style_names
    coefs = classifier.coef_[0]
    ranked = sorted(zip(feature_names, coefs), key=lambda item: abs(item[1]), reverse=True)
    return [
        {
            "feature": feature,
            "coefficient": f"{coef:.8f}",
            "direction": "ai_like" if coef > 0 else "published_like",
        }
        for feature, coef in ranked[:top_n]
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="all_abstract_variants_style_features.csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--top-n", type=int, default=60)
    parser.add_argument("--group-col", default="source_id", help="Column used to keep source variants together")
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    parser.add_argument(
        "--exclude-high-similarity-generated",
        action="store_true",
        help="Exclude ai_generate_from_metadata rows that are too similar to the published abstract",
    )
    args = parser.parse_args()

    require_sklearn()

    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(input_path)
    if not rows:
        raise SystemExit("Input CSV has no rows.")
    for index, row in enumerate(rows):
        row["__row_id"] = str(index)

    score_rows = [row for row in rows if row.get("text_variant") in SCORE_VARIANTS and row.get("abstract", "").strip()]
    similarity_audit_rows, similarity_by_row_id, high_similarity_generated_row_ids = build_similarity_audit(
        score_rows,
        args.group_col,
        args.similarity_threshold,
    )

    train_rows = []
    for row in rows:
        if row.get("text_variant") not in TRAIN_VARIANTS or not row.get("abstract", "").strip():
            continue
        if args.exclude_high_similarity_generated and row.get("__row_id") in high_similarity_generated_row_ids:
            continue
        train_rows.append(row)

    y = [TRAIN_VARIANTS[row["text_variant"]] for row in train_rows]
    if len(set(y)) != 2:
        raise SystemExit("Training data must contain both published_abstract and ai_generate_from_metadata.")

    train_idx, test_idx, split_summary = grouped_train_test_split(
        train_rows,
        y,
        args.group_col,
        args.test_size,
        args.seed,
    )
    train_subset = [train_rows[i] for i in train_idx]
    test_subset = [train_rows[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    text_model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    analyzer="word",
                    min_df=2,
                    max_features=8000,
                ),
            ),
            ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)),
        ]
    )

    style_model = Pipeline(
        [
            ("select_style", FunctionTransformer(style_matrix, validate=False)),
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)),
        ]
    )

    combined_preprocessor = FeatureUnion(
        [
            (
                "word_tfidf",
                Pipeline(
                    [
                        ("select_text", FunctionTransformer(abstract_list, validate=False)),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                lowercase=True,
                                strip_accents="unicode",
                                ngram_range=(1, 2),
                                min_df=2,
                                max_features=7000,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "char_tfidf",
                Pipeline(
                    [
                        ("select_text", FunctionTransformer(abstract_list, validate=False)),
                        (
                            "tfidf",
                            TfidfVectorizer(
                                lowercase=True,
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                max_features=5000,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "style",
                Pipeline(
                    [
                        ("select_style", FunctionTransformer(style_matrix, validate=False)),
                        ("scale", StandardScaler()),
                    ]
                ),
            ),
        ]
    )
    combined_model = Pipeline(
        [
            ("features", combined_preprocessor),
            ("logreg", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=args.seed)),
        ]
    )

    models = {
        "text_tfidf_logreg": text_model,
        "style_logreg": style_model,
        "combined_logreg": combined_model,
    }

    metrics: dict[str, Any] = {
        "input": str(input_path),
        "n_total_rows": len(rows),
        "n_training_rows": len(train_rows),
        "n_scored_rows": len(score_rows),
        "train_variants": TRAIN_VARIANTS,
        "test_size": args.test_size,
        "seed": args.seed,
        "split": split_summary,
        "similarity_audit": {
            "similarity_metric": "char_wb_tfidf_3_5_cosine",
            "similarity_threshold": args.similarity_threshold,
            "n_pairs_audited": len(similarity_audit_rows),
            "n_high_similarity_pairs": sum(
                1 for row in similarity_audit_rows if row.get("high_similarity_flag") == "1"
            ),
            "n_high_similarity_ai_generated_pairs": len(high_similarity_generated_row_ids),
            "exclude_high_similarity_generated": args.exclude_high_similarity_generated,
            "n_excluded_ai_generated_training_rows": len(high_similarity_generated_row_ids)
            if args.exclude_high_similarity_generated
            else 0,
        },
        "style_features": STYLE_FEATURES,
        "models": {},
    }

    prediction_rows = [dict(row) for row in score_rows]
    for row in prediction_rows:
        similarity_info = similarity_by_row_id.get(row.get("__row_id", ""))
        row["similarity_to_published_abstract"] = (
            similarity_info.get("similarity_to_published_abstract", "") if similarity_info else ""
        )
        row["high_similarity_to_published"] = (
            similarity_info.get("high_similarity_to_published", "") if similarity_info else ""
        )

    for name, model in models.items():
        if name == "combined_logreg":
            X_train = train_subset
            X_test = test_subset
            X_score = prediction_rows
        elif name == "style_logreg":
            X_train = train_subset
            X_test = test_subset
            X_score = prediction_rows
        else:
            X_train = [row["abstract"] for row in train_subset]
            X_test = [row["abstract"] for row in test_subset]
            X_score = [row["abstract"] for row in prediction_rows]

        model.fit(X_train, y_train)
        test_scores = list(model.predict_proba(X_test)[:, 1])
        score_values = list(model.predict_proba(X_score)[:, 1])

        score_key = f"{name}_ai_like_score"
        for row, score in zip(prediction_rows, score_values):
            row[score_key] = f"{score:.6f}"

        metrics["models"][name] = {
            "holdout": classification_metrics(y_test, test_scores),
            "score_summary": summarize_scores(prediction_rows, score_key),
        }

        joblib.dump(model, out_dir / f"{name}.joblib")

        if name == "text_tfidf_logreg":
            top_rows = top_linear_features(
                model.named_steps["tfidf"],
                model.named_steps["logreg"],
                args.top_n,
            )
            write_csv(out_dir / f"{name}_top_features.csv", top_rows, ["feature", "coefficient", "direction"])
        elif name == "style_logreg":
            coefs = model.named_steps["logreg"].coef_[0]
            top_rows = sorted(
                [
                    {
                        "feature": feature,
                        "coefficient": f"{coef:.8f}",
                        "direction": "ai_like" if coef > 0 else "published_like",
                    }
                    for feature, coef in zip(STYLE_FEATURES, coefs)
                ],
                key=lambda row: abs(float(row["coefficient"])),
                reverse=True,
            )
            write_csv(out_dir / f"{name}_top_features.csv", top_rows, ["feature", "coefficient", "direction"])
        elif name == "combined_logreg":
            top_rows = top_combined_features(
                model.named_steps["features"],
                model.named_steps["logreg"],
                args.top_n,
            )
            write_csv(out_dir / f"{name}_top_features.csv", top_rows, ["feature", "coefficient", "direction"])

    similarity_fields = ["similarity_to_published_abstract", "high_similarity_to_published"]
    prediction_fields = [field for field in score_rows[0].keys() if field != "__row_id"] if score_rows else []
    for field in similarity_fields:
        if field not in prediction_fields:
            prediction_fields.append(field)
    for name in models:
        field = f"{name}_ai_like_score"
        if field not in prediction_fields:
            prediction_fields.append(field)
    write_csv(out_dir / "predictions.csv", prediction_rows, prediction_fields)

    similarity_fields_out = [
        args.group_col,
        "text_variant",
        "generation_model",
        "year",
        "period",
        "title",
        "published_word_count",
        "variant_word_count",
        "cosine_similarity_to_published",
        "high_similarity_threshold",
        "high_similarity_flag",
    ]
    write_csv(out_dir / "similarity_audit.csv", similarity_audit_rows, similarity_fields_out)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(f"Training rows: {len(train_rows)}")
    print(f"Scored rows: {len(prediction_rows)}")
    print(
        "Grouped split: "
        f"train_groups={split_summary['n_train_groups']}, "
        f"test_groups={split_summary['n_test_groups']}, "
        f"overlap={split_summary['n_group_overlap']}"
    )
    print(
        "Similarity audit: "
        f"pairs={len(similarity_audit_rows)}, "
        f"high_ai_generated={len(high_similarity_generated_row_ids)}"
    )
    for name, model_metrics in metrics["models"].items():
        holdout = model_metrics["holdout"]
        print(
            f"{name}: "
            f"accuracy={holdout['accuracy']:.3f}, "
            f"f1={holdout['f1']:.3f}, "
            f"roc_auc={holdout['roc_auc']:.3f}"
        )
    print(f"Wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
