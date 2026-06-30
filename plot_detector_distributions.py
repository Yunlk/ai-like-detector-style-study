"""
Plot AI-like score distributions from local detector predictions.

Input:
    local_detector_results/predictions.csv

Outputs:
    - PNG figures
    - score_distribution_summary.csv

The script uses matplotlib only, so it should work in a simple local Python
environment once matplotlib is installed.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


DEFAULT_SCORE_COLUMNS = [
    "text_tfidf_logreg_ai_like_score",
    "style_logreg_ai_like_score",
    "combined_logreg_ai_like_score",
]

GROUP_LABELS = {
    "published_abstract": "真实发表摘要",
    "published_pre_chatgpt": "ChatGPT普及前真实摘要",
    "published_post_chatgpt": "ChatGPT普及后真实摘要",
    "ai_generate_from_metadata": "AI根据元数据生成摘要",
    "ai_polish_original_abstract": "AI润色原始摘要",
}

SCORE_LABELS = {
    "text_tfidf_logreg_ai_like_score": "文本TF-IDF模型",
    "style_logreg_ai_like_score": "风格特征模型",
    "combined_logreg_ai_like_score": "文本+风格组合模型",
}

COLORS = {
    "published_abstract": "#4C78A8",
    "published_pre_chatgpt": "#4C78A8",
    "published_post_chatgpt": "#F58518",
    "ai_generate_from_metadata": "#54A24B",
    "ai_polish_original_abstract": "#B279A2",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def setup_chinese_matplotlib() -> None:
    import matplotlib
    from matplotlib import font_manager

    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            matplotlib.rcParams["font.sans-serif"] = [font_name]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


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


def collect_scores(rows: list[dict[str, str]], score_col: str, group: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        variant = row.get("text_variant", "")
        period = row.get("period", "")
        include = False
        if group == "published_abstract":
            include = variant == "published_abstract"
        elif group == "published_pre_chatgpt":
            include = variant == "published_abstract" and period == "pre_chatgpt"
        elif group == "published_post_chatgpt":
            include = variant == "published_abstract" and period == "post_chatgpt"
        else:
            include = variant == group
        if include:
            score = to_float(row.get(score_col, ""))
            if score is not None:
                values.append(score)
    return values


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - idx) + sorted_values[hi] * (idx - lo)


def summary_row(score_col: str, group: str, values: list[float]) -> dict[str, str]:
    if not values:
        return {
            "score_column": score_col,
            "group": group,
            "n": "0",
            "mean": "",
            "median": "",
            "p25": "",
            "p75": "",
            "min": "",
            "max": "",
            "share_ge_0_50": "",
            "share_ge_0_80": "",
        }
    return {
        "score_column": score_col,
        "group": group,
        "n": str(len(values)),
        "mean": f"{statistics.mean(values):.6f}",
        "median": f"{statistics.median(values):.6f}",
        "p25": f"{quantile(values, 0.25):.6f}",
        "p75": f"{quantile(values, 0.75):.6f}",
        "min": f"{min(values):.6f}",
        "max": f"{max(values):.6f}",
        "share_ge_0_50": f"{sum(v >= 0.50 for v in values) / len(values):.6f}",
        "share_ge_0_80": f"{sum(v >= 0.80 for v in values) / len(values):.6f}",
    }


def style_axes(ax, title: str, xlabel: str = "AI-like分数") -> None:
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("数量")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.axvline(0.5, color="#333333", linestyle="--", linewidth=1, alpha=0.75)
    ax.axvline(0.8, color="#333333", linestyle=":", linewidth=1, alpha=0.75)


def plot_hist_overlay(rows: list[dict[str, str]], score_col: str, groups: list[str], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    setup_chinese_matplotlib()
    bins = [i / 20 for i in range(21)]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    for group in groups:
        values = collect_scores(rows, score_col, group)
        if not values:
            continue
        ax.hist(
            values,
            bins=bins,
            alpha=0.48,
            label=f"{GROUP_LABELS.get(group, group)} (n={len(values)})",
            color=COLORS.get(group),
            edgecolor="white",
            linewidth=0.6,
        )
    style_axes(ax, title)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_box(rows: list[dict[str, str]], score_col: str, groups: list[str], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    setup_chinese_matplotlib()
    data = [collect_scores(rows, score_col, group) for group in groups]
    labels = [GROUP_LABELS.get(group, group) for group in groups]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    try:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True)
    for patch, group in zip(bp["boxes"], groups):
        patch.set_facecolor(COLORS.get(group, "#999999"))
        patch.set_alpha(0.65)
    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.5)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_ylabel("AI-like分数")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1, alpha=0.75)
    ax.axhline(0.8, color="#333333", linestyle=":", linewidth=1, alpha=0.75)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(rows: list[dict[str, str]], score_cols: list[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    setup_chinese_matplotlib()
    groups = ["published_abstract", "ai_generate_from_metadata", "ai_polish_original_abstract"]
    fig, axes = plt.subplots(1, len(score_cols), figsize=(5.2 * len(score_cols), 4.8), dpi=160, sharey=True)
    if len(score_cols) == 1:
        axes = [axes]
    bins = [i / 20 for i in range(21)]
    for ax, score_col in zip(axes, score_cols):
        for group in groups:
            values = collect_scores(rows, score_col, group)
            ax.hist(values, bins=bins, alpha=0.45, label=GROUP_LABELS.get(group, group), color=COLORS.get(group))
        title = SCORE_LABELS.get(score_col, score_col.replace("_ai_like_score", "").replace("_", " "))
        style_axes(ax, title)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("不同检测器的AI-like分数分布", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="local_detector_results/predictions.csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--main-score",
        default="combined_logreg_ai_like_score",
        help="Score column used for the main figures.",
    )
    args = parser.parse_args()

    rows = read_csv(Path(args.input))
    if not rows:
        raise SystemExit("Input CSV has no rows.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    available_scores = [col for col in DEFAULT_SCORE_COLUMNS if col in rows[0]]
    if args.main_score not in rows[0]:
        raise SystemExit(f"Score column not found: {args.main_score}")

    summary_rows: list[dict[str, str]] = []
    summary_groups = [
        "published_abstract",
        "published_pre_chatgpt",
        "published_post_chatgpt",
        "ai_generate_from_metadata",
        "ai_polish_original_abstract",
    ]
    for score_col in available_scores:
        for group in summary_groups:
            summary_rows.append(summary_row(score_col, group, collect_scores(rows, score_col, group)))

    write_csv(
        out_dir / "score_distribution_summary.csv",
        summary_rows,
        [
            "score_column",
            "group",
            "n",
            "mean",
            "median",
            "p25",
            "p75",
            "min",
            "max",
            "share_ge_0_50",
            "share_ge_0_80",
        ],
    )

    plot_hist_overlay(
        rows,
        args.main_score,
        ["published_abstract", "ai_generate_from_metadata", "ai_polish_original_abstract"],
        out_dir / "main_distribution_published_vs_ai.png",
        "真实发表摘要与AI对照摘要的分数分布",
    )
    plot_hist_overlay(
        rows,
        args.main_score,
        ["published_pre_chatgpt", "published_post_chatgpt"],
        out_dir / "published_pre_vs_post_chatgpt.png",
        "真实发表摘要：ChatGPT普及前后对比",
    )
    plot_hist_overlay(
        rows,
        args.main_score,
        ["ai_generate_from_metadata", "ai_polish_original_abstract"],
        out_dir / "ai_generated_vs_ai_polished.png",
        "AI生成摘要与AI润色摘要对比",
    )
    plot_box(
        rows,
        args.main_score,
        [
            "published_pre_chatgpt",
            "published_post_chatgpt",
            "ai_generate_from_metadata",
            "ai_polish_original_abstract",
        ],
        out_dir / "main_score_boxplot.png",
        "不同文本组的AI-like分数箱线图",
    )
    if available_scores:
        plot_model_comparison(rows, available_scores, out_dir / "model_comparison_distributions.png")

    print(f"Rows: {len(rows)}")
    print(f"Score columns: {available_scores}")
    print(f"Main score: {args.main_score}")
    print(f"Wrote figures and summary to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
