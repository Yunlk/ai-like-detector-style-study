"""
Plot the abstract-level prompt-sensitivity experiment summary.

Input:
    prompt_sensitivity_abstract_summary.csv

Output:
    A PNG bar chart comparing combined AI-like scores by prompt style and model.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


STYLE_LABELS = {
    "neutral_metadata": "普通元数据",
    "standard_academic": "标准学术",
    "over_template": "强模板化",
    "anti_template": "反模板化",
}

MODEL_LABELS = {
    "deepseek-ai/DeepSeek-V4-Pro": "DeepSeek-V4-Pro",
    "tencent/Hunyuan-MT-7B": "Hunyuan-MT-7B",
}

STYLE_ORDER = ["standard_academic", "anti_template", "neutral_metadata", "over_template"]


def configure_chinese_font() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for font_path in font_candidates:
        path = Path(font_path)
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            font_name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    configure_chinese_font()

    rows = read_summary(Path(args.input))
    generated = [row for row in rows if row.get("text_variant") == "prompt_sensitivity_generated"]
    baseline_rows = [row for row in rows if row.get("text_variant") == "published_abstract"]
    if not generated:
        raise SystemExit("No prompt_sensitivity_generated rows found.")

    models = sorted({row["model"] for row in generated})
    styles = [style for style in STYLE_ORDER if any(row["prompt_style"] == style for row in generated)]
    baseline = float(baseline_rows[0]["combined_mean"]) if baseline_rows else None

    values: dict[tuple[str, str], float] = {}
    ge_08: dict[tuple[str, str], float] = {}
    for row in generated:
        key = (row["prompt_style"], row["model"])
        values[key] = float(row["combined_mean"])
        ge_08[key] = float(row["combined_ge_0_8"])

    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=180)
    x_positions = list(range(len(styles)))
    width = 0.34 if len(models) > 1 else 0.48
    offsets = [(-width / 2), (width / 2)] if len(models) == 2 else [0]
    colors = ["#3b82f6", "#16a34a", "#f97316", "#9333ea"]

    for model_index, model in enumerate(models):
        bar_x = [x + offsets[model_index] for x in x_positions]
        bar_values = [values.get((style, model), 0.0) for style in styles]
        bars = ax.bar(
            bar_x,
            bar_values,
            width=width,
            label=MODEL_LABELS.get(model, model),
            color=colors[model_index % len(colors)],
            alpha=0.88,
        )
        for bar, style in zip(bars, styles):
            value = bar.get_height()
            high_share = ge_08.get((style, model), 0.0)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}\n≥0.8:{high_share:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    if baseline is not None:
        ax.axhline(baseline, color="#374151", linestyle="--", linewidth=1.4)
        ax.text(
            len(styles) - 0.35,
            baseline + 0.012,
            f"真实摘要基线 {baseline:.3f}",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#374151",
        )

    ax.set_title("提示词风格对本地 AI-like 分数的影响（摘要级实验）", fontsize=15, pad=14)
    ax.set_ylabel("Combined AI-like 分数均值")
    ax.set_xlabel("提示词风格")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([STYLE_LABELS.get(style, style) for style in styles])
    ax.set_ylim(0, 1.04)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "注：分数来自本地检测器，表示接近“AI从元数据生成摘要”训练正例的程度；不是实际 AI 参与比例。",
        fontsize=8.5,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
