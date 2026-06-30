"""
Summarize manually collected AIGC detector rates for prompt sensitivity variants.

Fill `aigc_rate` in prompt_sensitivity_detection_sheet.csv, then run this script
to create a Markdown summary table and a short manuscript-ready paragraph.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_rate(value: str) -> float | None:
    value = value.strip().replace("%", "")
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if 0 <= number <= 1:
        number *= 100
    return number


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="prompt_sensitivity_experiment/prompt_sensitivity_detection_sheet.csv")
    parser.add_argument("--out", default="prompt_sensitivity_experiment/prompt_sensitivity_summary.md")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_path = Path(args.out)

    with input_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    parsed_rows: list[dict[str, str]] = []
    rates: list[float] = []
    for row in rows:
        rate = parse_rate(row.get("aigc_rate", ""))
        row["aigc_rate_normalized"] = "" if rate is None else f"{rate:.2f}"
        parsed_rows.append(row)
        if rate is not None:
            rates.append(rate)

    lines: list[str] = [
        "# 提示词敏感性补充实验汇总",
        "",
        "| 版本 | 模型 | AIGC率 | 说明 |",
        "|---|---|---:|---|",
    ]
    for row in parsed_rows:
        rate_text = row["aigc_rate_normalized"] + "%" if row["aigc_rate_normalized"] else "未填写"
        lines.append(
            f"| {row.get('variant_name', row.get('variant_id', ''))} "
            f"| {row.get('model', '')} "
            f"| {rate_text} "
            f"| {row.get('description', '')} |"
        )

    lines.append("")
    if len(rates) >= 2:
        spread = max(rates) - min(rates)
        min_row = min(
            (row for row in parsed_rows if row.get("aigc_rate_normalized")),
            key=lambda item: float(item["aigc_rate_normalized"]),
        )
        max_row = max(
            (row for row in parsed_rows if row.get("aigc_rate_normalized")),
            key=lambda item: float(item["aigc_rate_normalized"]),
        )
        lines.extend(
            [
                "## 可写入论文的结果表述",
                "",
                (
                    f"在提示词敏感性补充实验中，同一研究内容在不同提示词风格约束下获得的 AIGC 率出现差异。"
                    f"最高分版本为“{max_row.get('variant_name', '')}”"
                    f"（{float(max_row['aigc_rate_normalized']):.2f}%），"
                    f"最低分版本为“{min_row.get('variant_name', '')}”"
                    f"（{float(min_row['aigc_rate_normalized']):.2f}%），"
                    f"二者相差 {spread:.2f} 个百分点。"
                    "这一结果提示，AIGC 检测分数可能受到提示词诱导的写作风格影响，"
                    "因此不宜被直接解释为文本真实 AI 参与比例。"
                ),
            ]
        )
    else:
        lines.extend(
            [
                "## 待补充",
                "",
                "请先在 CSV 的 `aigc_rate` 列填写各版本的检测结果，再重新运行本脚本。",
            ]
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
