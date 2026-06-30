"""
Small prompt-sensitivity experiment for AIGC detector scores.

The script creates controlled manuscript variants from the same source text:

- original_text: unchanged source manuscript
- standard_academic: conventional polished academic rewrite
- over_template: deliberately formulaic academic rewrite
- anti_template: less template-like rewrite with varied sentence patterns

Use --dry-run to only create prompt files and a blank detection sheet. Use a
SiliconFlow/OpenAI-compatible API key in SILICONFLOW_API_KEY to generate the
rewritten variants.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import date
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree


DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"


@dataclass(frozen=True)
class VariantProfile:
    variant_id: str
    variant_name: str
    description: str
    instruction: str


VARIANTS: list[VariantProfile] = [
    VariantProfile(
        variant_id="01_standard_academic",
        variant_name="标准学术提示词版本",
        description="常规学术润色，追求清晰、规范、完整。",
        instruction=(
            "请将下列论文稿件改写为规范、流畅、正式的中文学术论文表达。"
            "保持原有研究问题、数据、结论、图表编号、引用编号和作者声明不变。"
            "不要新增实验结果，不要删除关键数字，不要改变 AIGC 检测率数值。"
            "输出完整稿件正文，不要输出解释。"
        ),
    ),
    VariantProfile(
        variant_id="02_over_template",
        variant_name="强模板化学术提示词版本",
        description="故意强化模板化、平衡句式和标准论文腔。",
        instruction=(
            "请将下列论文稿件改写为高度标准化、模板化、期刊论文式的中文学术表达。"
            "每一节尽量使用清晰的总分结构、谨慎限定语、规范连接词和对称句式。"
            "可以使用“本文旨在”“结果表明”“进一步提示”“需要指出的是”等常见学术表达。"
            "必须保持原有研究问题、数据、结论、图表编号、引用编号和作者声明不变。"
            "不要新增实验结果，不要删除关键数字，不要改变 AIGC 检测率数值。"
            "输出完整稿件正文，不要输出解释。"
        ),
    ),
    VariantProfile(
        variant_id="03_anti_template",
        variant_name="反模板化提示词版本",
        description="控制语义不变，但降低模板化、过度平衡和机械连接。",
        instruction=(
            "请在不改变论文事实、数据和结论的前提下，改写下列中文论文稿件，目标是减少模板化 AI 学术腔。"
            "具体要求：避免过多使用“本文旨在”“结果表明”“进一步提示”等固定套话；"
            "避免每段都呈现机械的总分总结构；保留适度自然的句长变化；"
            "减少过度对称、过度平衡、过度谨慎的句式；"
            "能用具体说法时不要使用泛泛抽象词；"
            "不要改写成口语文章，仍需保持学术论文体裁。"
            "必须保持原有研究问题、数据、结论、图表编号、引用编号和作者声明不变。"
            "不要新增实验结果，不要删除关键数字，不要改变 AIGC 检测率数值。"
            "输出完整稿件正文，不要输出解释。"
        ),
    ),
]


def read_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml_data = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml_data)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        parts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return read_docx_text(path)
    return path.read_text(encoding="utf-8-sig")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_prompt(source_text: str, profile: VariantProfile) -> str:
    return (
        f"{profile.instruction}\n\n"
        "【待改写稿件开始】\n"
        f"{source_text}\n"
        "【待改写稿件结束】"
    )


def chat_completion(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一名谨慎的中文学术论文编辑。你必须保持事实、数字、引用编号和图表编号不变，"
                    "只按照用户指定的风格约束改写文本。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "ai-like-detector-prompt-sensitivity/0.1",
    }

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(endpoint, data=request_body, headers=headers, method="POST")
            with urlopen(request, timeout=timeout) as response:
                data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"].strip()
            if not text:
                raise RuntimeError("Empty model response.")
            return text
        except HTTPError as exc:
            last_error = exc
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code in {408, 409, 429, 500, 502, 503, 504} and attempt < retries - 1:
                wait_seconds = 6 * (attempt + 1)
                print(f"HTTP {exc.code}; retry in {wait_seconds}s. {error_body[:180]}", file=sys.stderr)
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc
        except (URLError, TimeoutError, RemoteDisconnected, ConnectionError, OSError) as exc:
            last_error = exc
            if attempt < retries - 1:
                wait_seconds = 6 * (attempt + 1)
                print(f"Network error; retry in {wait_seconds}s. {exc}", file=sys.stderr)
                time.sleep(wait_seconds)
                continue
            break

    raise RuntimeError("Request failed after retries.") from last_error


def write_detection_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "variant_id",
        "variant_name",
        "description",
        "model",
        "file_path",
        "detector_platform",
        "detector_version",
        "detection_date",
        "aigc_rate",
        "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_protocol(path: Path, *, input_path: Path, model: str, variants: list[VariantProfile]) -> None:
    variant_lines = "\n".join(f"- `{item.variant_id}`：{item.description}" for item in variants)
    text = f"""# 提示词敏感性补充实验记录

## 实验目的

本补充实验用于检验：在同一研究内容、同一模型条件下，仅改变提示词风格约束，是否会导致 AIGC 检测平台给出不同的 AIGC 率。

## 控制变量

- 原始稿件：`{input_path}`
- 生成模型：`{model}`
- 主要内容、数据、图表编号、引用编号：保持不变
- 变化因素：提示词对写作风格的约束

## 版本

- `original_text`：未改写原稿
{variant_lines}

## 操作步骤

1. 运行 `prompt_sensitivity_experiment.py` 生成不同版本。
2. 将 `variants` 文件夹内每个 `.md` 文本分别提交到同一个 AIGC 检测平台。
3. 将平台名称、检测日期和 AIGC 率填入 `prompt_sensitivity_detection_sheet.csv`。
4. 运行 `summarize_prompt_sensitivity_results.py` 生成汇总表和可写入论文的结论段。

## 可写入论文的方法描述

为检验 AIGC 检测分数对提示词风格控制的敏感性，本文进一步设置小规模补充实验。在同一研究内容和同一大语言模型条件下，分别使用标准学术提示词、强模板化学术提示词和反模板化提示词生成稿件变体，并将各版本提交至同一 AIGC 检测平台。该实验不用于评估文本真实性，而用于观察检测分数是否会随提示词诱导的写作风格变化而波动。
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source manuscript, .md/.txt/.docx")
    parser.add_argument("--out-dir", default="prompt_sensitivity_experiment")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    parser.add_argument("--base-url", default=os.getenv("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-input-chars", type=int, default=22000)
    parser.add_argument("--dry-run", action="store_true", help="Only write prompts/protocol/sheet; do not call API.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    prompt_dir = out_dir / "prompts"
    variant_dir = out_dir / "variants"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    variant_dir.mkdir(parents=True, exist_ok=True)

    source_text = normalize_text(read_text(input_path))
    if len(source_text) > args.max_input_chars:
        print(
            f"Input has {len(source_text)} chars; truncating to {args.max_input_chars}. "
            "Increase --max-input-chars if your model context allows it.",
            file=sys.stderr,
        )
        source_text = source_text[: args.max_input_chars]

    original_path = variant_dir / "00_original_text.md"
    if args.overwrite or not original_path.exists():
        original_path.write_text(source_text + "\n", encoding="utf-8")

    rows: list[dict[str, str]] = [
        {
            "variant_id": "00_original_text",
            "variant_name": "原始稿件",
            "description": "未改写原稿，用作基线。",
            "model": "none",
            "file_path": str(original_path),
            "detector_platform": "",
            "detector_version": "",
            "detection_date": date.today().isoformat(),
            "aigc_rate": "",
            "notes": "",
        }
    ]

    api_key = os.getenv(args.api_key_env, "").strip()
    if not args.dry_run and not api_key:
        raise SystemExit(
            f"Missing API key. Set it first in PowerShell:\n"
            f'$env:{args.api_key_env}="YOUR_KEY"'
        )

    for profile in VARIANTS:
        prompt = build_prompt(source_text, profile)
        prompt_path = prompt_dir / f"{profile.variant_id}.txt"
        output_path = variant_dir / f"{profile.variant_id}.md"
        prompt_path.write_text(prompt, encoding="utf-8")

        if args.dry_run:
            status_note = "dry_run: prompt generated, variant not generated yet"
        elif output_path.exists() and not args.overwrite:
            status_note = "existing variant kept"
        else:
            print(f"Generating {profile.variant_id} with {args.model}...")
            generated = chat_completion(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
            )
            output_path.write_text(generated + "\n", encoding="utf-8")
            status_note = "generated"

        rows.append(
            {
                "variant_id": profile.variant_id,
                "variant_name": profile.variant_name,
                "description": profile.description,
                "model": args.model,
                "file_path": str(output_path),
                "detector_platform": "",
                "detector_version": "",
                "detection_date": date.today().isoformat(),
                "aigc_rate": "",
                "notes": status_note,
            }
        )

    write_detection_sheet(out_dir / "prompt_sensitivity_detection_sheet.csv", rows)
    write_protocol(out_dir / "prompt_sensitivity_protocol.md", input_path=input_path, model=args.model, variants=VARIANTS)
    print(f"Wrote experiment files to {out_dir}")
    if args.dry_run:
        print("Dry run complete. Re-run without --dry-run to call the model API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
