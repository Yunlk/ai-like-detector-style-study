"""
Generate AI-control abstracts with SiliconFlow's OpenAI-compatible API.

The script reads the prompt CSV created by make_ai_prompts.py and writes a new
CSV with generated_text filled in. It is resumable: existing successful rows in
the output file are skipped unless --overwrite is passed.

Environment variables:
    SILICONFLOW_API_KEY   Required. Your SiliconFlow API key.
    SILICONFLOW_BASE_URL  Optional. Defaults to https://api.siliconflow.cn/v1
    SILICONFLOW_MODEL     Optional. Defaults to dsv4pro

Examples:
    setx SILICONFLOW_API_KEY "YOUR_SILICONFLOW_KEY"  # persistent PowerShell/CMD setting

    python work/pilot_tools/call_siliconflow_controls.py ^
      --input outputs/real_published_abstracts_ai_control_prompts.csv ^
      --out outputs/real_published_abstracts_ai_controls_generated.csv ^
      --model dsv4pro ^
      --limit 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "dsv4pro"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_existing(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}

    rows = read_csv(path)
    existing: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("source_id", ""), row.get("prompt_type", ""))
        if row.get("status") == "ok" and row.get("generated_text", "").strip():
            existing[key] = row
    return existing


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
    retry_sleep: float,
) -> tuple[str, dict[str, Any]]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write concise, publication-style academic abstracts. "
                    "Follow the user's constraints exactly and output only the requested abstract."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "ai-detector-style-study/0.1",
    }

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(endpoint, data=body, headers=headers, method="POST")
            with urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"].strip()
            return text, data

        except HTTPError as exc:
            last_error = exc
            retry_after = exc.headers.get("Retry-After")
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code in {408, 409, 429, 500, 502, 503, 504} and attempt < retries - 1:
                wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else retry_sleep * (attempt + 1)
                print(f"HTTP {exc.code}; wait {wait_seconds:.0f}s then retry. {error_body[:200]}", file=sys.stderr)
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc

        except URLError as exc:
            last_error = exc
            if attempt < retries - 1:
                wait_seconds = retry_sleep * (attempt + 1)
                print(f"Network error; wait {wait_seconds:.0f}s then retry. {exc}", file=sys.stderr)
                time.sleep(wait_seconds)
                continue
            break

    raise RuntimeError("Request failed after retries.") from last_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Prompt CSV from make_ai_prompts.py")
    parser.add_argument("--out", required=True, help="Output CSV with generated_text")
    parser.add_argument("--model", default=os.getenv("SILICONFLOW_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=360)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=8.0)
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds between successful calls")
    parser.add_argument("--limit", type=int, default=0, help="Only process N pending rows; 0 means all")
    parser.add_argument("--prompt-type", default="", help="Optional filter, e.g. ai_generate_from_metadata")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without calling the API")
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key and not args.dry_run:
        raise SystemExit(
            f"Missing API key. Set it first, for example in PowerShell:\n"
            f'$env:{args.api_key_env}="YOUR_SILICONFLOW_KEY"'
        )

    input_path = Path(args.input)
    output_path = Path(args.out)
    prompt_rows = read_csv(input_path)
    existing = {} if args.overwrite else load_existing(output_path)

    output_fieldnames = list(prompt_rows[0].keys()) if prompt_rows else []
    for field in [
        "status",
        "error",
        "model",
        "base_url",
        "temperature",
        "max_tokens",
        "generated_at",
        "raw_response_id",
    ]:
        if field not in output_fieldnames:
            output_fieldnames.append(field)

    output_rows: list[dict[str, str]] = []
    processed = 0

    for row in prompt_rows:
        key = (row.get("source_id", ""), row.get("prompt_type", ""))
        if args.prompt_type and row.get("prompt_type") != args.prompt_type:
            if key in existing:
                output_rows.append(existing[key])
            else:
                skipped = dict(row)
                skipped.update(
                    {
                        "status": "skipped_filter",
                        "error": "",
                        "model": args.model,
                        "base_url": args.base_url,
                        "temperature": str(args.temperature),
                        "max_tokens": str(args.max_tokens),
                        "generated_at": "",
                        "raw_response_id": "",
                    }
                )
                output_rows.append(skipped)
            continue

        if key in existing:
            output_rows.append(existing[key])
            continue

        if args.limit and processed >= args.limit:
            pending = dict(row)
            pending.update(
                {
                    "status": "pending",
                    "error": "",
                    "model": args.model,
                    "base_url": args.base_url,
                    "temperature": str(args.temperature),
                    "max_tokens": str(args.max_tokens),
                    "generated_at": "",
                    "raw_response_id": "",
                }
            )
            output_rows.append(pending)
            continue

        print(f"[{processed + 1}] {row.get('prompt_type')} | {row.get('title', '')[:80]}", file=sys.stderr)

        result = dict(row)
        result.update(
            {
                "model": args.model,
                "base_url": args.base_url,
                "temperature": str(args.temperature),
                "max_tokens": str(args.max_tokens),
                "generated_at": now_iso(),
                "raw_response_id": "",
            }
        )

        if args.dry_run:
            result.update({"status": "dry_run", "error": "", "generated_text": ""})
        else:
            try:
                generated_text, raw = chat_completion(
                    api_key=api_key,
                    base_url=args.base_url,
                    model=args.model,
                    prompt=row.get("prompt", ""),
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    retries=args.retries,
                    retry_sleep=args.retry_sleep,
                )
                result.update(
                    {
                        "status": "ok",
                        "error": "",
                        "generated_text": generated_text,
                        "raw_response_id": str(raw.get("id", "")),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - keep row-level failures resumable.
                result.update({"status": "error", "error": str(exc), "generated_text": ""})

        output_rows.append(result)
        processed += 1
        write_csv(output_path, output_rows, output_fieldnames)
        if not args.dry_run:
            time.sleep(args.sleep)

    write_csv(output_path, output_rows, output_fieldnames)
    print(f"Wrote {len(output_rows)} rows to {output_path}")
    print(f"Processed new rows: {processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
