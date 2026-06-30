"""
Generate AI-control abstracts with SiliconFlow using per-row model assignment.

Input prompt CSV must include a `model` column. This script skips rows already
completed with status=ok and generated_text, so it can be safely resumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv_atomic(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8-sig",
        newline="",
        delete=False,
        dir=str(path.parent),
    ) as file:
        temp_path = Path(file.name)
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def make_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("source_id", ""),
        row.get("prompt_type", ""),
        row.get("prompt_style", ""),
        row.get("model", ""),
    )


def load_existing(path: Path) -> dict[tuple[str, str, str, str], dict[str, str]]:
    if not path.exists():
        return {}
    existing: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in read_csv(path):
        if row.get("status") == "ok" and row.get("generated_text", "").strip():
            existing[make_key(row)] = row
    return existing


def output_fieldnames(prompt_rows: list[dict[str, str]]) -> list[str]:
    fieldnames = list(prompt_rows[0].keys()) if prompt_rows else []
    for field in [
        "generated_text",
        "status",
        "error",
        "base_url",
        "temperature",
        "max_tokens",
        "generated_at",
        "raw_response_id",
    ]:
        if field not in fieldnames:
            fieldnames.append(field)
    return fieldnames


def initialized_row(
    row: dict[str, str],
    *,
    status: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    error: str = "",
) -> dict[str, str]:
    result = dict(row)
    result.update(
        {
            "status": status,
            "error": error,
            "base_url": base_url,
            "temperature": str(temperature),
            "max_tokens": str(max_tokens),
            "generated_at": "",
            "raw_response_id": "",
        }
    )
    result.setdefault("generated_text", "")
    return result


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
        "User-Agent": "ai-detector-style-study/0.4",
    }

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(endpoint, data=body, headers=headers, method="POST")
            with urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"].strip()
            if not text:
                raise RuntimeError(f"Empty model response: {json.dumps(data, ensure_ascii=False)[:500]}")
            return text, data
        except HTTPError as exc:
            last_error = exc
            retry_after = exc.headers.get("Retry-After")
            error_body = exc.read().decode("utf-8", errors="replace")
            should_retry = exc.code in {408, 409, 429, 500, 502, 503, 504}
            if should_retry and attempt < retries - 1:
                wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else retry_sleep * (attempt + 1)
                wait_seconds += random.uniform(0, 1.5)
                print(f"HTTP {exc.code}; wait {wait_seconds:.1f}s then retry. {error_body[:200]}", file=sys.stderr)
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc
        except URLError as exc:
            last_error = exc
            if attempt < retries - 1:
                wait_seconds = retry_sleep * (attempt + 1) + random.uniform(0, 1.5)
                print(f"Network error; wait {wait_seconds:.1f}s then retry. {exc}", file=sys.stderr)
                time.sleep(wait_seconds)
                continue
            break

    raise RuntimeError("Request failed after retries.") from last_error


def process_one(index: int, row: dict[str, str], args: argparse.Namespace, api_key: str) -> tuple[int, dict[str, str]]:
    model = row.get("model", "").strip()
    result = initialized_row(
        row,
        status="running",
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    result["generated_at"] = now_iso()

    if not model:
        result.update({"status": "error", "error": "Missing per-row model.", "generated_text": ""})
        return index, result

    if args.dry_run:
        result.update({"status": "dry_run", "error": "", "generated_text": ""})
        return index, result

    try:
        generated_text, raw = chat_completion(
            api_key=api_key,
            base_url=args.base_url,
            model=model,
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

    return index, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default=os.getenv("SILICONFLOW_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=360)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prompt-type", default="")
    parser.add_argument("--models", default="", help="Optional comma-separated allowlist of models to process")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.save_every < 1:
        raise SystemExit("--save-every must be >= 1")

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key and not args.dry_run:
        raise SystemExit(
            f"Missing API key. Set it first, for example in PowerShell:\n"
            f'$env:{args.api_key_env}="YOUR_SILICONFLOW_KEY"'
        )

    allowed_models = {item.strip() for item in args.models.split(",") if item.strip()}
    prompt_rows = read_csv(Path(args.input))
    existing = {} if args.overwrite else load_existing(Path(args.out))
    fieldnames = output_fieldnames(prompt_rows)

    output_rows: list[dict[str, str]] = []
    tasks: list[tuple[int, dict[str, str]]] = []
    selected_count = 0

    for row in prompt_rows:
        key = make_key(row)
        row_model = row.get("model", "").strip()

        filtered = False
        if args.prompt_type and row.get("prompt_type") != args.prompt_type:
            filtered = True
        if allowed_models and row_model not in allowed_models:
            filtered = True
        if filtered:
            output_rows.append(
                existing.get(
                    key,
                    initialized_row(
                        row,
                        status="skipped_filter",
                        base_url=args.base_url,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    ),
                )
            )
            continue

        if key in existing:
            output_rows.append(existing[key])
            continue

        if args.limit and selected_count >= args.limit:
            output_rows.append(
                initialized_row(
                    row,
                    status="pending",
                    base_url=args.base_url,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            )
            continue

        index = len(output_rows)
        output_rows.append(
            initialized_row(
                row,
                status="queued",
                base_url=args.base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        )
        tasks.append((index, row))
        selected_count += 1

    write_csv_atomic(Path(args.out), output_rows, fieldnames)
    if not tasks:
        print(f"Wrote {len(output_rows)} rows to {args.out}")
        print("Processed new rows: 0")
        return 0

    completed = 0
    print(f"Queued {len(tasks)} rows with {args.workers} workers.", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, index, row, args, api_key): (index, row)
            for index, row in tasks
        }
        for future in as_completed(futures):
            index, original = futures[future]
            try:
                _, result = future.result()
            except Exception as exc:
                result = initialized_row(
                    original,
                    status="error",
                    base_url=args.base_url,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    error=str(exc),
                )
            output_rows[index] = result
            completed += 1
            print(
                f"[{completed}/{len(tasks)}] {result.get('status')} | "
                f"{result.get('model', '')} | {result.get('prompt_type', '')} | "
                f"{result.get('title', '')[:70]}",
                file=sys.stderr,
            )
            if completed % args.save_every == 0:
                write_csv_atomic(Path(args.out), output_rows, fieldnames)

    write_csv_atomic(Path(args.out), output_rows, fieldnames)
    print(f"Wrote {len(output_rows)} rows to {args.out}")
    print(f"Processed new rows: {completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
