"""
Collect pilot-study paper abstracts from OpenAlex.

Example:
    python collect_openalex_abstracts.py --years 2010-2021 --per-year 5 --email your@email.com --out data_pre_llm.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def parse_year_range(value: str) -> list[int]:
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""

    positions: list[tuple[int, str]] = []
    for word, indexes in index.items():
        for position in indexes:
            positions.append((position, word))

    positions.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positions)


def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def request_json(url: str, *, retries: int = 8, sleep: float = 10.0) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "ai-detector-style-study/0.1"})
            with urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))

        except HTTPError as exc:
            last_error = exc

            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait_seconds = int(retry_after)
                else:
                    wait_seconds = sleep * (attempt + 1)

                print(f"HTTP 429 限流，等待 {wait_seconds:.0f} 秒后重试...", file=sys.stderr)
                time.sleep(wait_seconds)
                continue

            wait_seconds = sleep * (attempt + 1)
            print(f"HTTP {exc.code} 错误，等待 {wait_seconds:.0f} 秒后重试...", file=sys.stderr)
            time.sleep(wait_seconds)

        except URLError as exc:
            last_error = exc
            wait_seconds = sleep * (attempt + 1)
            print(f"网络错误，等待 {wait_seconds:.0f} 秒后重试...", file=sys.stderr)
            time.sleep(wait_seconds)

        except Exception as exc:
            last_error = exc
            wait_seconds = sleep * (attempt + 1)
            print(f"未知错误，等待 {wait_seconds:.0f} 秒后重试...", file=sys.stderr)
            time.sleep(wait_seconds)

    raise RuntimeError(f"Request failed after {retries} attempts: {url}") from last_error


def fetch_openalex_year(
    *,
    year: int,
    per_year: int,
    seed: int,
    email: str | None,
    api_key: str | None,
    language: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    sample_size = min(max(per_year * 5, 100), 1000)

    query = {
        "filter": f"publication_year:{year},type:article",
        "sample": str(sample_size),
        "seed": str(seed + year),
        "select": ",".join(
            [
                "id",
                "doi",
                "display_name",
                "publication_year",
                "language",
                "type",
                "primary_location",
                "topics",
                "abstract_inverted_index",
            ]
        ),
    }

    if email:
        query["mailto"] = email

    if api_key:
        query["api_key"] = api_key

    url = f"{OPENALEX_WORKS_URL}?{urlencode(query)}"
    data = request_json(url)
    results = data.get("results", [])

    random.Random(seed + year).shuffle(results)

    for item in results:
        if len(rows) >= per_year:
            break

        openalex_id = item.get("id", "")
        if not openalex_id or openalex_id in seen_ids:
            continue

        item_language = item.get("language") or ""
        if language != "any" and item_language != language:
            continue

        abstract = clean_html(abstract_from_inverted_index(item.get("abstract_inverted_index")))
        if len(abstract.split()) < 80:
            continue

        source = ((item.get("primary_location") or {}).get("source") or {})
        topics = item.get("topics") or []
        topic = topics[0].get("display_name", "") if topics else ""

        rows.append(
            {
                "id": openalex_id,
                "doi": item.get("doi") or "",
                "year": str(item.get("publication_year") or year),
                "language": item_language,
                "discipline": topic,
                "journal": source.get("display_name") or "",
                "title": clean_html(item.get("display_name") or ""),
                "abstract": abstract,
                "group": "human_pre_llm" if year <= 2021 else "human_post_llm",
                "source_type": "openalex_article_abstract",
                "word_count": str(len(abstract.split())),
            }
        )
        seen_ids.add(openalex_id)

    return rows


def write_csv(path: str, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "id",
        "doi",
        "year",
        "language",
        "discipline",
        "journal",
        "title",
        "abstract",
        "group",
        "source_type",
        "word_count",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2010-2021", help="Year range, e.g. 2010-2021 or 2010,2015,2020")
    parser.add_argument("--per-year", type=int, default=10)
    parser.add_argument("--language", default="en", help="Use 'en' for English or 'any' for no language filter")
    parser.add_argument("--email", default="", help="Optional contact email for OpenAlex polite usage")
    parser.add_argument("--api-key", default=os.getenv("OPENALEX_API_KEY", ""), help="Optional OpenAlex API key")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=5.0, help="Seconds to wait between yearly requests")
    parser.add_argument("--out", default="openalex_abstracts_pilot.csv")
    args = parser.parse_args()

    all_rows: list[dict[str, str]] = []
    years = parse_year_range(args.years)

    for year in years:
        rows = fetch_openalex_year(
            year=year,
            per_year=args.per_year,
            seed=args.seed,
            email=args.email or None,
            api_key=args.api_key or None,
            language=args.language,
        )

        print(f"{year}: collected {len(rows)} rows", file=sys.stderr)
        all_rows.extend(rows)

        # 每完成一年就保存一次，避免中途失败导致前面结果丢失。
        write_csv(args.out, all_rows)

        time.sleep(args.sleep)

    write_csv(args.out, all_rows)
    print(f"Wrote {len(all_rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())