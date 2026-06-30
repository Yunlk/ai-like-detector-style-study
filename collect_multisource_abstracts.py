"""
Collect scholarly abstracts from multiple metadata platforms.

Supported sources:
    - OpenAlex Works API
    - PubMed / NCBI E-utilities
    - arXiv API
    - Semantic Scholar Graph API
    - Crossref REST API, best-effort because abstracts are often absent

The script is intentionally conservative:
    - collects abstracts and metadata, not full text
    - uses per-source limits and sleep intervals
    - supports output de-duplication
    - writes a shared schema for later AI-control prompt generation

Example:
    python collect_multisource_abstracts.py ^
      --sources openalex,pubmed,arxiv,semanticscholar ^
      --years 2010-2021,2023-2026 ^
      --per-year 20 ^
      --email you@example.com ^
      --out real_abstracts_multisource.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import re
import ssl
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


USER_AGENT = "ai-detector-style-study/0.3"
MAX_RETRY_WAIT_SECONDS = 180
OPENALEX_ENDPOINT = "https://api.openalex.org/works"
PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"
S2_PAPER_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
CROSSREF_WORKS = "https://api.crossref.org/works"

DEFAULT_ARXIV_CATEGORIES = [
    "cs.CL",
    "cs.AI",
    "cs.LG",
    "stat.ML",
    "physics.soc-ph",
    "math.ST",
]
DEFAULT_PUBMED_TERMS = [
    "clinical",
    "genomics",
    "epidemiology",
    "neuroscience",
    "public health",
    "oncology",
]
DEFAULT_S2_FIELDS = "paperId,title,abstract,year,venue,publicationVenue,externalIds,fieldsOfStudy,s2FieldsOfStudy,publicationTypes,journal"


OUTPUT_FIELDS = [
    "id",
    "doi",
    "year",
    "period",
    "language",
    "discipline",
    "journal",
    "title",
    "abstract",
    "group",
    "source_type",
    "source_platform",
    "platform_type",
    "publisher",
    "is_preprint",
    "source_url",
    "api_source_id",
    "ai_policy_known",
    "requires_ai_disclosure",
    "copyediting_exempt_from_disclosure",
    "policy_source_url",
    "policy_checked_date",
    "word_count",
]


def now_policy_date() -> str:
    return date.today().isoformat()


def parse_years(spec: str) -> list[int]:
    years: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            years.update(range(int(start), int(end) + 1))
        else:
            years.add(int(part))
    return sorted(years)


def period_for_year(year: int) -> str:
    if year <= 2021:
        return "pre_chatgpt"
    if year == 2022:
        return "transition_2022"
    return "post_chatgpt"


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff]", text or ""))


def abstract_from_openalex_inverted_index(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    pairs: list[tuple[int, str]] = []
    for word, positions in index.items():
        for pos in positions:
            pairs.append((pos, word))
    return " ".join(word for _, word in sorted(pairs))


def request_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    retries: int = 4,
    retry_sleep: float = 5.0,
    timeout: int = 45,
    ssl_context: ssl.SSLContext | None = None,
) -> bytes:
    request_headers = {"User-Agent": USER_AGENT}
    if headers:
        request_headers.update(headers)

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers=request_headers)
            with urlopen(req, timeout=timeout, context=ssl_context) as response:
                return response.read()
        except HTTPError as exc:
            last_error = exc
            body = exc.read().decode("utf-8", errors="replace")
            retry_after = exc.headers.get("Retry-After")
            if "Insufficient budget" in body:
                raise RuntimeError(f"HTTP {exc.code}: API budget exhausted. {body[:240]}") from exc
            if exc.code in {408, 409, 429, 500, 502, 503, 504} and attempt < retries - 1:
                wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else retry_sleep * (attempt + 1)
                if wait_seconds > MAX_RETRY_WAIT_SECONDS:
                    raise RuntimeError(
                        f"HTTP {exc.code}: retry-after too long ({wait_seconds}s). {body[:240]}"
                    ) from exc
                wait_seconds += random.uniform(0, 1.5)
                print(f"HTTP {exc.code}; wait {wait_seconds:.1f}s. {body[:160]}", file=sys.stderr)
                time.sleep(wait_seconds)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            last_error = exc
            if attempt < retries - 1:
                wait_seconds = retry_sleep * (attempt + 1) + random.uniform(0, 1.5)
                print(f"Network error; wait {wait_seconds:.1f}s. {exc}", file=sys.stderr)
                time.sleep(wait_seconds)
                continue
    raise RuntimeError("Request failed after retries.") from last_error


def request_json(url: str, **kwargs: Any) -> dict[str, Any]:
    return json.loads(request_bytes(url, **kwargs).decode("utf-8"))


def ssl_context_from_args(args: argparse.Namespace) -> ssl.SSLContext | None:
    if args.insecure_ssl:
        print("WARNING: SSL verification is disabled for this run.", file=sys.stderr)
        return ssl._create_unverified_context()
    if args.use_certifi:
        try:
            import certifi  # type: ignore

            return ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            print("certifi is not installed; falling back to the system certificate store.", file=sys.stderr)
    return None


def read_existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        return {(row.get("source_platform", ""), row.get("id", "")) for row in rows}


def append_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    out = {field: row.get(field, "") for field in OUTPUT_FIELDS}
    out["title"] = clean_text(out["title"])
    out["abstract"] = clean_text(out["abstract"])
    out["journal"] = clean_text(out["journal"])
    out["discipline"] = clean_text(out["discipline"])
    out["publisher"] = clean_text(out["publisher"])
    out["language"] = out["language"] or "en"
    out["period"] = out["period"] or period_for_year(int(out["year"]))
    out["word_count"] = str(word_count(out["abstract"]))
    return out


def fetch_openalex_year(year: int, per_year: int, args: argparse.Namespace) -> list[dict[str, str]]:
    params = {
        "filter": f"publication_year:{year},type:article,has_abstract:true",
        "sample": str(max(per_year * 4, per_year)),
        "seed": str(args.seed + year),
        "select": "id,doi,display_name,publication_year,language,type,primary_location,topics,abstract_inverted_index",
    }
    if args.email:
        params["mailto"] = args.email
    url = f"{OPENALEX_ENDPOINT}?{urllib.parse.urlencode(params)}"
    headers = {}
    api_key = os.getenv("OPENALEX_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = request_json(url, headers=headers, ssl_context=args.ssl_context)
    output: list[dict[str, str]] = []
    for item in data.get("results", []):
        abstract = abstract_from_openalex_inverted_index(item.get("abstract_inverted_index"))
        if word_count(abstract) < args.min_words:
            continue
        location = item.get("primary_location") or {}
        source = location.get("source") or {}
        topics = item.get("topics") or []
        discipline = topics[0].get("display_name", "") if topics else ""
        output.append(
            normalize_row(
                {
                    "id": item.get("id", ""),
                    "doi": item.get("doi", "") or "",
                    "year": str(item.get("publication_year", year)),
                    "period": period_for_year(year),
                    "language": item.get("language", "") or "en",
                    "discipline": discipline,
                    "journal": source.get("display_name", "") or "",
                    "title": item.get("display_name", "") or "",
                    "abstract": abstract,
                    "group": f"real_{period_for_year(year)}",
                    "source_type": "published_abstract",
                    "source_platform": "openalex",
                    "platform_type": "metadata_index",
                    "publisher": source.get("host_organization_name", "") or "",
                    "is_preprint": "0",
                    "source_url": item.get("id", ""),
                    "api_source_id": item.get("id", ""),
                    "policy_checked_date": now_policy_date(),
                }
            )
        )
        if len(output) >= per_year:
            break
    return output


def fetch_pubmed_year(year: int, per_year: int, args: argparse.Namespace) -> list[dict[str, str]]:
    term = random.choice(args.pubmed_terms)
    search_query = f'("{year}"[Date - Publication] : "{year}"[Date - Publication]) AND english[Language] AND {term}'
    params = {
        "db": "pubmed",
        "term": search_query,
        "retmode": "json",
        "retmax": str(max(per_year * 4, per_year)),
        "sort": "relevance",
        "tool": "ai_detector_style_study",
    }
    if args.email:
        params["email"] = args.email
    ncbi_key = os.getenv("NCBI_API_KEY", "").strip()
    if ncbi_key:
        params["api_key"] = ncbi_key
    search_url = f"{PUBMED_ESEARCH}?{urllib.parse.urlencode(params)}"
    ids = request_json(search_url, ssl_context=args.ssl_context).get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    output: list[dict[str, str]] = []
    for start in range(0, len(ids), args.pubmed_fetch_batch):
        id_batch = ids[start : start + args.pubmed_fetch_batch]
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_batch),
            "retmode": "xml",
            "tool": "ai_detector_style_study",
        }
        if args.email:
            fetch_params["email"] = args.email
        if ncbi_key:
            fetch_params["api_key"] = ncbi_key
        fetch_url = f"{PUBMED_EFETCH}?{urllib.parse.urlencode(fetch_params)}"
        root = ET.fromstring(request_bytes(fetch_url, ssl_context=args.ssl_context))

        for article in root.findall(".//PubmedArticle"):
            row = pubmed_article_to_row(article, year, args)
            if row is None:
                continue
            output.append(row)
            if len(output) >= per_year:
                return output
        if args.sleep:
            time.sleep(min(args.sleep, 1.0))
    return output


def pubmed_article_to_row(article: ET.Element, year: int, args: argparse.Namespace) -> dict[str, str] | None:
        pmid = "".join(article.findtext(".//PMID") or "").strip()
        title = "".join(article.findtext(".//ArticleTitle") or "").strip()
        abstract_parts = [node.text or "" for node in article.findall(".//Abstract/AbstractText")]
        abstract = clean_text(" ".join(abstract_parts))
        if word_count(abstract) < args.min_words:
            return None

        journal = "".join(article.findtext(".//Journal/Title") or "").strip()
        pub_year = article.findtext(".//JournalIssue/PubDate/Year") or str(year)
        doi = ""
        for node in article.findall(".//ArticleId"):
            if node.attrib.get("IdType") == "doi":
                doi = node.text or ""
                break
        mesh_terms = [node.findtext("DescriptorName") or "" for node in article.findall(".//MeshHeading")]
        discipline = "; ".join(term for term in mesh_terms[:3] if term)
        return normalize_row(
            {
                "id": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "doi": f"https://doi.org/{doi}" if doi and not doi.startswith("http") else doi,
                "year": str(pub_year) if str(pub_year).isdigit() else str(year),
                "period": period_for_year(year),
                "language": "en",
                "discipline": discipline,
                "journal": journal,
                "title": title,
                "abstract": abstract,
                "group": f"real_{period_for_year(year)}",
                "source_type": "published_abstract",
                "source_platform": "pubmed",
                "platform_type": "biomedical_index",
                "publisher": "",
                "is_preprint": "0",
                "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "api_source_id": pmid,
                "policy_checked_date": now_policy_date(),
            }
        )


def fetch_arxiv_year(year: int, per_year: int, args: argparse.Namespace) -> list[dict[str, str]]:
    category = random.choice(args.arxiv_categories)
    start = random.randint(0, max(args.arxiv_max_start, 0)) if args.arxiv_max_start else 0
    query = f"cat:{category} AND submittedDate:[{year}01010000 TO {year}12312359]"
    params = {
        "search_query": query,
        "start": str(start),
        "max_results": str(max(per_year * 2, per_year)),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{ARXIV_ENDPOINT}?{urllib.parse.urlencode(params)}"
    root = ET.fromstring(request_bytes(url, ssl_context=args.ssl_context))
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    output: list[dict[str, str]] = []
    for entry in root.findall("atom:entry", ns):
        arxiv_id = entry.findtext("atom:id", default="", namespaces=ns)
        title = entry.findtext("atom:title", default="", namespaces=ns)
        summary = entry.findtext("atom:summary", default="", namespaces=ns)
        published = entry.findtext("atom:published", default="", namespaces=ns)
        primary = entry.find("arxiv:primary_category", ns)
        discipline = primary.attrib.get("term", category) if primary is not None else category
        doi = entry.findtext("arxiv:doi", default="", namespaces=ns)
        abstract = clean_text(summary)
        if word_count(abstract) < args.min_words:
            continue
        output.append(
            normalize_row(
                {
                    "id": arxiv_id,
                    "doi": f"https://doi.org/{doi}" if doi and not doi.startswith("http") else doi,
                    "year": published[:4] if published[:4].isdigit() else str(year),
                    "period": period_for_year(year),
                    "language": "en",
                    "discipline": discipline,
                    "journal": "arXiv",
                    "title": title,
                    "abstract": abstract,
                    "group": f"real_{period_for_year(year)}",
                    "source_type": "preprint_abstract",
                    "source_platform": "arxiv",
                    "platform_type": "preprint_server",
                    "publisher": "arXiv",
                    "is_preprint": "1",
                    "source_url": arxiv_id,
                    "api_source_id": arxiv_id.rsplit("/", 1)[-1],
                    "policy_checked_date": now_policy_date(),
                }
            )
        )
        if len(output) >= per_year:
            break
    return output


def fetch_semanticscholar_year(year: int, per_year: int, args: argparse.Namespace) -> list[dict[str, str]]:
    query = random.choice(args.s2_queries)
    params = {
        "query": query,
        "year": str(year),
        "fields": DEFAULT_S2_FIELDS,
        "limit": str(min(max(per_year * 4, per_year), 100)),
    }
    url = f"{S2_PAPER_SEARCH}?{urllib.parse.urlencode(params)}"
    headers = {}
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if s2_key:
        headers["x-api-key"] = s2_key
    data = request_json(url, headers=headers, ssl_context=args.ssl_context)

    output: list[dict[str, str]] = []
    for item in data.get("data", []):
        abstract = clean_text(item.get("abstract") or "")
        if word_count(abstract) < args.min_words:
            continue
        external = item.get("externalIds") or {}
        doi = external.get("DOI", "")
        venue = item.get("venue") or ""
        journal_data = item.get("journal") or {}
        if not venue:
            venue = journal_data.get("name", "")
        fields = item.get("fieldsOfStudy") or []
        s2_fields = item.get("s2FieldsOfStudy") or []
        discipline = "; ".join(fields[:3]) if fields else "; ".join((f.get("category", "") for f in s2_fields[:3]))
        publication_types = item.get("publicationTypes") or []
        is_preprint = "1" if any("Preprint" in str(pt) for pt in publication_types) else "0"
        paper_id = item.get("paperId", "")
        output.append(
            normalize_row(
                {
                    "id": f"https://www.semanticscholar.org/paper/{paper_id}",
                    "doi": f"https://doi.org/{doi}" if doi and not doi.startswith("http") else doi,
                    "year": str(item.get("year") or year),
                    "period": period_for_year(year),
                    "language": "en",
                    "discipline": discipline,
                    "journal": venue,
                    "title": item.get("title", ""),
                    "abstract": abstract,
                    "group": f"real_{period_for_year(year)}",
                    "source_type": "published_or_preprint_abstract",
                    "source_platform": "semanticscholar",
                    "platform_type": "metadata_index",
                    "publisher": "",
                    "is_preprint": is_preprint,
                    "source_url": f"https://www.semanticscholar.org/paper/{paper_id}",
                    "api_source_id": paper_id,
                    "policy_checked_date": now_policy_date(),
                }
            )
        )
        if len(output) >= per_year:
            break
    return output


def fetch_crossref_year(year: int, per_year: int, args: argparse.Namespace) -> list[dict[str, str]]:
    query = random.choice(args.crossref_queries)
    filters = f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31,type:journal-article"
    params = {
        "filter": filters,
        "query": query,
        "rows": str(min(max(per_year * 5, per_year), 100)),
        "select": "DOI,title,abstract,published-print,published-online,container-title,publisher,subject,URL",
    }
    if args.email:
        params["mailto"] = args.email
    url = f"{CROSSREF_WORKS}?{urllib.parse.urlencode(params)}"
    data = request_json(url, ssl_context=args.ssl_context)
    items = data.get("message", {}).get("items", [])

    output: list[dict[str, str]] = []
    for item in items:
        abstract = clean_text(item.get("abstract") or "")
        if word_count(abstract) < args.min_words:
            continue
        title = " ".join(item.get("title") or [])
        journal = "; ".join(item.get("container-title") or [])
        subjects = item.get("subject") or []
        doi = item.get("DOI", "")
        output.append(
            normalize_row(
                {
                    "id": f"https://doi.org/{doi}" if doi else item.get("URL", ""),
                    "doi": f"https://doi.org/{doi}" if doi else "",
                    "year": str(year),
                    "period": period_for_year(year),
                    "language": "en",
                    "discipline": "; ".join(subjects[:3]),
                    "journal": journal,
                    "title": title,
                    "abstract": abstract,
                    "group": f"real_{period_for_year(year)}",
                    "source_type": "published_abstract",
                    "source_platform": "crossref",
                    "platform_type": "doi_metadata_index",
                    "publisher": item.get("publisher", ""),
                    "is_preprint": "0",
                    "source_url": item.get("URL", ""),
                    "api_source_id": doi,
                    "policy_checked_date": now_policy_date(),
                }
            )
        )
        if len(output) >= per_year:
            break
    return output


FETCHERS = {
    "openalex": fetch_openalex_year,
    "pubmed": fetch_pubmed_year,
    "arxiv": fetch_arxiv_year,
    "semanticscholar": fetch_semanticscholar_year,
    "crossref": fetch_crossref_year,
}


def dedupe_rows(rows: list[dict[str, str]], existing_keys: set[tuple[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen = set(existing_keys)
    seen_doi: set[str] = set()
    for row in rows:
        key = (row.get("source_platform", ""), row.get("id", ""))
        doi = row.get("doi", "").lower().strip()
        if key in seen:
            continue
        if doi and doi in seen_doi:
            continue
        seen.add(key)
        if doi:
            seen_doi.add(doi)
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="openalex,pubmed,arxiv,semanticscholar")
    parser.add_argument("--years", default="2010-2021,2023-2026")
    parser.add_argument("--per-year", type=int, default=10, help="Rows per source per year")
    parser.add_argument("--target-total", type=int, default=0, help="Stop after this many newly collected rows")
    parser.add_argument("--out", required=True)
    parser.add_argument("--email", default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-words", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--use-certifi", action="store_true", help="Use certifi's CA bundle if certifi is installed")
    parser.add_argument("--insecure-ssl", action="store_true", help="Disable SSL verification; use only if local certs fail")
    parser.add_argument("--arxiv-categories", default=",".join(DEFAULT_ARXIV_CATEGORIES))
    parser.add_argument("--arxiv-max-start", type=int, default=150)
    parser.add_argument("--pubmed-terms", default=",".join(DEFAULT_PUBMED_TERMS))
    parser.add_argument("--pubmed-fetch-batch", type=int, default=100)
    parser.add_argument("--s2-queries", default="machine learning,clinical trial,education,materials,public health,social science")
    parser.add_argument("--crossref-queries", default="machine learning,clinical,education,materials,social science")
    args = parser.parse_args()
    args.ssl_context = ssl_context_from_args(args)

    random.seed(args.seed)
    sources = [source.strip().lower() for source in args.sources.split(",") if source.strip()]
    unknown = [source for source in sources if source not in FETCHERS]
    if unknown:
        raise SystemExit(f"Unknown source(s): {unknown}. Available: {sorted(FETCHERS)}")

    args.arxiv_categories = [item.strip() for item in args.arxiv_categories.split(",") if item.strip()]
    args.pubmed_terms = [item.strip() for item in args.pubmed_terms.split(",") if item.strip()]
    args.s2_queries = [item.strip() for item in args.s2_queries.split(",") if item.strip()]
    args.crossref_queries = [item.strip() for item in args.crossref_queries.split(",") if item.strip()]

    years = parse_years(args.years)
    output_path = Path(args.out)
    existing_keys = read_existing_keys(output_path)
    total_new = 0

    for year in years:
        for source in sources:
            if args.target_total and total_new >= args.target_total:
                print(f"Reached target-total={args.target_total}.")
                return 0

            print(f"Fetching {source} {year}...", file=sys.stderr)
            try:
                rows = FETCHERS[source](year, args.per_year, args)
            except Exception as exc:  # noqa: BLE001 - keep multisource run resumable.
                print(f"ERROR {source} {year}: {exc}", file=sys.stderr)
                continue

            rows = dedupe_rows(rows, existing_keys)
            if args.target_total:
                rows = rows[: max(args.target_total - total_new, 0)]
            append_csv(output_path, rows)
            for row in rows:
                existing_keys.add((row.get("source_platform", ""), row.get("id", "")))
            total_new += len(rows)
            print(f"{source} {year}: wrote {len(rows)} rows; total_new={total_new}", file=sys.stderr)
            if args.sleep:
                time.sleep(args.sleep)

    print(f"Done. New rows: {total_new}. Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
