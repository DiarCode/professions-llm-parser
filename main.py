import argparse
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from app_io.writers import write_preflight_report, write_professions_json
from domain.dto import UploadProfessionDto
from domain.enums import PROFESSION_CATEGORY
from providers.openai_professions_provider import OpenAIProfessionsProvider

load_dotenv()

OUT_DIR = Path(os.getenv("OUT_DIR", "out"))
# higher parallelism for Stage B
CONCURRENCY = int(os.getenv("CONCURRENCY", "12"))


def _slug(s: str) -> str:
    return s.strip().lower().replace(" ", "-")


def _parse_categories(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    items = [x.strip().upper() for x in s.split(",") if x.strip()]
    ok = set([e.value for e in PROFESSION_CATEGORY])
    out = [x for x in items if x in ok]
    return out or None


def load_names_file(path: Path) -> List[str]:
    """Read TXT file with one profession per line. Supports # comments. Deduplicates."""
    if not path.exists():
        return []
    out: List[str] = []
    seen: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = (raw or "").strip()
            if not line or line.startswith("#"):
                continue
            key = line.lower()
            if key not in seen:
                seen.add(key)
                out.append(line)
    return out


async def _stage_a_collect_names(
    provider: OpenAIProfessionsProvider,
    locale: str,
    categories: Optional[List[str]],
    per_category_target: int
) -> List[str]:
    names: List[str] = []
    if categories:
        for cat in tqdm(categories, desc="Сбор названий по категориям", total=len(categories)):
            lst = await provider.list_profession_names(locale, cat, per_category_target)
            names.extend(lst)
    else:
        lst = await provider.list_profession_names(locale, None, per_category_target * 6)
        names.extend(lst)

    dedup, seen = [], set()
    for n in names:
        k = (n or "").strip().lower()
        if n and k not in seen:
            seen.add(k)
            dedup.append(n)

    if not dedup:
        dedup = provider.seed_names(categories)
    return dedup


async def _stage_b_enrich_details(
    provider: OpenAIProfessionsProvider,
    locale: str,
    names: List[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sem = asyncio.Semaphore(CONCURRENCY)
    rows: List[dict[str, Any]] = []
    preflight: List[dict[str, Any]] = []

    async def _one(name: str):
        nonlocal rows, preflight
        async with sem:
            try:
                detail = await provider.get_profession_detail(locale, name)
                dto = UploadProfessionDto(**detail)
                rows.append(dto.model_dump())
            except Exception as e:
                preflight.append(
                    {"profession": name, "reason": f"validation_or_fetch_error:{type(e).__name__}"})

    tasks = [asyncio.create_task(_one(n)) for n in names]
    pbar = tqdm(total=len(tasks), desc="Обогащение деталями (Stage B)")
    for fut in asyncio.as_completed(tasks):
        await fut
        pbar.update(1)
    pbar.close()

    return rows, preflight


async def main():
    parser = argparse.ArgumentParser(
        description="Professions pipeline (2025): use TXT names if provided, else discover → enrich."
    )
    parser.add_argument("--locale", required=True,
                        help="Locale/country (e.g., Kazakhstan, Russia, Global)")
    parser.add_argument(
        "--categories", help="Comma-separated enum categories (e.g., TECHNOLOGY,MEDICINE)")
    parser.add_argument("--per-category", type=int, default=60,
                        help="Target names per category in Stage A")
    parser.add_argument("--max", type=int, default=None,
                        help="Hard cap for final validated rows after Stage B")
    parser.add_argument("--names-file", type=Path,
                        help="TXT file with one profession per line; if exists, Stage A is skipped")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    provider = OpenAIProfessionsProvider()

    categories = _parse_categories(args.categories)

    # ===== Stage A (optional): read names from file, else fetch =====
    names: List[str] = []
    if args.names_file:
        names = load_names_file(args.names_file)
        if names:
            print(
                f"Использую профессии из файла: {args.names_file}  (всего {len(names)})")
    if not names:
        names = await _stage_a_collect_names(provider, args.locale, categories, args.per_category)
        print(f"Собрано названий профессий (Stage A): {len(names)}")

    # ===== Stage B: enrich details =====
    rows, preflight = await _stage_b_enrich_details(provider, args.locale, names)

    if args.max and len(rows) > args.max:
        rows = rows[:args.max]

    slug = _slug(args.locale)
    prof_path = write_professions_json(OUT_DIR, slug, rows)
    report_path = write_preflight_report(OUT_DIR, preflight)

    print(
        "OK\n"
        f" Locale: {args.locale}\n"
        f" Professions JSON: {prof_path}\n"
        f" Preflight report: {report_path}\n"
        f" Names used: {len(names)}  Valid details: {len(rows)}  Skipped: {len(preflight)}"
    )

if __name__ == "__main__":
    asyncio.run(main())
