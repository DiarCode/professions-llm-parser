import argparse
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from providers.openai_professions_provider import OpenAIProfessionsProvider
from app_io.writers import write_professions_json, write_preflight_report
from domain.dto import UploadProfessionDto
from domain.enums import PROFESSION_CATEGORY

load_dotenv()

OUT_DIR = Path(os.getenv("OUT_DIR","out"))
CONCURRENCY = int(os.getenv("CONCURRENCY","8"))

def _slug(s: str) -> str:
    return s.strip().lower().replace(" ", "-")

def _parse_categories(s: Optional[str]) -> Optional[List[str]]:
    if not s: return None
    items = [x.strip().upper() for x in s.split(",") if x.strip()]
    # keep only valid enum names
    ok = set([e.value for e in PROFESSION_CATEGORY])
    out = [x for x in items if x in ok]
    return out or None

async def main():
    parser = argparse.ArgumentParser(
        description="Fetch 2025-relevant professions list with details (RU text, enum EN), validated to UploadProfessionsDto."
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    provider = OpenAIProfessionsProvider()

    categories = _parse_categories(args.categories)
    payload = await provider.list_professions(locale='Global', categories='all', max_items=None)

    raw = payload.get("professions") or []
    preflight: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    # validate + coerce
    for p in raw:
        try:
            dto = UploadProfessionDto(**p)
            rows.append(dto.model_dump())
        except Exception as e:
            preflight.append({"profession": p.get("name") or "unknown", "reason": f"validation_error:{type(e).__name__}"})

    # write
    slug = _slug(args.locale)
    prof_path = write_professions_json(OUT_DIR, slug, rows)
    report_path = write_preflight_report(OUT_DIR, preflight)

    print(
        "OK\n"
        f" Professions JSON: {prof_path}\n"
        f" Preflight report: {report_path}\n"
        f" Professions: {len(rows)}  Skipped: {len(preflight)}"
    )

if __name__ == "__main__":
    asyncio.run(main())
