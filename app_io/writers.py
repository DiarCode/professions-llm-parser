from pathlib import Path
from typing import List, Dict, Any
import json

def write_professions_json(out_dir: Path, locale_slug: str, rows: List[Dict[str, Any]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"professions-{locale_slug}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump([r for r in rows], f, ensure_ascii=False, indent=2)
    return path

def write_preflight_report(out_dir: Path, preflight: List[Dict[str, Any]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "upload_preflight_report.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(preflight, f, ensure_ascii=False, indent=2)
    return path
