# providers/openai_professions_provider.py
import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

# ---------------- utils ----------------


def _to_jsonable(obj):
    import enum
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, enum.Enum):
        return obj.value
    return obj


def _schema_text(schema: dict) -> str:
    return json.dumps(schema.get("schema", schema), ensure_ascii=False)


def _safe_json(s: str) -> dict[str, Any]:
    try:
        return json.loads(s or "{}")
    except Exception:
        # try to salvage a JSON object
        try:
            i, j = s.find("{"), s.rfind("}")
            if i >= 0 and j > i:
                return json.loads(s[i:j+1])
        except Exception:
            pass
        return {}


def _dedup_strings(items: list[str]) -> list[str]:
    out, seen = [], set()
    for it in items:
        t = (it or "").strip()
        k = t.lower()
        if t and k not in seen:
            seen.add(k)
            out.append(t)
    return out

# ---------------- enums (mirror TS) ----------------


PROFESSION_CATEGORY_VALUES = [
    "TECHNOLOGY", "MEDICINE", "EDUCATION", "FINANCE", "ENGINEERING",
    "ARTS", "BUSINESS", "LAW", "SCIENCE", "SOCIAL_SCIENCES", "GOVERNMENT", "AGRICULTURE"
]
POPULARITY_VALUES = ["LOW", "MEDIUM", "HIGH"]

# ---------------- JSON Schemas ----------------

# Stage A: names only (safe, flexible)
PROFESSION_NAMES_SCHEMA = {
    "name": "ProfessionNamesPayload",
    "schema": {
        "type": "object",
        "properties": {
            "names": {"type": "array", "items": {"type": "string"}},
            "sourceNote": {"type": ["string", "null"]}
        },
        "required": ["names"],
        "additionalProperties": False
    },
    "strict": True
}

# Stage B: one profession with full details (maps to UploadProfessionDto)
PROFESSION_DETAIL_SCHEMA = {
    "name": "ProfessionDetail",
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "category": {"type": "string", "enum": PROFESSION_CATEGORY_VALUES},
            "description": {"type": ["string", "null"]},
            "startSalary": {"type": ["number", "null"], "minimum": 0},
            "endSalary": {"type": ["number", "null"], "minimum": 0},
            "popularity": {"type": ["string", "null"], "enum": POPULARITY_VALUES + [None]},
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "uniqueItems": True
            }
        },
        "required": ["name", "category", "skills"],
        "additionalProperties": False
    },
    "strict": True
}

# ---------------- prompts (2025, web-first, RU text, enums EN) ----------------

SYS_NAMES = (
    "Ты — строгий исследователь с доступом к веб-поиску. Задача: собрать АКТУАЛЬНЫЙ на 2025 год список профессий "
    "(официальные названия ролей/профессий). Источники: государственные классификаторы, профстандарты, крупные агрегаторы вакансий, "
    "отраслевые справочники. Объединяй синонимы в одну форму (наиболее общий официальный титул). "
    "Не включай устаревшие или вымышленные роли. Текст — на русском (только названия). Отвечай СТРОГО JSON по схеме."
)


def _user_names(locale: str, category: str | None, max_items: int | None) -> str:
    cat = category or "ВСЕ КАТЕГОРИИ"
    cap = f"Нужно ~{max_items} названий" if max_items else "Нужно МАКСИМАЛЬНО ПОЛНЫЙ список (100+ если возможно)"
    return (
        f"Локаль/страна: {locale}\n"
        f"Категория: {cat}\n"
        f"{cap}\n"
        "Верни массив 'names' со списком названий профессий без описаний."
    )


SYS_DETAIL = (
    "Ты — строгий исследователь с доступом к веб-поиску. Для указанной профессии составь АКТУАЛЬНОЕ на 2025 описание "
    "и атрибуты. Источники: официальные профстандарты, отраслевые справочники, крупные агрегаторы вакансий. "
    "Текст — на русском (кроме значений enum). Не выдумывай данные. Зарплаты — в KZT (Тенге) валюте Казахстана, если нет достоверных данных — null. "
    "Навыки — 5–10 ключевых, без дубликатов, кратко. Категорию выбери из enum."
)


def _user_detail(locale: str, name: str) -> str:
    return (
        f"Локаль/страна: {locale}\n"
        f"Профессия: {name}\n"
        "Верни один JSON-объект по схеме: name, category (enum), description (если доступно), "
        "startSalary/endSalary (числа или null), popularity (LOW/MEDIUM/HIGH или null), skills (список строк, 5–10)."
    )

# ---------------- provider ----------------


class OpenAIProfessionsProvider:
    """
    Two-phase, fault-tolerant provider:
      - list_profession_names(locale, categories) -> List[str]
      - get_profession_detail(locale, name) -> Dict
    Uses Responses API with web_search first, then non-tools, then Chat fallback.
    Never raises; returns safe shapes.
    """

    def __init__(self, debug: bool = False):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required")
        self.client = AsyncOpenAI(api_key=key)
        self.web_model = os.getenv("OPENAI_WEB_MODEL", "gpt-5-mini")
        self.fallback_model = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-5-mini")
        self.use_web = os.getenv("USE_WEB_SEARCH", "1").lower() in {
            "1", "true", "yes"}
        self.debug = debug or (os.getenv("DEBUG_LLM", "").lower() in {
                               "1", "true", "yes"})
        self.req_timeout = float(os.getenv("LLM_TIMEOUT_SEC", "60"))
        self.layer_retries = int(os.getenv("LLM_LAYER_RETRIES", "2"))

    # ---- internals ----
    async def _with_timeout(self, coro):
        return await asyncio.wait_for(coro, timeout=self.req_timeout)

    def _extract_text(self, resp) -> str:
        txt = getattr(resp, "output_text", "") or ""
        if not txt and getattr(resp, "output", None):
            try:
                txt = resp.output[0].content[0].text
            except Exception:
                txt = ""
        if not txt and getattr(resp, "choices", None):
            try:
                txt = resp.choices[0].message.content or ""
            except Exception:
                txt = ""
        return txt

    async def _responses_call(self, system: str, user: str, schema: dict[str, Any], with_tools: bool) -> dict[str, Any]:
        kwargs = {
            "model": self.web_model if with_tools else self.fallback_model,
            "input": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": schema},
            "temperature": 0,
        }
        if with_tools and self.use_web:
            kwargs["tools"] = [{"type": "web_search"}]
            kwargs["tool_choice"] = "auto"
        try:
            resp = await self._with_timeout(self.client.responses.create(**kwargs))
            text = self._extract_text(resp)
            if self.debug:
                print(
                    "\n--- RESP (tools={}) ---\n{}\n---\n".format(with_tools, (text or "")[:800]))
            return _safe_json(text)
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError, BadRequestError, Exception) as e:
            if self.debug:
                print(f"[WARN] responses failed: {type(e).__name__}: {e}")
            return {}

    async def _chat_call(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._with_timeout(self.client.chat.completions.create(
                model=self.fallback_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user +
                        "\nВерни ТОЛЬКО JSON по схеме:\n" + _schema_text(schema)}
                ],
                response_format={"type": "json_object"},
                temperature=0,
            ))
            text = self._extract_text(resp)
            if self.debug:
                print("\n--- CHAT ---\n{}\n---\n".format((text or "")[:800]))
            return _safe_json(text)
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError, BadRequestError, Exception) as e:
            if self.debug:
                print(f"[WARN] chat failed: {type(e).__name__}: {e}")
            return {}

    async def _json_schema_safe(self, system: str, user: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        async def _try(fn):
            last = {}
            for i in range(self.layer_retries):
                out = await fn()
                if out:
                    return out
                await asyncio.sleep(min(1.0*(i+1), 3.0))
                last = out
            return last

        out = await _try(lambda: self._responses_call(system, user, schema, with_tools=True))
        if out:
            return out
        out = await _try(lambda: self._responses_call(system, user, schema, with_tools=False))
        if out:
            return out
        out = await _try(lambda: self._chat_call(system, user, schema))
        return out or {}

    # ---- Stage A: names list ----
    async def list_profession_names(self, locale: str, category: Optional[str], max_items: Optional[int]) -> List[str]:
        """
        Returns a list of profession names (strings). Never raises. May be empty.
        """
        user = _user_names(locale, category, max_items)
        out = await self._json_schema_safe(SYS_NAMES, user, PROFESSION_NAMES_SCHEMA)
        names = out.get("names") or []
        names = [n for n in names if isinstance(n, str)]
        return _dedup_strings(names)

    # ---- Stage B: per-profession details ----
    async def get_profession_detail(self, locale: str, name: str) -> Dict[str, Any]:
        """
        Returns a dict matching PROFESSION_DETAIL_SCHEMA. Never raises.
        """
        user = _user_detail(locale, name)
        out = await self._json_schema_safe(SYS_DETAIL, user, PROFESSION_DETAIL_SCHEMA)
        # Normalize/guard
        out.setdefault("name", name)
        # safe default; will be validated later
        out.setdefault("category", "BUSINESS")
        out.setdefault("description", None)
        out.setdefault("startSalary", None)
        out.setdefault("endSalary", None)
        out.setdefault("popularity", None)
        if not isinstance(out.get("skills"), list):
            out["skills"] = []
        # salary sanity
        s, e = out.get("startSalary"), out.get("endSalary")
        if isinstance(s, (int, float)) and isinstance(e, (int, float)) and s is not None and e is not None and s > e:
            out["startSalary"], out["endSalary"] = e, s
        return out

    # ---- Fallback seeding if names list is empty (keeps pipeline useful) ----
    def seed_names(self, categories: Optional[List[str]] = None) -> List[str]:
        base = {
            "TECHNOLOGY": [
                "Разработчик программного обеспечения", "Инженер по данным", "Инженер DevOps",
                "Специалист по кибербезопасности", "Аналитик данных", "ML-инженер", "Системный администратор",
                "Тестировщик (QA-инженер)", "Архитектор программного обеспечения", "Продакт-менеджер"
            ],
            "MEDICINE": [
                "Врач-терапевт", "Хирург", "Стоматолог", "Медсестра", "Фармацевт",
                "Радиолог", "Анестезиолог", "Педиатр", "Врач общей практики", "Лаборант"
            ],
            "BUSINESS": [
                "Финансовый аналитик", "Маркетолог", "Бизнес-аналитик", "Менеджер по продажам",
                "HR-менеджер", "Логист", "Закупщик", "Операционный менеджер", "Аудитор", "Бухгалтер"
            ],
        }
        if categories:
            chosen = []
            for c in categories:
                chosen.extend(base.get(c, []))
            return _dedup_strings(chosen)
        # Global seed
        seed = []
        for v in base.values():
            seed.extend(v)
        return _dedup_strings(seed)
