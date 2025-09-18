import os, json, asyncio, re
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI
from openai import APIConnectionError, RateLimitError, APITimeoutError, InternalServerError, BadRequestError

# ---------- utils ----------
def _to_jsonable(obj):
    import enum
    if isinstance(obj, dict):  return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple): return [_to_jsonable(v) for v in obj]
    if isinstance(obj, enum.Enum): return obj.value
    return obj

def _safe_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s or "{}")
    except Exception:
        try:
            i, j = s.find("{"), s.rfind("}")
            if i >= 0 and j > i: return json.loads(s[i:j+1])
        except Exception:
            pass
        return {}

def _schema_text(schema: dict) -> str:
    return json.dumps(_to_jsonable(schema.get("schema", schema)), ensure_ascii=False)

# ---------- enums for schema ----------
PROFESSION_CATEGORY_VALUES = [
    "TECHNOLOGY","MEDICINE","EDUCATION","FINANCE","ENGINEERING",
    "ARTS","BUSINESS","LAW","SCIENCE","SOCIAL_SCIENCES","GOVERNMENT","AGRICULTURE"
]
POPULARITY_VALUES = ["LOW","MEDIUM","HIGH"]

# ---------- JSON Schemas ----------
PROFESSION_SCHEMA = {
  "name": "Profession",
  "schema": {
    "type": "object",
    "properties": {
      "name": {"type":"string"},
      "category": {"type":"string","enum": PROFESSION_CATEGORY_VALUES},
      "description": {"type":["string","null"]},
      "startSalary": {"type":["number","null"], "minimum": 0},
      "endSalary": {"type":["number","null"], "minimum": 0},
      "popularity": {"type":["string","null"], "enum": POPULARITY_VALUES + [None]},
      "skills": {
        "type":"array",
        "items":{"type":"string"},
        "uniqueItems": True
      }
    },
    "required": ["name","category","skills"],
    "additionalProperties": False
  },
  "strict": True
}

PROFESSIONS_PAYLOAD_SCHEMA = {
  "name": "ProfessionsPayload",
  "schema": {
    "type": "object",
    "properties": {
      "professions": {
        "type": "array",
        "items": PROFESSION_SCHEMA["schema"],
        "minItems": 50
      },
      "sourceNote": {"type":["string","null"]}
    },
    "required": ["professions"],
    "additionalProperties": False
  },
  "strict": True
}

# ---------- system/user prompts (2025, web-first, strict) ----------
SYS_PROF_LIST = (
  "Ты — строгий исследователь с доступом к веб-поиску. Твоя задача — составить актуальный на 2025 год "
  "список профессий с деталями. Источники: официальные описания профессий, отраслевые классификаторы, "
  "актуальные справочники вакансий, профстандарты. Не включай явно устаревшие или несуществующие роли. "
  "Синонимы/переформулировки объединяй в одну профессию. Текст — на русском (кроме значений enum). "
  "Отвечай строго JSON по заданной схеме."
)

def _user_prof_list(locale: str, categories: Optional[List[str]], max_items: Optional[int]) -> str:
    cats = ", ".join(categories) if categories else "ВСЕ КАТЕГОРИИ"
    cap = f"Макс.позиций: {max_items}" if max_items else "Макс.позиций: на усмотрение (но не менее 50, без воды)"
    return (
        f"Локаль/страна: {locale}\n"
        f"Категории: {cats}\n"
        f"{cap}\n"
        "Для каждой профессии верни: name, category (enum), краткое описание (если доступно), "
        "startSalary/endSalary в местной валюте как числа (можно null, если нет достоверных данных), "
        "popularity (LOW/MEDIUM/HIGH, можно null, если источники противоречивы), skills (минимум 5 ключевых навыков).\n"
        "skills — это список значимых компетенций без дубликатов, лаконично.\n"
        "startSalary <= endSalary; если не можешь подтвердить — ставь null. "
        "Не придумывай данных. Дедуплицируй профессии, используй наиболее общий официальный титул."
    )

# ---------- provider ----------
class OpenAIProfessionsProvider:
    def __init__(self, debug: bool = False):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required")
        self.client = AsyncOpenAI(api_key=key)
        self.web_model = os.getenv("OPENAI_WEB_MODEL", "gpt-5-mini")
        self.fallback_model = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-5-mini")
        self.use_web = (os.getenv("USE_WEB_SEARCH","1").lower() in {"1","true","yes"})
        self.debug = debug or (os.getenv("DEBUG_LLM","").lower() in {"1","true","yes"})
        self.req_timeout = float(os.getenv("LLM_TIMEOUT_SEC","60"))
        self.layer_retries = int(os.getenv("LLM_LAYER_RETRIES","2"))

    async def _with_timeout(self, coro):
        return await asyncio.wait_for(coro, timeout=self.req_timeout)

    def _extract_text(self, resp) -> str:
        txt = getattr(resp, "output_text", "") or ""
        if not txt and getattr(resp,"output",None):
            try: txt = resp.output[0].content[0].text
            except Exception: txt=""
        if not txt and getattr(resp,"choices",None):
            try: txt = resp.choices[0].message.content or ""
            except Exception: txt=""
        return txt

    async def _responses_call(self, system: str, user: str, schema: Dict[str,Any], with_tools: bool) -> Dict[str,Any]:
        kwargs = {
            "model": self.web_model if with_tools else self.fallback_model,
            "input":[{"role":"system","content":system},{"role":"user","content":user}],
            "response_format":{"type":"json_schema","json_schema":schema},
            "temperature":0,
        }
        if with_tools and self.use_web:
            kwargs["tools"] = [{"type":"web_search"}]
            kwargs["tool_choice"] = "auto"
        try:
            resp = await self._with_timeout(self.client.responses.create(**kwargs))
            text = self._extract_text(resp)
            if self.debug: print("\n--- RESP ---\n", (text or "")[:1200], "\n---\n")
            return _safe_json(text)
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError, BadRequestError, Exception) as e:
            if self.debug: print(f"[WARN] responses({with_tools=}) failed: {type(e).__name__}: {e}")
            return {}

    async def _chat_call(self, system: str, user: str, schema: Dict[str,Any]) -> Dict[str,Any]:
        try:
            resp = await self._with_timeout(self.client.chat.completions.create(
                model=self.fallback_model,
                messages=[
                    {"role":"system","content":system},
                    {"role":"user","content": user + "\nВерни ТОЛЬКО JSON по этой схеме:\n" + _schema_text(schema)}
                ],
                response_format={"type":"json_object"},
                temperature=0,
            ))
            text = self._extract_text(resp)
            if self.debug: print("\n--- CHAT ---\n", (text or "")[:1200], "\n---\n")
            return _safe_json(text)
        except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError, BadRequestError, Exception) as e:
            if self.debug: print(f"[WARN] chat failed: {type(e).__name__}: {e}")
            return {}

    async def _json_schema_safe(self, system: str, user: str, schema: Dict[str,Any]) -> Dict[str,Any]:
        async def _try(fn):
            last = {}
            for i in range(self.layer_retries):
                out = await fn()
                if out: return out
                await asyncio.sleep(min(1.0*(i+1), 3.0))
                last = out
            return last

        out = await _try(lambda: self._responses_call(system, user, schema, with_tools=True))
        if out: return out
        out = await _try(lambda: self._responses_call(system, user, schema, with_tools=False))
        if out: return out
        out = await _try(lambda: self._chat_call(system, user, schema))
        return out or {}

    async def list_professions(self, locale: str, categories: Optional[List[str]], max_items: Optional[int]) -> Dict[str,Any]:
        user = _user_prof_list(locale, categories, max_items)
        out = await self._json_schema_safe(SYS_PROF_LIST, user, PROFESSIONS_PAYLOAD_SCHEMA)
        # always return sane shape
        out.setdefault("professions", [])
        if not isinstance(out["professions"], list): out["professions"] = []
        # post-fix invalid salary ranges if any
        for p in out["professions"]:
            try:
                s = p.get("startSalary"); e = p.get("endSalary")
                if isinstance(s,(int,float)) and isinstance(e,(int,float)) and s is not None and e is not None and s>e:
                    p["startSalary"], p["endSalary"] = e, s
            except Exception:
                continue
        return out
