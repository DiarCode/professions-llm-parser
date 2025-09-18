from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from .enums import PROFESSION_CATEGORY, POPULARITY

class UploadProfessionDto(BaseModel):
    name: str = Field(min_length=1)
    category: PROFESSION_CATEGORY
    description: Optional[str] = None
    startSalary: Optional[float] = Field(default=None, ge=0)
    endSalary: Optional[float] = Field(default=None, ge=0)
    popularity: Optional[POPULARITY] = None
    skills: List[str] = Field(default_factory=list)

    @field_validator("skills", mode="before")
    @classmethod
    def _clean_skills(cls, v):
        if not isinstance(v, list):
            return []
        out = []
        seen = set()
        for s in v:
            if isinstance(s, str):
                t = s.strip()
                if t and t not in seen:
                    seen.add(t)
                    out.append(t)
        return out

    @field_validator("endSalary")
    @classmethod
    def _range_ok(cls, v, info):
        start = info.data.get("startSalary")
        if v is not None and start is not None and v < start:
            # swap to satisfy constraint rather than failing the row
            return start
        return v
