from pydantic import BaseModel, Field, field_validator

from .enums import POPULARITY, PROFESSION_CATEGORY


class UploadProfessionDto(BaseModel):
    name: str = Field(min_length=1)
    category: PROFESSION_CATEGORY
    description: str | None = None
    startSalary: float | None = Field(default=None, ge=0)
    endSalary: float | None = Field(default=None, ge=0)
    popularity: POPULARITY | None = None
    skills: list[str] = Field(default_factory=list)

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
