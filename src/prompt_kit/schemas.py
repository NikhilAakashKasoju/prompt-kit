from pydantic import BaseModel, Field

class SummarizeRequest(BaseModel):
    text: str = Field(min_length=1)
    max_sentences: int = Field(default=3, ge=1, le=10)

class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1)
    labels: list[str] = Field(min_length=2)

class ExtractRequest(BaseModel):
    text: str = Field(min_length=1)
    fields: list[str] = Field(min_length = 1)

class Summary(BaseModel):
    summary: str
    key_points: list[str]

class Classification(BaseModel):
    label: str = Field(min_length = 1)
    confidence: float = Field(ge = 0, le = 1)
    reasoning: str = Field(min_length = 1)

class Extraction(BaseModel):
    data: dict[str, str | None]
