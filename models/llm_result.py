from typing_extensions import Literal
from pydantic import ConfigDict

from models.base import BaseModel
from typing import Optional


class ClassifierResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["category", "referenced_message_ids", "confidence", "reason"]
        },
    )
    category: str
    confidence: Optional[float] = None
    reason: Optional[str] = None
    referenced_message_ids: list[str]


class Topic(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"required": ["topic", "confidence", "reason"]},
    )
    topic: str
    confidence: Optional[float] = None
    reason: Optional[str] = None


class TopicSuggesterResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"required": ["topics"]},
    )
    topics: list[Topic]


class MessageGeneratorResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "required": ["message", "confidence", "reason"]
        },
    )
    message: str
    confidence: Optional[float] = None
    reason: Optional[str] = None

