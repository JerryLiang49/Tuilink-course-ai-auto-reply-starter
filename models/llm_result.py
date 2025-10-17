from typing_extensions import Literal
from pydantic import ConfigDict

from models.base import BaseModel
from typing import Optional


class ClassifierResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str
    confidence: Optional[float] = None
    reason: Optional[str] = None
    referenced_message_ids: list[str]


class Topic(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str
    confidence: Optional[float] = None
    reason: Optional[str] = None


class TopicSuggesterResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topics: list[Topic]


class MessageGeneratorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str
    confidence: Optional[float] = None
    reason: Optional[str] = None

