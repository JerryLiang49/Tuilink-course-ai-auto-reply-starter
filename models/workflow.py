from models.base import BaseModel
from models.context import Context
from models.llm_result import (
    ActionSummarizerResult,
    ClassifierResult,
    FulfilledAction,
    InferencerResult,
    MessageGeneratorResult,
    TopicSuggesterResult,
)


class State(BaseModel):
    """Mutable workflow state passed between the Lambda handler and nodes.

    The orchestrator fills this model incrementally. Callers may send a partially
    completed state back into the handler, allowing already-computed steps such
    as classification or topic suggestion to be reused instead of regenerated.
    """

    # Human-readable marker for where orchestration stopped or completed.
    step: str | None = None

    # Required input: conversation history plus optional user/referrer profiles.
    context: Context

    # Classification of the current conversation state.
    classified_category: ClassifierResult | None = None

    # Actions/questions extracted from the referrer's request.
    required_actions: ActionSummarizerResult | None = None

    # User-provided answers or files that fulfill required actions.
    fulfilled_actions: list[FulfilledAction] | None = None

    # Inferred conclusion after comparing required actions with fulfilled inputs.
    inferred_result: InferencerResult | None = None

    # Candidate topics the assistant believes the reply should cover.
    suggested_topics: TopicSuggesterResult | None = None

    # Topic strings selected for final reply generation.
    selected_topics: list[str] | None = None

    # Final generated reply message and optional debug metadata.
    generated_reply_message: MessageGeneratorResult | None = None
