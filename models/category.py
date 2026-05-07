from models.base import BaseModel


class Category(BaseModel):
    """LLM-visible category definition used during conversation classification.

    These objects are loaded from ``input/categories.json`` and inserted into
    the classifier prompt. They describe the possible conversation states the
    model may choose from.
    """

    # Stable category id returned by the classifier, e.g. "agree_schedule_call".
    category: str

    # Short definition of when this category applies.
    description: str

    # Extra guidance and examples that help disambiguate similar categories.
    clarification: str


class ExtendedCategory(Category):
    """Category definition plus workflow-control flags.

    The classifier only returns ``category``. The orchestrator then looks up the
    matching ``ExtendedCategory`` to decide whether to stop, ask for human input,
    or continue automatically to topic suggestion and message generation.
    """

    # True when the job seeker must provide information or make a decision first.
    human_action_required: bool

    # False for categories where the conversation should end without a new reply.
    reply_needed: bool
