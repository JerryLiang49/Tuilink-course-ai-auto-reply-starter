from typing import Literal

from models.base import BaseModel


class UserProfile(BaseModel):
    """Known information about the job seeker asking for help.

    This is optional context for the LLM. The current starter only stores the
    user's name, but the separate model makes it easy to add fields such as
    target role, background, resume summary, or location later.
    """

    # Display name of the job seeker who will receive the generated reply.
    name: str


class ReferrerProfile(BaseModel):
    """Known information about the referrer or professional contact.

    This profile is also optional. It lets the prompt refer to the other person
    by name and provides a natural place to add company, title, relationship, or
    hiring-team metadata in future iterations.
    """

    # Display name of the person the job seeker is messaging.
    name: str


class ConversationMessage(BaseModel):
    """One message in the referral conversation history.

    Messages are ordered in ``Context.messages``. The classifier and generator
    use the latest messages most heavily, so stable ids and timestamps are
    important for explaining which messages influenced a classification.
    """

    # Stable message identifier, used by classifier debug output references.
    id: str

    # Speaker role. The workflow assumes only these two sides of the conversation.
    role: Literal["job seeker", "referrer"]

    # Sender display name, preserved separately from role for prompt clarity.
    name: str

    # Main text content of the message.
    body_text: str

    # Extracted text from attachments, if any. Keep as an empty string when absent.
    attachments_text: str

    # Delivery timestamp, currently represented as an integer epoch value.
    delivered_at: int


class Context(BaseModel):
    """Full input context passed into the AI auto-reply workflow.

    The Lambda handler receives a ``State`` object, and ``context`` is the
    required part of that state. The optional profiles give the LLM more
    personalization signal, while ``messages`` carries the actual conversation
    to classify and answer.
    """

    # Chronological conversation history. The workflow expects at least one item.
    messages: list[ConversationMessage]

    # Optional job seeker metadata for personalization.
    user_profile: UserProfile | None = None

    # Optional referrer metadata for personalization.
    referrer_profile: ReferrerProfile | None = None
