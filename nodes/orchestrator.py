from rexpand_pyutils_file import read_file

from models.category import Category, ExtendedCategory
from models.workflow import State
from nodes.action_summarizer import summarize_actions
from nodes.classifier import classify_conversation
from nodes.inferencer import infer_from_actions
from nodes.message_generator import generate_message
from nodes.topic_suggester import suggest_topics


CATEGORIES = [Category(**category) for category in read_file("./input/categories.json")]
EXTENDED_CATEGORY_LOOKUP = {
    category["category"]: ExtendedCategory(**category)
    for category in read_file("./input/categories.json")
}


def auto_select_topics_and_generate_reply_message(state: State) -> State:
    # Skip re-generation if a reply message already exists
    if state.generated_reply_message is not None:
        state.step = "end: reply generated"
        return state

    state.selected_topics = [topic.topic for topic in state.suggested_topics.topics]
    state.generated_reply_message = generate_message(
        state.context,
        state.classified_category,
        state.selected_topics,
        required_actions=state.required_actions,
        fulfilled_actions=state.fulfilled_actions,
        inferred_result=state.inferred_result,
        dry_run=False,
    )
    state.step = "end: reply generated"
    return state


def all_required_actions_fulfilled(state: State) -> bool:
    """Return whether every summarized action has matching fulfilled input."""

    if state.required_actions is None:
        return False

    required_action_ids = {
        action.action_id for action in state.required_actions.actions
    }

    # If the summarizer found no concrete actions, there is nothing to wait for.
    if not required_action_ids:
        return True

    fulfilled_action_ids = set()
    for action in state.fulfilled_actions or []:
        has_user_input = bool(action.answer or action.attachments_text)
        if action.is_fulfilled and has_user_input:
            fulfilled_action_ids.add(action.action_id)

    return required_action_ids.issubset(fulfilled_action_ids)


def suggest_topics_and_generate_reply_message(state: State) -> State:
    """Run the final topic-suggestion and message-generation portion."""

    if state.suggested_topics is None:
        state.suggested_topics = suggest_topics(
            state.context,
            state.classified_category,
            required_actions=state.required_actions,
            fulfilled_actions=state.fulfilled_actions,
            inferred_result=state.inferred_result,
            dry_run=False,
        )

    # Always auto-select topics and generate reply message in this starter app.
    return auto_select_topics_and_generate_reply_message(state)


def orchestrate(
    state: State,
) -> State:
    # 1) Classify conversation (skip if already classified)
    if state.classified_category is None:
        state.classified_category = classify_conversation(
            state.context, CATEGORIES, dry_run=False
        )

    # Get the extended category with post-processing indicators
    extended_category = EXTENDED_CATEGORY_LOOKUP[state.classified_category.category]

    # 2) If no reply is needed, finish
    if not extended_category.reply_needed:
        state.step = "end: no reply needed"
        return state

    # 3) Branch based on whether human action is required
    if extended_category.human_action_required:
        # Summarize what the referrer asked the job seeker to provide.
        if state.required_actions is None:
            state.required_actions = summarize_actions(
                state.context,
                state.classified_category,
                dry_run=False,
            )

        # Pause here until the caller returns with fulfilled_actions in State.
        if not all_required_actions_fulfilled(state):
            state.step = "awaiting: user actions"
            return state

        # Interpret fulfilled actions before suggesting topics or final wording.
        if state.inferred_result is None:
            state.inferred_result = infer_from_actions(
                state.context,
                state.classified_category,
                state.required_actions,
                state.fulfilled_actions or [],
                dry_run=False,
            )

        # If the inferencer still sees missing details, keep the workflow paused.
        if not state.inferred_result.actions_fulfilled:
            state.step = "awaiting: user actions"
            return state

    return suggest_topics_and_generate_reply_message(state)
