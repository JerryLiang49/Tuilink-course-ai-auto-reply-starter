from rexpand_pyutils_file import read_file

from models.category import Category, ExtendedCategory
from models.workflow import State
from nodes.classifier import classify_conversation
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
        dry_run=False,
    )
    state.step = "end: reply generated"
    return state


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
        # TODO: Finish the workflow with human action required
        state.step = "todo: finish the workflow"
        return state

    else:
        # No human action required → suggest topics (skip if already suggested)
        if state.suggested_topics is None:
            state.suggested_topics = suggest_topics(
                state.context, state.classified_category, dry_run=False
            )

        # Always auto-select topics and generate reply message (no extra topic selection step)
        return auto_select_topics_and_generate_reply_message(state)
