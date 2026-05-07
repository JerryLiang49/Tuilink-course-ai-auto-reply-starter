import json
from langchain_core.messages import HumanMessage, SystemMessage

from models.context import Context
from models.llm_result import ActionSummarizerResult, ClassifierResult
from utils.llm import (
    invoke_llm,
    LLM_USE_CACHE,
    LLM_INCLUDE_DEBUG_FIELDS,
    prune_debug_fields_from_schema,
)


def summarize_actions(
    context: Context,
    classified_category: ClassifierResult,
    dry_run: bool = False,
) -> ActionSummarizerResult:
    """Extract required job-seeker actions from the latest referrer messages.

    This node only runs when the classifier category says human action is
    required. Its output lets the workflow pause and ask the job seeker for
    missing information instead of generating a premature reply.
    """

    system_prompt = """\
You are a helpful assistant that summarizes actions or answers required from a job seeker before continuing a referral conversation.
Only extract actions explicitly requested by the referrer in the conversation.
Focus on the latest referrer messages.
Each action should be specific, short, and fulfillable by the job seeker.
Never invent requirements that are not present in the conversation.
Use stable snake_case action_id values, such as `send_resume`, `share_job_link`, or `confirm_availability`.
"""

    if LLM_INCLUDE_DEBUG_FIELDS:
        system_prompt += """
For each required action, provide confidence score, reason, and referenced message ids.
"""

    user_prompt = f"""\
Conversation Messages:
{context.messages}

Classified Category:
{classified_category}
"""

    if context.user_profile:
        user_prompt += f"""
Job Seeker Profile:
{context.user_profile}
"""

    if context.referrer_profile:
        user_prompt += f"""
Referrer Profile:
{context.referrer_profile}
"""

    schema = ActionSummarizerResult.model_json_schema()
    if not LLM_INCLUDE_DEBUG_FIELDS:
        schema = prune_debug_fields_from_schema(schema)

    output_schema = {
        "name": "action_summarizer_result",
        "strict": True,
        "type": "json_schema",
        "schema": schema,
    }

    if dry_run:
        return system_prompt, user_prompt, output_schema

    response = invoke_llm(
        input=[
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        text={"format": output_schema},
        use_cache=LLM_USE_CACHE,
    )

    return ActionSummarizerResult(**json.loads(response.content[0]["text"]))
