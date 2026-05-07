import json
from langchain_core.messages import HumanMessage, SystemMessage

from models.context import Context
from models.llm_result import (
    ActionSummarizerResult,
    ClassifierResult,
    FulfilledAction,
    InferencerResult,
)
from utils.llm import (
    invoke_llm,
    LLM_USE_CACHE,
    LLM_INCLUDE_DEBUG_FIELDS,
    prune_debug_fields_from_schema,
)


def infer_from_actions(
    context: Context,
    classified_category: ClassifierResult,
    required_actions: ActionSummarizerResult,
    fulfilled_actions: list[FulfilledAction],
    dry_run: bool = False,
) -> InferencerResult:
    """Infer the conversation result after required actions are fulfilled.

    The fulfilled actions are raw user inputs. This node converts them into a
    compact conclusion, such as whether the referral still seems possible and
    which facts should guide the reply.
    """

    system_prompt = """\
You are a helpful assistant that infers the current referral outcome after a job seeker fulfills requested actions.
Review the conversation, the required actions, and the fulfilled action inputs.
Decide whether all required actions are fulfilled.
Infer whether the referral path is possible, not possible, or unknown.
Never claim an action is fulfilled unless the fulfilled action input provides enough information.
Never make up facts beyond the conversation and fulfilled actions.
"""

    if LLM_INCLUDE_DEBUG_FIELDS:
        system_prompt += """
Provide confidence score and reason for the inference.
"""

    user_prompt = f"""\
Conversation Messages:
{context.messages}

Classified Category:
{classified_category}

Required Actions:
{required_actions}

Fulfilled Actions:
{fulfilled_actions}
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

    schema = InferencerResult.model_json_schema()
    if not LLM_INCLUDE_DEBUG_FIELDS:
        schema = prune_debug_fields_from_schema(schema)

    output_schema = {
        "name": "inferencer_result",
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

    return InferencerResult(**json.loads(response.content[0]["text"]))
