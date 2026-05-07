import json
from langchain_core.messages import HumanMessage, SystemMessage

from models.context import Context
from models.llm_result import (
    ActionSummarizerResult,
    ClassifierResult,
    FulfilledAction,
    InferencerResult,
    TopicSuggesterResult,
)
from utils.llm import (
    invoke_llm,
    LLM_USE_CACHE,
    LLM_INCLUDE_DEBUG_FIELDS,
    prune_debug_fields_from_schema,
)


def suggest_topics(
    context: Context,
    classified_category: ClassifierResult,
    required_actions: ActionSummarizerResult | None = None,
    fulfilled_actions: list[FulfilledAction] | None = None,
    inferred_result: InferencerResult | None = None,
    dry_run: bool = False,
) -> TopicSuggesterResult:
    system_prompt = f"""\
You are a helpful assistant that suggests at most 4 conversation topic bullet points for the job seeker to reply for a potential referral or intro call.
Each bullet point should be an extremely short simple phrase (not sentence).
You will be given the conversation history, classified category, actions required by the referrer and fulfilled by the job seeker (optional), inferred result regarding the referral possibility (optional), job seeker profile (optional), and referrer profile (optional).
When suggesting topics, always evaluate the latest messages first and from the existing facts.
As long as the referral is possible, suggest `Ask for a brief call` in the topics to help they keep the conversation going and know each other better.
Never suggest todo items that are not in the conversation history or fulfilled actions.
Never make assumptions.
Never make up facts.

Topic Examples (but not limited to):
1. Thank you
2. Express interest
3. Express background match
4. Ask for a brief call
5. Ask for alternative referrers
6. Follow-up
"""

    if LLM_INCLUDE_DEBUG_FIELDS:
        system_prompt += """
You will need to provide conversation topics along with confidence score and reason.
"""

    user_prompt = f"""\
Conversation Messages:
{context.messages}

Classified Category:
{classified_category}
"""

    if required_actions:
        user_prompt += f"""
Required Actions From Referrer:
{required_actions}
"""

    if fulfilled_actions:
        user_prompt += f"""
Fulfilled Actions From Job Seeker:
{fulfilled_actions}
"""

    if inferred_result:
        user_prompt += f"""
Inferred Result:
{inferred_result}
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

    schema = TopicSuggesterResult.model_json_schema()
    if not LLM_INCLUDE_DEBUG_FIELDS:
        schema = prune_debug_fields_from_schema(schema)

    output_schema = {
        "name": "topic_suggester_result",
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

    return TopicSuggesterResult(**json.loads(response.content[0]["text"]))
