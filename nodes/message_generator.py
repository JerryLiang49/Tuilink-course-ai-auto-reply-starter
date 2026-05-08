import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from models.context import Context
from models.llm_result import (
    ActionSummarizerResult,
    ClassifierResult,
    FulfilledAction,
    InferencerResult,
    MessageGeneratorResult,
)
from utils.llm import (
    OPENAI_API_KEY,
    invoke_llm,
    LLM_USE_CACHE,
    LLM_INCLUDE_DEBUG_FIELDS,
    prune_debug_fields_from_schema,
)


def generate_message(
    context: Context,
    classified_category: ClassifierResult,
    selected_topics: list[str],
    required_actions: ActionSummarizerResult | None = None,
    fulfilled_actions: list[FulfilledAction] | None = None,
    inferred_result: InferencerResult | None = None,
    dry_run: bool = False,
) -> MessageGeneratorResult:
    system_prompt = f"""\
You are a helpful assistant that generates a message for the job seeker to reply for a potential referral or intro call.
You will be given the conversation history, classified category, selected topics for new message, actions required by the referrer and fulfilled by the job seeker (optional), inferred result regarding the referral possibility (optional), job seeker profile (optional), and referrer profile (optional).
When generating the message, always evaluate the latest existing messages first.
Never make up facts.
Never mention anything you don't know based on the user prompt.
Never use placeholders in the message since this will be the final message content sent to the referrer.
If needed to ask some questions to the referrer, never ask more than 2 questions at the same time since the referrer might get overwhelmed.
"""

    if LLM_INCLUDE_DEBUG_FIELDS:
        system_prompt += """
You will need to generate a message along with confidence score and reason.
"""

    user_prompt = f"""\
Conversation Messages:
{context.messages}

Classified Category:
{classified_category}

Selected Topics:
{selected_topics}
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

    schema = MessageGeneratorResult.model_json_schema()
    if not LLM_INCLUDE_DEBUG_FIELDS:
        schema = prune_debug_fields_from_schema(schema)

    output_schema = {
        "name": "message_generator_result",
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
        # Use GPT 5 nano for better performance
        llm=ChatOpenAI(model="gpt-5-nano", api_key=OPENAI_API_KEY),
    )

    return MessageGeneratorResult(**json.loads(response.content[0]["text"]))
