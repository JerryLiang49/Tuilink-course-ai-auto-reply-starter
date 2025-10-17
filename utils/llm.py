import logging
import hashlib
import os
from typing import Any, Optional
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables.config import RunnableConfig
from langchain_core.language_models.base import LanguageModelInput
from rexpand_pyutils_file import read_file, write_file
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")


LLM_USE_CACHE: bool = os.getenv("LLM_USE_CACHE", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)

# Toggle for including debug fields (reason, confidence) in prompts/schemas
LLM_INCLUDE_DEBUG_FIELDS: bool = os.getenv(
    "LLM_INCLUDE_DEBUG_FIELDS", "true"
).strip().lower() in ("1", "true", "yes", "y", "on")

# Model selection and temperature
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4.1-mini").strip()
try:
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0").strip())
except ValueError:
    LLM_TEMPERATURE = 0.0


logging.info(f"LLM_MODEL: {LLM_MODEL}")
logging.info(f"LLM_TEMPERATURE: {LLM_TEMPERATURE}")
logging.info(f"LLM_USE_CACHE: {LLM_USE_CACHE}")
logging.info(f"LLM_INCLUDE_DEBUG_FIELDS: {LLM_INCLUDE_DEBUG_FIELDS}")


def _model_supports_temperature(model_name: str) -> bool:
    # Newer reasoning-style models like gpt-5 do not accept temperature
    return not (model_name.startswith("gpt-5"))


def prune_debug_fields_from_schema(schema: dict) -> dict:
    """Return a copy of the JSON schema without debug fields (reason, confidence).

    This removes these from any "properties" and from "required" arrays, recursively.
    """
    DEBUG_FIELDS = {"reason", "confidence"}

    def _prune(node: Any) -> Any:
        if isinstance(node, dict):
            new_node: dict[str, Any] = {}
            for key, value in node.items():
                if key == "properties" and isinstance(value, dict):
                    filtered_props = {
                        prop_key: _prune(prop_val)
                        for prop_key, prop_val in value.items()
                        if prop_key not in DEBUG_FIELDS
                    }
                    new_node[key] = filtered_props
                elif key == "required" and isinstance(value, list):
                    new_node[key] = [r for r in value if r not in DEBUG_FIELDS]
                else:
                    new_node[key] = _prune(value)
            return new_node
        elif isinstance(node, list):
            return [_prune(item) for item in node]
        else:
            return node

    return _prune(schema)


temperature_arg: Optional[float] = (
    LLM_TEMPERATURE if _model_supports_temperature(LLM_MODEL) else None
)

default_llm = ChatOpenAI(
    model=LLM_MODEL, temperature=temperature_arg, api_key=OPENAI_API_KEY
)


def invoke_llm(
    input: LanguageModelInput,
    config: Optional[RunnableConfig] = None,
    *,
    use_cache: bool = LLM_USE_CACHE,
    verbose: bool = False,
    llm: Optional[ChatOpenAI] = default_llm,
    **kwargs: Any,
) -> BaseMessage:
    if use_cache:
        # Create a hash of the input string
        input_hash = hashlib.md5((str(input) + "|" + str(config)).encode()).hexdigest()
        filepath = f"./.cache/{input_hash}.json"

        cached_response = read_file(filepath)
        if cached_response is not None:
            if verbose:
                logging.info(f"Cache hit: {filepath}")

            return AIMessage(**cached_response)
        else:
            if verbose:
                logging.info(f"Cache miss: {filepath}")

            response: BaseMessage = llm.invoke(input, config, **kwargs)
            write_file(filepath, response.model_dump())
            return response
    else:
        return llm.invoke(input, config, **kwargs)
