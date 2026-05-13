import json
import logging
from base64 import b64decode
from json import JSONDecodeError

from utils.json import to_json_compatible
from models.workflow import State
from nodes.orchestrator import orchestrate

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
}


def build_response(status_code: int, body: dict):
    """Build an API Gateway compatible JSON response."""

    return {
        "statusCode": status_code,
        "headers": CORS_HEADERS,
        "body": json.dumps(body),
    }


def parse_request_body(event) -> dict:
    """Parse the JSON body sent by API Gateway.

    The online API is expected to receive either {"state": {...}} or the State
    payload directly. API Gateway can also base64-encode bodies, so decode that
    before parsing JSON.
    """

    raw_body = event.get("body")
    if raw_body is None:
        raise ValueError("Missing request body")

    if event.get("isBase64Encoded"):
        raw_body = b64decode(raw_body).decode("utf-8")

    if isinstance(raw_body, dict):
        return raw_body

    parsed_body = json.loads(raw_body)
    if not isinstance(parsed_body, dict):
        raise ValueError("Request body must be a JSON object")

    return parsed_body


def extract_state(payload: dict) -> State:
    """Create workflow State from the supported request shapes."""

    state_payload = payload.get("state", payload)
    if not isinstance(state_payload, dict):
        raise ValueError("Request body must contain a JSON object state")

    return State(**state_payload)


def get_reply_message(state: State) -> str | None:
    """Return final generated wording when the workflow has produced it."""

    if state.generated_reply_message is None:
        return None

    return state.generated_reply_message.message


def get_required_action_prompt(state: State) -> str | None:
    """Return the prompt to show the user when more action input is needed."""

    if state.required_actions is None:
        return None

    return state.required_actions.prompt_to_user


def handler(event, context):
    """Lambda entrypoint for online AI auto-reply generation."""

    logger.info("Received event: %s", json.dumps(event))

    http_method = event.get("httpMethod")

    if http_method == "OPTIONS":
        return build_response(200, {"ok": True})

    if http_method == "GET":
        return build_response(
            200,
            {
                "message": "AI Auto Reply API is running",
                "usage": "POST /ai-reply with JSON body {'state': {...}}",
            },
        )

    try:
        payload = parse_request_body(event)
        state = extract_state(payload)
        state = orchestrate(state)

        return build_response(
            200,
            {
                "step": state.step,
                "reply_message": get_reply_message(state),
                "required_action_prompt": get_required_action_prompt(state),
                "state": to_json_compatible(state),
            },
        )
    except (JSONDecodeError, ValueError) as e:
        logger.warning("Bad request: %s", e)
        return build_response(400, {"error": str(e)})
    except Exception as e:
        logger.exception("Failed to generate AI auto reply")
        return build_response(500, {"error": str(e)})
