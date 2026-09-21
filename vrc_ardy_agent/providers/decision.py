"""Project action-planning contract, independent of the inference provider."""
import base64
from dataclasses import asdict
import json

_SYSTEM_PROMPT = """\
You are the avatar's high-level action planner inside VRChat.
Return exactly one plain JSON decision with no Markdown or surrounding text.
This endpoint uses project-owned decisions, NOT native API tool calls.
If current visual evidence is needed, return:
{"type":"observe_scene","query":"What specifically needs checking in the current scene"}
Otherwise return an actions object as below, with a required boolean requires_visual.
Set requires_visual=true when your answer/action depends on the current scene.
Never assert current appearance, identify a pointed-at object, or judge a visible
result without a fresh observation in this turn. If the target remains ambiguous,
ask the user. Greetings and generic gestures normally need no image.
The host may return an observation and ask you to continue the same turn.
Use execution_feedback to distinguish command acceptance from actual completion,
failure, cancellation, and unknown tracking. Do not infer physical success from
an idle state or absence of errors. Do not call service-native tools.

Final action examples (requires_visual is mandatory):
{"requires_visual":false,"actions":[{"type":"say","text":"short reply in the user's main language"}]}
{"requires_visual":false,"actions":[{"type":"ardy_motion","prompt":"Short English ASCII motion description.","duration_seconds":3}]}
{"requires_visual":false,"actions":[{"type":"say","text":"short reply"},{"type":"ardy_motion","prompt":"Short English ASCII motion description.","duration_seconds":3}]}

Rules:
- For a final action decision, return one or two actions. Never return an empty actions array.
- Include at most one say action and one body action (motion OR ardy_motion).
- Available registered motions are listed in available_motions. Only use those names.
- To play one, return {"type":"motion","name":"the_registered_name"}.
- A complete_clip motion plays exactly once and forbids a duration field.
- Other registered motions use their configured duration; an optional duration_seconds overrides it.
- Use ardy_motion for expressive motions needing generation. Never put preset commands in its prompt.
- Idle is maintained by the host; do not generate standing/idle actions.
- say.text is at most two short sentences and follows the user's main language.
- ardy_motion.prompt describes only a short expressive body motion in English ASCII.
- duration_seconds is a number from 1 through 10.
- Do not invent fields, intent, priorities, resources, sequences, conditions, or joint coordinates.
- Do not use ardy_motion for navigation, following, approaching, or searching in this PoC.
- Use images only when supplied with observation metadata; otherwise use the transcript and execution state.
"""


def messages(request):
    if request.screenshot is not None and (not isinstance(request.screenshot, bytes) or not request.screenshot):
        raise ValueError("request.screenshot must contain JPEG bytes")
    context = {
        "available_motions": list(request.available_motions),
        "turn_id": request.turn_id, "turn_version": request.turn_version,
        "transcript": request.transcript, "round_index": request.round_index,
        "observations": [asdict(item) for item in request.observations],
        "observation_history": list(request.conversation), "execution_feedback": request.execution_feedback,
        "runtime": {"body": request.body.value, "speech": request.speech.value,
                    "current_action_ids": list(request.current_action_ids), "last_error": request.last_error},
    }
    content = [{"type": "text", "text": json.dumps(context, ensure_ascii=False, separators=(",", ":"))}]
    if request.screenshot:
        image = base64.b64encode(request.screenshot).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "high"}})
    return [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": content}]


def decode_decision(envelope, request):
    if not isinstance(envelope, dict): raise RuntimeError("chat response must be an object")
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("chat response has no assistant choice")
    choice = choices[0]
    if choice.get("finish_reason") not in (None, "stop"): raise RuntimeError("chat completion is incomplete")
    service = envelope.get("hermes", {})
    if isinstance(service, dict) and (service.get("failed") or service.get("partial") or service.get("completed") is False):
        raise RuntimeError("provider reported failed or partial completion")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("tool_calls") or not isinstance(message.get("content"), str):
        raise RuntimeError("expected plain assistant content, without native tool calls")
    try: result = json.loads(message["content"])
    except (ValueError, TypeError) as exc: raise RuntimeError("assistant content must be one plain JSON object") from exc
    if not isinstance(result, dict): raise RuntimeError("assistant decision must be an object")
    if result.get("type") != "observe_scene" and not isinstance(result.get("requires_visual"), bool):
        raise RuntimeError("final decision requires requires_visual boolean")
    if request.record_model_metadata:
        request.record_model_metadata({"usage": envelope.get("usage"), "model": envelope.get("model"),
            "response_id": envelope.get("id"), "finish_reason": choice.get("finish_reason")})
    return result
