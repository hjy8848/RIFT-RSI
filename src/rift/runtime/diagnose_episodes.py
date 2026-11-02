"""Summarize persisted native episodes without exposing their private contents."""
from collections import Counter
import argparse
from pathlib import Path


def _business_success(episode):
    return getattr(episode.outcome, "business_goal_success", None)


def classify_episode(episode):
    """Assign one descriptive category using only recorded evidence.

    The final two labels describe observations, not a claim about the model's
    intent. A finished unsuccessful episode after a final rejected operation
    is distinguishable from other early finishes.
    """
    status = episode.status
    reason = episode.terminal_reason.lower()
    calls = [call for call in episode.model_calls if call.role == "service"]
    if status in ("service_protocol_error", "agent_error"):
        response = calls[-1].response if calls else {}
        choices = response.get("choices", []) if isinstance(response, dict) else []
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else []
            if isinstance(tool_calls, list) and len(tool_calls) > 1:
                return "multiple_tool_calls"
        if any(term in reason for term in ("unobserved", "previously observed", "never observed")):
            return "unobserved_object"
        if "schema" in reason or "model_output_invalid" in reason or "structured function call" in reason:
            return "schema_violation"
        if status == "service_protocol_error":
            return "other_service_protocol_error"
        return "agent_error"
    if status == "budget_exhausted":
        return "budget_exhausted"
    if status == "customer_finished":
        if _business_success(episode) is True:
            return "success"
        if _business_success(episode) is False:
            if episode.tool_executions and episode.tool_executions[-1].result.get("success") is False:
                return "finished_after_rejected_operation"
            return "customer_finished_unsuccessful"
        return "customer_finished_unknown"
    return status


def episode_metrics(episode):
    """Return lengths only when token counts are present in recorded calls."""
    lengths = []
    for call in episode.model_calls:
        if call.role != "service":
            continue
        generated = call.response.get("generated_token_ids") if isinstance(call.response, dict) else None
        if isinstance(generated, list):
            lengths.append(len(generated))
        elif type(call.completion_tokens) is int:
            lengths.append(call.completion_tokens)
    task = getattr(episode, 'task', None)
    refusal_codes = {goal.reason_code for goal in getattr(task, 'refusal_goals', ())}
    if not refusal_codes:
        refusal_class = None
    elif refusal_codes == {'CUSTOMER_DECLINED'}:
        refusal_class = 'customer_declined'
    elif 'CUSTOMER_DECLINED' in refusal_codes:
        refusal_class = 'mixed_refusal'
    else:
        refusal_class = 'rule_refusal'
    return {"category": classify_episode(episode), "tool_calls": len(episode.tool_executions),
            "refusal_class": refusal_class,
            "response_token_lengths": lengths}


def confirmation_order_compliance(episode):
    """Return whether a reply and Customer turn preceded the first state-changing tool."""
    reply_seen = customer_after_reply = False
    for event in episode.events:
        if event.kind == 'service_reply':
            reply_seen = True
        elif (event.kind == 'customer_transition' and reply_seen
              and event.payload.get('transition_kind') == 'decision'):
            customer_after_reply = True
        elif event.kind == 'service_tool' and event.before_state_hash != event.after_state_hash:
            return reply_seen and customer_after_reply
    return None


def summarize_capture_store(root):
    """Load and hash-check every episode in a capture store; fail on bad evidence."""
    from rift.verl.capture_store import NativeCaptureStore
    path = Path(root)
    if not path.is_dir():
        raise ValueError(f"capture store directory does not exist: {path}")
    store = NativeCaptureStore(path, serialization="diagnostic-read-only", attestor=None)
    counts = Counter()
    for item in sorted(path.glob("*.json")):
        episode = store.resolve_episode(item.stem)
        counts[classify_episode(episode)] += 1
    return dict(sorted(counts.items()))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize native capture-store episode outcomes")
    parser.add_argument("capture_store", help="directory containing native Episode capture JSON files")
    args = parser.parse_args(argv)
    counts = summarize_capture_store(args.capture_store)
    print("category\tcount")
    for category, count in counts.items():
        print(f"{category}\t{count}")
    print(f"total\t{sum(counts.values())}")


if __name__ == "__main__":
    main()
