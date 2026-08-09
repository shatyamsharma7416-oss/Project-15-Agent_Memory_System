import json
from tools.facts_retrieve import retrieve_facts

TOOL_REGISTRY = {
    "retrieve_facts": {
        "func": retrieve_facts,
    }
    # "create_event": {"func": calendar_tools.create_event, "reversibility": "reversible"},
}

def select_service(function_call_part):
    tool_name = function_call_part.name
    args_dict = json.loads(function_call_part.arguments)

    entry = TOOL_REGISTRY.get(tool_name)
    if entry is None:
        return f"Error: unknown tool '{tool_name}'"

    return entry["func"](**args_dict)

