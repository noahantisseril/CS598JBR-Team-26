import json
import os
import re
from pathlib import Path
from typing import List

ROOT = Path("extracted_trajectories")

def load_trajectory_file(instance_id):
    agent_name, problem_name = instance_id.split("@")
    instance_dir = ROOT / instance_id
    traj_file = instance_dir / f"{problem_name}.traj"
    text = traj_file.read_text(encoding="utf-8", errors="ignore").strip()
    obj = json.loads(text)
    return obj["trajectory"]

reproduction_keywords = ["reproduce", "reproduction", "debug", "debugging"]

def contains_words_in_list(text):
    t = (text or "").lower()
    return any(k in t for k in reproduction_keywords)
    
def extract_filename(action):
    pattern = r"create\s+([^\s]+)\s+--file_text"
    match = re.search(pattern, action)

    if match:
        return match.group(1)
    return None

def locate_reproduction_code(instance_id):
    traj_steps = load_trajectory_file(instance_id)
    reproduction_steps = []
    for idx, step in enumerate(traj_steps):
        thought = step["thought"]
        action = step["action"]

        action_create_flag = "create" in str(action).lower()
        filename = extract_filename(action)
        if action_create_flag and filename:
            if contains_words_in_list(filename) or contains_words_in_list(thought):
                reproduction_steps.append(idx)

    return reproduction_steps

    
search_keywords = [
    "find", "grep", "ls", "cd", "cat", "str_replace_editor view", 
]

def locate_search(instance_id):
    traj_steps = load_trajectory_file(instance_id)

    results = []

    for idx, step in enumerate(traj_steps):
        action_vals = step["action"].lower().strip().split(" ")
        use_multi = False
        for action_val in action_vals:
            if action_val == "str_replace_editor":
                use_multi = True
                continue
            if use_multi and action_val == "view":
                results.append(idx)
                break
            else:
                use_multi = False
                
            if any(k == action_val for k in search_keywords):
                results.append(idx)
                break

    return results

def locate_tool_use(instance_id):
    traj_steps = load_trajectory_file(instance_id)
    tool_counts = {}

    for step in steps:
        query = step.get("query", None)

        messages = step.get("messages", None)
        possible_args = ["view", "create", "str_replace", "insert", "undo_edit"]
        bash_args = ["find", "grep", "cat", "ls", "cd"]
        possible_args.extend(bash_args)
        # models have diff format
        iterable_convo = None
        if query:
            iterable_convo = query
        elif messages:
            iterable_convo = messages
    for step in traj_steps:
        for info in iterable_convo:
            tool_calls_list = info.get("tool_calls", None)
            if not tool_calls_list:
                continue
            for tool_call_dict in tool_calls_list:
                if not tool_call_dict:
                    continue
                tool_name = tool_call_dict["function"]["name"]
                # if no found arguments, just put the tool name
                if tool_call_dict["function"]["arguments"] == "{}":
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                else:
                    # otherwise put in the args
                    for tool in possible_args:
                        if tool in tool_call_dict["function"]["arguments"]:
                            tool_counts[tool] = tool_counts.get(tool, 0) + 1
    return tool_counts

def main():
    reproduction_lines = []
    instance_dirs = sorted([p.name for p in ROOT.iterdir() if p.is_dir()])

    for instance in instance_dirs:
        steps = locate_reproduction_code(instance)
        reproduction_lines.append(f"{instance}: {steps}")

    with open("locate_reproduction_code.log", "w", encoding="utf-8") as f:
        f.write("\n".join(reproduction_lines))


    search_lines = []
    for instance in instance_dirs:
        steps = locate_search(instance)
        search_lines.append(f"{instance}: {steps}")

    with open("locate_search.log", "w", encoding="utf-8") as f:
        f.write("\n".join(search_lines))

if __name__ == "__main__":
    main()
