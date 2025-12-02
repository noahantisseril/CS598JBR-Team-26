import json
import os
import re
from pathlib import Path
from typing import List

ROOT = Path("extracted_trajectories")

def load_trajectory_file(traj_path):
    text = traj_path.read_text(encoding="utf-8", errors="ignore").strip()
    obj = json.loads(text)
    return obj["trajectory"]

reproduction_keywords = ["repro", "reproduce", "reproduction", "debug", "debugging"]

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
    agent_name, problem_name = instance_id.split("@")
    instance_dir = ROOT / instance_id
    traj_file = instance_dir / f"{problem_name}.traj"
    traj_steps = load_trajectory_file(traj_file)
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

    
ACTION_PATTERNS = [
    r"\bfind\b", r"\bgrep\b", r"\bls\b", r"\bcd\b", r"\bcat\b",
]

ACTION_RE = re.compile("|".join(ACTION_PATTERNS), re.IGNORECASE)

def step_contains_search(step):
    if not isinstance(step, dict):
        return False
            
    action_val = step.get("action").lower()

    if not action_val.strip():
        return False

    return isinstance(action_val, str) and ACTION_RE.search(action_val)

def locate_search(instance_id):
    agent_name, problem_name = instance_id.split("@")
    instance_dir = ROOT / instance_id
    traj_file = instance_dir / f"{problem_name}.traj"
    traj_steps = load_trajectory_file(traj_file)

    results = []

    for idx, step in enumerate(traj_steps):
        if step_contains_search(step):
            results.append(idx)

    return results


def locate_tool_use(instance_id: str):
    # get instance ID
    agent_name, problem_name = instance_id.split("@")
    instance_dir = ROOT / instance_id
    traj_file = instance_dir / f"{problem_name}.traj"
    
    # get trajectory
    steps = load_trajectory_file(traj_file)
    if not steps:
        return {}
    
    # count tool usage
    tool_counts = {}
    
    for step in steps:
        query = step.get("query", None)

        messages = step.get("messages", None)
        # models have diff format
        iterable_convo = None
        if query:
            iterable_convo = query
        elif messages:
            iterable_convo = messages
        for info in iterable_convo:
            tool_calls_list = info["tool_calls"] if "tool_calls" in info else None
            if not tool_calls_list:
                continue
            for tool_call_dict in tool_calls_list:
                if not tool_call_dict:
                    continue
                tool_counts[tool_call_dict["function"]["name"]] = tool_counts.get(tool_call_dict["function"]["name"], 0) + 1
    return tool_counts

def main():
    reproduction_lines = []
    instance_dirs = sorted([p.name for p in ROOT.iterdir() if p.is_dir()])

    for instance in instance_dirs:
        steps = locate_reproduction_code(instance)
        reproduction_lines.append(f"{instance}: {steps}")
        print(f"{instance}: {steps}")

    with open("locate_reproduction_code.log", "w", encoding="utf-8") as fh:
        fh.write("\n".join(reproduction_lines))

    
    search_lines = []
    for instance in instance_dirs:
        steps = locate_search(instance)
        search_lines.append(f"{instance}: {steps}")
    with open("locate_search.log", "w", encoding="utf-8") as fh:
        fh.write("\n".join(search_lines))

if __name__ == "__main__":
    main()
