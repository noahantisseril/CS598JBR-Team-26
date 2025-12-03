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
    steps = load_trajectory_file(instance_id)
    if not steps:
        return {}

    # count tool usage
    tool_counts = {}
    for step in steps:
        action = step.get("action", None).split()
        possible_args = ["view", "create", "str_replace", "insert", "undo_edit"]
        bash_args = ["find", "grep", "cat", "ls", "cd"]
        use_multi = False
        for word in action:
            if word == "str_replace_editor":
                use_multi = True
                continue
            if use_multi and word in possible_args:
                tool_counts[word] = tool_counts.get(word, 0) + 1
            if word in bash_args + ["submit"]:
                tool_counts[word] = tool_counts.get(word, 0) + 1
            use_multi = False
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

    tool_lines = []
    for instance in instance_dirs:
        instance_dict = {"instance" : instance}
        steps = locate_tool_use(instance)
        instance_dict.update(steps)
        tool_lines.append(instance_dict)
    with open("locate_tool_use.log", "w", encoding="utf-8") as fh:
        json.dump(tool_lines, fh, indent=2)

if __name__ == "__main__":
    main()
