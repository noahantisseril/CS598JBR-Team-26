#!/usr/bin/env python3
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
    # explicit SWE-Agent commands
    r"\bfind_file\b", r"\bsearch_file\b", r"\bsearch_dir\b",
    # common shell/file commands
    r"\bfind\b", r"\bgrep\b", r"\bls\b", r"\bcd\b", r"\bcat\b",
    r"\brg\b", r"\bf\bdag\b", r"\back\b", r"\bag\b", r"\bfd\b",
    # SWE-Agent editor interactions that list/view files
    r"str_replace_editor\s+(?:view|list|create|open|replace)\b",
    # 'open file', 'view file', 'show file'
    r"\bopen\b", r"\bview\b", r"\bshow\b", r"\bread\b", r"\blist\b",
    # search words
    r"\bsearch\b", r"\bfind\b", r"\blook for\b", r"\bnavigate\b", r"\bexplore\b"
]

# join into compiled regex
ACTION_RE = re.compile("|".join(ACTION_PATTERNS), re.IGNORECASE)

def step_contains_search(step):
    """
    Heuristic: examine fields in step dict (action, observation, thought, query, response)
    and return True if it looks like a search/navigation step.
    """
    if not isinstance(step, dict):
        return False

    text_fields = []
    # gather textual fields safely
    for key in ("action", "observation", "thought", "response"):
        val = step.get(key)
        if isinstance(val, str):
            text_fields.append(val)
            
    # some trajectories embed a 'query' list (like openai messages)
    if "query" in step and isinstance(step["query"], list):
        for q in step["query"]:
            if isinstance(q, dict):
                # look at content strings inside messages
                for subk in ("content", "text"):
                    if subk in q and isinstance(q[subk], str):
                        text_fields.append(q[subk])
                    elif subk in q and isinstance(q[subk], list):
                        for item in q[subk]:
                            if "text" in item and isinstance(item["text"], str):
                                text_fields.append(item["text"])
            elif isinstance(q, str):
                text_fields.append(q)

    print("text fields", text_fields)
    # build one combined text
    combined = "\n".join(text_fields).lower()

    if not combined.strip():
        return False

    # print("combined", combined)

    # If action field explicitly matches our regex, consider it a search
    action_val = step.get("action")
    if isinstance(action_val, str) and ACTION_RE.search(action_val):
        return True

    # Check observation / thought / combined for commands or keywords complete words
    if ACTION_RE.search(combined):
        return True

    # Additional heuristics:
    # - observations that list directories (the example had '/testbed\n/testbed/...') -> look for many leading slashes or file paths
    # - presence of "Here's the files and directories" or lines containing "/"
    if "here's the files" in combined or "here's the files and directories" in combined:
        return True
    # if lots of '/testbed' like paths show up, it's probably a search/list
    path_like_tokens = re.findall(r"/[A-Za-z0-9_\-./]+", combined)
    if len(path_like_tokens) >= 3:
        return True

    # grep-like output: lines containing ':' after a filename or 'matches' etc.
    if re.search(r"^[^\n]+:\d+:", combined, re.M):
        return True

    return False


def locate_search(instance_id):
    agent_name, problem_name = instance_id.split("@")
    instance_dir = ROOT / instance_id
    traj_file = instance_dir / f"{problem_name}.traj"
    traj_steps = load_trajectory_file(traj_file)
    
    if traj_steps is None:
        print(f"[WARN] No trajectory file parsed for {instance_id}")
        return []

    results = []

    # iterate steps and collect indices
    for idx, step in enumerate(traj_steps):
        if step_contains_search(step):
            results.append(idx)

    return results


def locate_tool_use(instance_id: str) -> Dict[str, int]:
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
    if not ROOT.exists():
        print(f"[ERROR] Root extracted directory not found: {ROOT}")
        return

    reproduction_lines = []
    instance_dirs = sorted([p.name for p in ROOT.iterdir() if p.is_dir()])

    for instance in instance_dirs:
        steps = locate_reproduction_code(instance)
        reproduction_lines.append(f"{instance}: {steps}")
        print(f"[OK] {instance}: {steps}")

    # write a logfile
    with open("locate_reproduction_code.log", "w", encoding="utf-8") as fh:
        fh.write("\n".join(reproduction_lines))

    print(f"\nWrote results to locate_reproduction_code.log")
    
    search_lines = []
    for instance in instance_dirs:
        steps = locate_search(instance)
        search_lines.append(f"{instance}: {steps}")
        print(f"[OK] {instance}: {steps}")
    # write a logfile
    with open("locate_search.log", "w", encoding="utf-8") as fh:
        fh.write("\n".join(search_lines))
        
    print(f"\nWrote results to locate_search.log")

if __name__ == "__main__":
    main()
