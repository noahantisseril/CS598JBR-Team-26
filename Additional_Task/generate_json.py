import json
import os
import re

def load_reproduction_log(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"([^:]+):\s*\[(.*?)\]", line)
            if m:
                pid = m.group(1)
                indices = m.group(2).strip()
                if indices:
                    indices = [int(x.strip()) for x in indices.split(",")]
                else:
                    indices = []
                data[pid] = indices
    return data

def load_search_log(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"([^:]+):\s*\[(.*?)\]", line)
            if m:
                pid = m.group(1)
                indices = m.group(2).strip()
                if indices:
                    indices = [int(x.strip()) for x in indices.split(",")]
                else:
                    indices = []
                data[pid] = indices
    return data


def load_tool_use_log(path):
    with open(path, "r") as f:
        return json.load(f)

def generate_json(pid, repro, search, tools):
    did_generate_repro = len(repro) > 0
    did_search = len(search) > 0

    return {
        "Traj ID": pid.split("@")[1],
        "Issue Summary": "",
        "Interaction Summary": "",

        "Reproduction Code": "",
        "1.1": "YES" if did_generate_repro else "NO",
        "1.2": "",
        
        "Search for the issue": "" if did_search else "None",
        "2.1": "YES" if did_search else "NO",
        "2.2": "",

        "Edit the Code": "",

        "Test changes on the reproduction code": "",
        "4.1": "",
        "4.2": "",

        "Tool-use analysis": tools if tools else {}
    }


def main():
    reproduction = load_reproduction_log("locate_reproduction_code.log")
    search = load_search_log("locate_search.log")
    tool_use = load_tool_use_log("locate_tool_use.log")

    all_pids = reproduction.keys()
    
    for pid in all_pids:
        repro_indices = reproduction.get(pid, [])
        search_indices = search.get(pid, [])
        tools = tool_use.get(pid, {})

        entry = generate_json(pid, repro_indices, search_indices, tools)
        pid = pid.split("@")[1]

        outpath = f"{pid}.json"
        with open(outpath, "w") as f:
            json.dump(entry, f, indent=2)

        print(f"Created {outpath}")


if __name__ == "__main__":
    main()
