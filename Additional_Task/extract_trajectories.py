import os
import shutil
from pathlib import Path

TRAJECTORY_IDS = [
    "20250522_sweagent_claude-4-sonnet-20250514@django__django-14122",
    "20250522_sweagent_claude-4-sonnet-20250514@django__django-11477",
    "20250522_sweagent_claude-4-sonnet-20250514@astropy__astropy-14182",
    "20250522_sweagent_claude-4-sonnet-20250514@matplotlib__matplotlib-25287",
    "20250522_sweagent_claude-4-sonnet-20250514@sympy__sympy-18199",
    "20250522_sweagent_claude-4-sonnet-20250514@matplotlib__matplotlib-24149",
    "20250522_sweagent_claude-4-sonnet-20250514@django__django-15098",
    "20250522_sweagent_claude-4-sonnet-20250514@django__django-15732",
    "20250522_sweagent_claude-4-sonnet-20250514@astropy__astropy-7606",
    "20250522_sweagent_claude-4-sonnet-20250514@astropy__astropy-13033",
    "20250522_sweagent_claude-4-sonnet-20250514@django__django-12325",
    "20250522_sweagent_claude-4-sonnet-20250514@matplotlib__matplotlib-22871",
    "20250522_sweagent_claude-4-sonnet-20250514@sphinx-doc__sphinx-10435",
    "20250522_sweagent_claude-4-sonnet-20250514@sympy__sympy-19040",
    "20250522_sweagent_claude-4-sonnet-20250514@django__django-14155",
    "20250522_sweagent_claude-4-sonnet-20250514@sympy__sympy-12489",
    "20250522_sweagent_claude-4-sonnet-20250514@psf__requests-5414",
    "20250511_sweagent_lm_32b@django__django-12308",
    "20250511_sweagent_lm_32b@django__django-12774",
    "20250511_sweagent_lm_32b@pylint-dev__pylint-8898"
]

OUTPUT_DIR = Path("extracted_trajectories")
ROOT_DIR = Path(".")

def extract_one_traj(full_id):
    agent, problem = full_id.split("@")
    src_path = ROOT_DIR / agent / "trajs" / problem

    if not src_path.exists():
        print(f"[WARN] Missing: {src_path}")
        return False

    dst_path = OUTPUT_DIR / full_id
    dst_path.mkdir(parents=True, exist_ok=True)

    # Copy full directory tree
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

    print(f"[OK] Extracted {full_id} -> {dst_path}")
    return True

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    count = 0
    for traj_id in TRAJECTORY_IDS:
        if extract_one_traj(traj_id):
            count += 1

    print(f"\nDone. Extracted {count}/{len(TRAJECTORY_IDS)} trajectories.")


if __name__ == "__main__":
    main()
