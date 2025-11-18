import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import tempfile
import subprocess
import os

#####################################################
# Please finish all TODOs in this file for MP3/task_1;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def match_java_code(response):
    pattern = r"\[Java Start\](.*?)\[Java End\]"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    no_backticks = r"```(?:java)?(.*?)```"
    match = re.search(no_backticks, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    if "[Java Start]" in response:
        return response.split("[Java Start]")[1].strip()
    if "[Java End]" in response:
        return response.split("[Java End]")[0].strip()

    return response.strip()


def final_clean_code(java_code):
    cleaned_lines = []
    for line in java_code.splitlines():
        line = line.strip()
        if line.startswith("import") or line.startswith("package"):
            continue
        if line.startswith("public class Main") or line.startswith("class Main"):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def wrap_in_solution_class(java_impl_code):
    if "class Solution" in java_impl_code:
        return java_impl_code
    return "class Solution {\n" + java_impl_code + "\n}\n"

def find_imports(declaration):
    imports = []
    for line in declaration.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import "):
            imports.append(stripped)
    return "\n".join(imports)

def execute_java(java_test_code, java_impl_code, declaration):
    imports = find_imports(declaration)
    solution_code = wrap_in_solution_class(java_impl_code)

    full_java = imports + "\n" + java_test_code.rstrip() + "\n" + solution_code + "\n"

    tmpdir = tempfile.mkdtemp(prefix="mp3_task1_")
    main_path = os.path.join(tmpdir, "Main.java")
    save_file(full_java, main_path)

    compile_proc = subprocess.run(
        ["javac", "Main.java"],
        cwd=tmpdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
    if compile_proc.returncode != 0:
        print("compilation error")
        return False

    run_proc = subprocess.run(
        ["java", "Main"],
        cwd=tmpdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
    if run_proc.returncode != 0:
        return False

    return True

def get_vanilla_prompt(python_code):
    return f"""You are an AI programming assistant utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

    ### Instruction:
    Can you translate the following Python code into Java?
    The new Java code must be enclosed between [Java Start] and [Java End]

    {python_code}

    ### Response:
    [Java Start]
    """

def get_crafted_prompt(python_code):
    return f"""You are an AI programming assistant utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

    ### Instruction:
    Can you translate the following Python code into Java?
    The new Java code must be enclosed between [Java Start] and [Java End].
    Do NOT include markdown ``` code fences in the Java output, only plain Java code.
    Do NOT write a main method.
    Do NOT write any import statements.
    Only write the method implementation(s) needed for the translation.

    {python_code}

    ### Response:
    [Java Start]
    """

def prompt_model(
    python_dataset,
    java_dataset,
    model_name: str = "deepseek-ai/deepseek-coder-6.7b-instruct",
    vanilla: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    results = []
    for entry in python_dataset:
        java_task_id = entry["task_id"]
        java_entry = java_dataset[java_task_id]
        java_test_code = java_entry["test"]

        python_code = entry["declaration"] + "\n" + entry["canonical_solution"]

        if vanilla:
            prompt = get_vanilla_prompt(python_code)
        else:
            prompt = get_crafted_prompt(python_code)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        raw_java = match_java_code(response)
        java_code = final_clean_code(raw_java)
        declaration = java_entry["declaration"]

        verdict = execute_java(java_test_code, java_code, declaration)

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_expected:\n{verdict}")
        results.append({
            "task_id": java_task_id,
            "prompt": prompt,
            "response": response,
            "is_correct": verdict
        })
    return results

def read_jsonl(file_path):
    dataset = []
    with jsonlines.open(file_path) as reader:
        for line in reader:
            dataset.append(line)
    return dataset

def read_jsonl_as_dict(file_path):
    dataset = {}
    with jsonlines.open(file_path) as reader:
        for line in reader:
            dataset[line["task_id"]] = line
    return dataset

def write_jsonl(results, file_path):
    with jsonlines.open(file_path, "w") as f:
        for item in results:
            f.write_all([item])

if __name__ == "__main__":
    """
    This Python script is to run prompt LLMs for code translation.
    Usage:
    `python3 task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

    Inputs:
    - <input_dataset>: A `.jsonl` file, which should be your team's dataset containing 20 HumanEval problems.
    - <model>: Specify the model to use. Options are "deepseek-ai/deepseek-coder-6.7b-base" or "deepseek-ai/deepseek-coder-6.7b-instruct".
    - <output_file>: A `.jsonl` file where the results will be saved.
    - <if_vanilla>: Set to 'True' or 'False' to enable vanilla prompt
    
    Outputs:
    - You can check <output_file> for detailed information.
    """
    args = sys.argv[1:]
    input_python_dataset = args[0]
    model = args[1]
    output_file = args[2]
    if_vanilla = args[3] # True or False
    if not input_python_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_python_dataset} should be a `.jsonl` file!")
    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")
    vanilla = True if if_vanilla == "True" else False

    input_java_dataset = "selected_humanevalx_java_82823851747266875283573544649953054055.jsonl"
    python_dataset = read_jsonl(input_python_dataset)
    java_dataset = read_jsonl_as_dict(input_java_dataset)
    results = prompt_model(python_dataset, java_dataset, model, vanilla)
    write_jsonl(results, output_file)