import os
import subprocess
import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import json

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def get_function_header(prompt):
    splitted = prompt.split("\n")
    for line in splitted:
        if "def" in line:
            return line
    return ""

def get_function_name(line):
    line = line.strip()
    if not line.startswith("def "):
        return None
    parts = line.split()
    if len(parts) < 2:
        return None
    func_part = parts[1]
    func_name = func_part.split("(")[0]
    return func_name


def rebuild_func(asserts, func_name):
    ans = ""
    ans += f"def {func_name}():\n"
    for assertion in asserts:
        ans += f"\t{assertion}"
    return ans + "\n\n"

def format_response(response, testing_function, task_id, is_vanilla):
    lines = response.split("\n")
    curr_func = ""
    curr_asserts = []
    type_prompting = "vanilla" if is_vanilla else "crafted"
    formatted_response = (
        "import pytest\n"
        f"from {task_id}_{type_prompting} import {testing_function}\n\n"
    )
    for line in lines:
        func_name = get_function_name(line)
        if func_name is not None:
            if curr_func != "":
                pruned_asserts = prune_asserts(curr_asserts, testing_function)
                formatted_response += rebuild_func(pruned_asserts, curr_func)
            curr_func = func_name
            curr_asserts = []
        elif line.strip().startswith("assert"):
            curr_asserts.append(line.strip())
    if curr_func != "":
        pruned_asserts = prune_asserts(curr_asserts, testing_function)
        formatted_response += rebuild_func(pruned_asserts, curr_func)
    return formatted_response

def prune_asserts(asserts, function_name):
    pruned_asserts = []
    for assertion in asserts:
        hold = assertion.split("==")
        if len(hold) != 2:
            continue
        
        curr_assert, expected_val = hold
        curr_assert = curr_assert.strip()
        expected_val = expected_val.strip()
        
        # Check if curr_assert has the structure: assert function_name(...)
        if not curr_assert.startswith("assert"):
            continue
        
        after_assert = curr_assert[6:].strip()  # "assert" is 6 chars
        
        after_function = after_assert[len(function_name):].strip()
        if not after_function.startswith("("):
            continue
        
        if ")" not in after_function:
            continue
        
        closing_paren_idx = after_function.rfind(")")
        after_paren = after_function[closing_paren_idx + 1:].strip()
        if after_paren != "":
            continue

        num_quotes = expected_val.count("\"")
        if num_quotes % 2 != 0:
            continue

        pruned_asserts.append(assertion + "\n")
    
    if len(pruned_asserts) == 0:
        pruned_asserts.append("\n\tpass\n")
    
    return pruned_asserts

def create_prompt(entry, task_id, vanilla=True):
    """Create test-generation prompt for a given dataset entry."""
    base_prompt = (
        "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, "
        "and you only answer questions related to computer science. For politically sensitive questions, "
        "security and privacy issues, and other non-computer science questions, you will refuse to answer.\n\n"
        "### Instruction:\n"
        "Generate a pytest test suite for the following code.\n\n"
        "Only write unit tests in the output and nothing else.\n\n"
    )

    if not vanilla:
        base_prompt += (
            "\n**Test Requirements:**\n"
            "- DO NOT REWRITE THE FUNCTION THAT IS BEING TESTED; assume it exists and is imported.\n"
            "- Be mindful of runtime; avoid very large inputs or slow operations.\n"
            "- Write at least 5 focused tests that together cover **all branches, edge conditions, and error cases**.\n"
            "- Include representative inputs: normal, boundary, empty/zero, and invalid types if relevant.\n"
            "- Use precise `assert` statements verifying the **exact expected outputs** with double equal signs for equality checking.\n"
            "- Follow correct Python syntax and indentation.\n"
        )


    base_prompt += f"{entry['prompt'] + "\n" + entry['canonical_solution']}\n\n### Response:\n"
    return base_prompt

def split_asserts_into_tests(test_content: str) -> str:
    """
    Splits multi-assert test functions into multiple pytest test functions.
    Example:
        def test_foo():
            assert f(1) == 2
            assert f(2) == 3
    becomes:
        def test_foo_case_1():
            assert f(1) == 2

        def test_foo_case_2():
            assert f(2) == 3
    """
    lines = test_content.splitlines(keepends=True)
    output = []
    inside_func = False
    func_name = None
    asserts = []
    indent = "    "

    for line in lines:
        stripped = line.strip()

        # Detect the start of a test function
        if stripped.startswith("def test_"):
            inside_func = True
            func_name = re.match(r"def\s+(\w+)\s*\(", stripped).group(1)
            asserts = []
            continue  # don’t include the original def line

        # Detect end of function or blank line after function
        if inside_func and (not stripped or not line.startswith(indent)):
            # Emit multiple test_ functions for each assert
            for i, a in enumerate(asserts, 1):
                output.append(f"\ndef {func_name}_case_{i}():\n{indent}{a}\n")
            inside_func = False
            func_name = None
            asserts = []

        # Collect assert lines
        if inside_func and stripped.startswith("assert "):
            asserts.append(stripped)
            continue

        # Copy imports and everything outside test_ functions
        if not inside_func:
            output.append(line)

    # Handle case where file ends inside a test
    if inside_func and asserts:
        for i, a in enumerate(asserts, 1):
            output.append(f"\ndef {func_name}_case_{i}():\n{indent}{a}\n")

    return "".join(output)

def prompt_model(dataset, model_name = "deepseek-ai/deepseek-coder-6.7b-instruct", vanilla = True):
    print(f"Working with {model_name} prompt type {vanilla}...")
    
    # TODO: download the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # TODO: load the model with quantization
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

    for entry in dataset:
        task_id = entry["task_id"].replace("/", "_")
        code_file = f"{task_id}_{'vanilla' if vanilla else 'crafted'}.py"
        test_file = f"{task_id}_test_{'vanilla' if vanilla else 'crafted'}.py"
        coverage_file = f"Coverage/{task_id}_test_{'vanilla' if vanilla else 'crafted'}.json"
        full_code = entry['prompt'] + "\n" + entry['canonical_solution']
        save_file(full_code, code_file)

        function_header = get_function_header(entry["prompt"])
        function_name = get_function_name(function_header)

        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        prompt = create_prompt(entry, task_id, vanilla=vanilla)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # TODO: prompt the model and get the response
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = output_text.split("### Response:")[-1].strip()
        response = split_asserts_into_tests(response)

        response = format_response(response, function_name, task_id, vanilla)

        save_file(response, test_file)

        # TODO: process the response, generate coverage and save it to results
        try:
            result = subprocess.run(
                [
                    "pytest",
                    test_file,
                    f"--cov={task_id}_{'vanilla' if vanilla else 'crafted'}",
                    f"--cov-report=json:{coverage_file}"
                ],
                capture_output=True,
                text=True,
            )

            print(result.stdout)
            print(result.stderr)

            # Parse coverage JSON if available
            coverage = None
            if os.path.exists(coverage_file):
                with open(coverage_file, "r") as f:
                    cov_json = json.load(f)
                    coverage = cov_json["totals"]["percent_covered"]

        except subprocess.TimeoutExpired:
            coverage = "TimeoutError: pytest execution took too long."
        except Exception as e:
            coverage = f"Error running pytest: {e}"

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\ncoverage:\n{coverage}")
        results.append({
            "task_id": entry["task_id"],
            "prompt": prompt,
            "response": response,
            "coverage": coverage
        })
        
    return results

def read_jsonl(file_path):
    dataset = []
    with jsonlines.open(file_path) as reader:
        for line in reader: 
            dataset.append(line)
    return dataset

def write_jsonl(results, file_path):
    with jsonlines.open(file_path, "w") as f:
        for item in results:
            f.write_all([item])

if __name__ == "__main__":
    """
    This Python script is to run prompt LLMs for code synthesis.
    Usage:
    `python3 task_2.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

    Inputs:
    - <input_dataset>: A `.jsonl` file, which should be your team's dataset containing 20 HumanEval problems.
    - <model>: Specify the model to use. Options are "deepseek-ai/deepseek-coder-6.7b-base" or "deepseek-ai/deepseek-coder-6.7b-instruct".
    - <output_file>: A `.jsonl` file where the results will be saved.
    - <if_vanilla>: Set to 'True' or 'False' to enable vanilla prompt
    
    Outputs:
    - You can check <output_file> for detailed information.
    """
    args = sys.argv[1:]
    input_dataset = args[0]
    model = args[1]
    output_file = args[2]
    if_vanilla = args[3] # True or False
    
    if not input_dataset.endswith(".jsonl"):
        raise ValueError(f"{input_dataset} should be a `.jsonl` file!")
    
    if not output_file.endswith(".jsonl"):
        raise ValueError(f"{output_file} should be a `.jsonl` file!")
    
    vanilla = True if if_vanilla == "True" else False
    
    dataset = read_jsonl(input_dataset)
    results = prompt_model(dataset, model, vanilla)
    write_jsonl(results, output_file)
