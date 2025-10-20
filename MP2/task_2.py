import os
import subprocess
import jsonlines
import sys
import torch
import ast
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def create_prompt(entry, task_id, vanilla=True):
    """Create test-generation prompt for a given dataset entry."""
    base_prompt = (
        "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, "
        "and you only answer questions related to computer science. For politically sensitive questions, "
        "security and privacy issues, and other non-computer science questions, you will refuse to answer.\n\n"
        "### Instruction:\n"
        "Generate a pytest test suite for the following code.\n\n"
        "Only write unit tests in the output and nothing else."
        f"Import the function from the module `{task_id}_{'vanilla' if vanilla else 'crafted'}.py` instead of redefining it.\n\n"
    )

    if not vanilla:
        base_prompt += (
            "The input code includes the full function definition (signature + body).\n"
            "Only write unit tests. Do not include explanations, comments, or extra text.\n"
            "\n### Test Requirements:\n"
            "- Write multiple test functions (e.g., `def test_case_1():`, `def test_case_2():`).\n"
            "- Cover at least 5 cases including typical inputs, edge cases, empty inputs, negative inputs, and unusual inputs.\n"
            "- Ensure high branch and line coverage; include all execution paths.\n"
            "- Check for no syntax errors. Think about the cases step by step.\n"
        )


    base_prompt += f"{entry['prompt'] + "\n" + entry['canonical_solution']}\n\n### Response:\n"
    return base_prompt

def remove_unterminated_last_line(text):
    lines = text.rstrip().splitlines()
    if not lines:
        return text

    last_line = lines[-1]

    # Check for unterminated syntax characters
    openers = "([{"
    closers = ")]}"
    quotes = "\"'"

    # Basic syntax balance check
    if (
        any(ch in last_line for ch in openers)
        and not any(ch in last_line for ch in closers)
    ) or (last_line.count('"') % 2 == 1) or (last_line.count("'") % 2 == 1):
        # Remove last line if it looks incomplete
        lines = lines[:-1]

    return "\n".join(lines) + "\n"

def drop_invalid_tests(test_content):
    """
    Cleans a test file string by:
      - Removing everything before the first `import pytest`
      - Keeping only valid import statements and valid test_ functions
      - Skipping malformed or incomplete test functions
    """
    # Keep only code starting at first import
    match = re.search(r"import pytest", test_content)
    if match:
        test_content = test_content[match.start():]

    lines = test_content.splitlines(keepends=True)
    valid_parts = []

    # Keep top-level imports
    for line in lines:
        if line.strip().startswith("import") or line.strip().startswith("from"):
            valid_parts.append(line)

    # Split roughly by test function definitions
    func_chunks = re.split(r"(?=^def test_)", test_content, flags=re.MULTILINE)

    for chunk in func_chunks:
        chunk = chunk.strip()
        if not chunk.startswith("def test_"):
            continue
        # Ensure it has some indentation after the def line
        if not re.search(r"\n\s+", chunk):
            # e.g., def test_case_18(): with no body
            continue
        try:
            ast.parse(chunk)
            valid_parts.append("\n" + chunk + "\n")
        except (SyntaxError, IndentationError):
            continue  # skip broken ones

    return "".join(valid_parts)

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
                top_p=0.95
            )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = output_text.split("### Response:")[-1].strip()

        # Strip ```python or ``` if they exist
        if response.startswith("```python"):
            response = response[len("```python"):].strip()
        if response.startswith("```"):
            response = response[len("```"):].strip()
        if response.endswith("```"):
            response = response[:-len("```")].strip()

        # Find the first occurrence of 'import pytest'
        idx = response.find("import pytest")
        if idx != -1:
            response = response[idx:]  # Keep everything from 'import pytest' onwards

        idx = response.find("```")
        if idx != -1:
            response = response[:idx].strip()

        response = drop_invalid_tests(remove_unterminated_last_line(response))

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
                # timeout=60
            )

            print(result.stdout)
            print(result.stderr)

            # Parse coverage JSON if available
            coverage = None
            if os.path.exists(coverage_file):
                with open(coverage_file, "r") as f:
                    cov_json = f.read()
                    coverage = cov_json

        except subprocess.TimeoutExpired:
            coverage = "TimeoutError: pytest execution took too long."
        except Exception as e:
            coverage = f"Error running pytest: {e}"

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\ncoverage:\n{coverage if isinstance(coverage, str) else 'OK'}")
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
