import jsonlines
import sys
import torch
import re
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import ast

#####################################################
# Please finish all TODOs in this file for MP2;
#####################################################

random.seed(1)

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

# Extract first test case from the test string
def extract_test_case(test_string):
    lines = test_string.strip().split('\n')
    random.shuffle(lines)
    output = []
    for line in lines:
        if 'assert candidate' in line or 'assert ' in line:
            match = re.search(r'assert\s+(?:candidate)?\s*\((.*?)\)\s*==\s*(.+)', line)
            if match:
                input_part = match.group(1).strip()
                rest = match.group(2).strip()
                
                # Extract until comma outside brackets
                expected_output = extract_until_comma_outside_brackets(rest)
                output.append((input_part, expected_output))
    return output

def extract_until_comma_outside_brackets(text):
    bracket_depth = 0
    result = []
    
    for char in text:
        if char == '[':
            bracket_depth += 1
            result.append(char)
        elif char == ']':
            bracket_depth -= 1
            result.append(char)
        elif char == ',' and bracket_depth == 0:
            break
        else:
            result.append(char)
    
    return ''.join(result).strip()

def smart_parse(s):
    if s is None:
        return None
    s = s.strip()
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError):
        try:
            return ast.literal_eval(f'"{s}"')
        except:
            return s

# Extract the output from between the [Output] [/Output] tags
def extract_output_from_response(response):
    match = re.search(r'\[Output\](.*?)\[/?Output\]', response, re.IGNORECASE | re.DOTALL)
    if match:
        ret_string = match.group(1).strip()
        if ret_string and (ret_string[0] == "(" and ret_string[-1] == ")"):
            return ret_string[1:-1]
        return ret_string
    return None

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
        testcases = extract_test_case(entry['test'])
        test_input, expected_output = "1", "2" # dont let them match by default
        if testcases:
            test_input, expected_output = testcases[0]
            testcases.pop(0)
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        prompt = f"""You are an AI programming assistant. You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

### Instruction:
If the input is {test_input}, what will the following code return?
The return value prediction must be enclosed between [Output] and [/Output] tags. For example : [Output]prediction[/Output]

{entry['prompt']}
{entry['canonical_solution']}

### Response:
"""
        
        if not vanilla:
            prompt = f"""
### Instruction:
Determine the return value of the given Python function for the given input.

The return value prediction MUST be enclosed between [Output] and [/Output] tags.

Here is a sample output: [Output]<return value>[/Output]

Follow the steps outlined in the thought process section.

Thought process:
- Trace the executed path step-by-step (functions called, loops, branches, early returns).
- Compute all intermediate values exactly using **Python semantics** (/, //, %, **, slicing).
- When comparing numbers, perform an explicit numeric check (no heuristics, no string/lexicographic logic). 
- Cross-check with 1-2 nearby sanity cases from the provided tests to confirm **type and format**.
- If any check contradicts earlier steps, re-trace and fix before answering.
- Once a final value is found, check if the value makes logical sense given the input and problem statement
- Output only the final value in the required tags.

Output format (STRICT):
- ONLY use canonical Python literals for return values (e.g., True/False, Lists, None, double quotes for strings).
- The output MUST be between [Output] and [/Output] tags
- Do NOT include backticks or single quotes.
- Infer the return type/format using the examples provided

Clearly and explicitly outline your thought process, adhering to the above guidelines.

Code:

{entry['prompt']}
{entry['canonical_solution']}

Example testcases:
"""
            for inp, outp in testcases:
                prompt += f"\nInput: {inp}, Output: [Output]{outp}[/Output]\n"

            prompt += f"""
Now determine the output for this input:
{test_input}

### Response:
"""
        
        # TODO: prompt the model and get the response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # TODO: process the response and save it to results
        predicted_output = extract_output_from_response(response)
        # We used ast literal eval to handle cases with differing whitespace
        # and formatting for python literals.
        # With this approach, the string is parsed to a python literal before comparison.
        # This prevents slight string inconsistencies from changing answer correctness
        verdict = smart_parse(predicted_output) == smart_parse(expected_output)
        # verdict = ast.literal_eval(predicted_output) == ast.literal_eval(expected_output)
        if not verdict:
            print("Mismatch:\n", predicted_output, expected_output)

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_correct:\n{verdict}")
        results.append({
            "task_id": entry["task_id"],
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

def write_jsonl(results, file_path):
    with jsonlines.open(file_path, "w") as f:
        for item in results:
            f.write_all([item])

if __name__ == "__main__":
    """
    This Python script is to run prompt LLMs for code synthesis.
    Usage:
    `python3 Task_1.py <input_dataset> <model> <output_file> <if_vanilla>`|& tee prompt.log

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
