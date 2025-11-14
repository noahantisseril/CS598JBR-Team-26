import jsonlines
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re

#####################################################
# Please finish all TODOs in this file for MP3/task_2;
#####################################################

def save_file(content, file_path):
    with open(file_path, 'w') as file:
        file.write(content)

def get_prompt(code, vanilla):
    if vanilla:
        return f"""You are an AI programming assistant utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

            ### Instruction:

            {code}

            Is the above code buggy or correct? Please explain your step by step reasoning. 
            The prediction should be enclosed within <start> and <end> tags. For example: <start>Buggy<end>

            ### Response:
            """
    else:
        return f"""You are an AI programming assistant utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.

            ### Instruction:

            You are given a Python function. 
            Determine whether the implementation is **logically correct** according to standard intended behavior, unless a docstring states otherwise.

            For this task, a function is **buggy** if ANY of the following are true:

            - It contains incorrect logic.
            - It incorrectly compares values (e.g., missing abs(), wrong inequality).
            - It returns the wrong result for any input.
            - It has mismatched behavior vs. typical expectations for the problem.

            Otherwise, it is **correct**.

            You MUST output one of the following two tokens only:
            - <start>Correct<end>
            - <start>Buggy<end>

            ### Additional Requirements
            - Before producing the final tokens, think silently and DO NOT reveal chain-of-thought.
            - Your visible explanation must be **brief**, high-level, and must NOT show intermediate reasoning.
            - Do NOT output anything outside the <start> and <end> tokens for the final answer.

            ### Question

            {code}

            Is the above code buggy or correct? Please explain your step by step reasoning. 
            The prediction should be enclosed within <start> and <end> tags. For example: <start>Buggy<end>

            ### Response:
            """
    
def parse_response(response):
    pattern = r'\<start\>(.*?)\<end\>'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    if match:
        verdict = match.group(1).strip().lower()
        if verdict == "buggy":
            return True
        elif verdict == "correct":
            return False
    return False

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
        code = entry['declaration'] + '\n' + entry['canonical_solution']
        # TODO: create prompt for the model
        # Tip : Use can use any data from the dataset to create 
        #       the prompt including prompt, canonical_solution, test, etc.
        prompt = get_prompt(code, vanilla)
        
        # TODO: prompt the model and get the response
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

        # TODO: process the response and save it to results
        verdict = parse_response(response)

        print(f"Task_ID {entry['task_id']}:\nprompt:\n{prompt}\nresponse:\n{response}\nis_expected:\n{verdict}")
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
    This Python script is to run prompt LLMs for bug detection.
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
