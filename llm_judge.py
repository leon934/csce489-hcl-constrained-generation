import subprocess
import shutil
from pathlib import Path
import json
import argparse

from datasets import load_dataset

from tqdm import tqdm

def ask_qwen_cli(prompt):
    """
    Wraps the 'qwen chat' CLI command to use it from Python.
    """

    if not shutil.which("qwen"):
        raise EnvironmentError("Qwen CLI not found. Run 'npm install -g @qwen-code/qwen-code' first.")

    try:
        result = subprocess.run(
            ["qwen", "chat", prompt],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        
        return result.stdout.strip()

    # handle err
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"

def main():
    parser = argparse.ArgumentParser(description="Compares two sets of Terraform outputs using an LLM Judge (Qwen) based on syntax, security, and isolation.")
    
    parser.add_argument(
      '--tf_no_grammar', 
      type=str, 
      required=True, 
      help="Path to the directory containing the baseline Terraform files (generated WITHOUT grammar constraints)."
    )
    parser.add_argument(
      '--tf_with_grammar', 
      type=str, 
      required=True, 
      help="Path to the directory containing the experimental Terraform files (generated WITH grammar constraints)."
    )
    parser.add_argument(
      '--eval', 
      action="store_true", 
      required=False, 
      help="If set, skips generation and calculates/prints the average scores from the existing 'llm_judge_responses.json' file. This is to be done AFTER this python file creates the JSON."
    )

    args = parser.parse_args()

    if args.eval:
      with open("./llm_judge_responses.json", "r") as f:
        metrics = json.load(f)

        response_1_scores = [metric["response_1_score"] for metric in metrics]
        response_2_scores = [metric["response_2_score"] for metric in metrics]

        response_1_avg = sum(response_1_scores) / len(response_1_scores)
        response_2_avg = sum(response_2_scores) / len(response_2_scores)

        print(f"On average:\n1. the model output with no grammar files averages: {response_1_avg:.2f}\n2. the model output with a grammar file averages: {response_2_avg:.2f}")
      
      return
      

    ds = load_dataset("autoiac-project/iac-eval", "default")['test']
    
    tf_no_grammar_path = Path(args.tf_no_grammar)
    tf_grammar_path = Path(args.tf_with_grammar)

    results = []
    json_output_path = Path("llm_judge_responses.json")
    seen_idx = None

    if json_output_path.exists():
      with open(json_output_path, "r") as f:
          results = json.load(f)
          seen_idx = set(entry["idx"] for entry in results)

    for i, example in tqdm(enumerate(ds)):
        if seen_idx and i in seen_idx:
            continue

        try:
            with open(tf_no_grammar_path / f"response_{i}.tf", "r") as f:
                no_grammar_resp = f.read()

            with open(tf_grammar_path / f"response_{i}.tf", "r") as f:
                grammar_resp = f.read()
        except Exception as _:
            tqdm.write(f"No file found in both directories for index {i}. Skipping.")
            continue

        curr_prompt = example["Prompt"]
        curr_intent = example["Intent"]

        golden_response = example["Reference output"]

        user_query = f"""
        You are a Senior DevOps Engineer. You must compare two Terraform code snippets that follow the same prompt:

        {curr_prompt}

        with the intent to:
        
        {curr_intent}

        The criteria is defined with these three items:
        1. Syntax & Validity
        2. Security best practices
        3. Resource isolation

        Rate it 1-10, where:
        1. 1 (Non-functional): the output contains no valid HCL code, severe syntax errors, or hallucinations
        2. 5 (Partially functional): the code has somewhat correct syntax but fails to meet specific requirements or has logical errors (e.g. missing dependencies) 
        3. 10 (Completely functional): the code is syntactically correct, it fulfills all prompt requirements, and passes "terraform validate".

        IMPORTANT: 
        1. Return ONLY the raw JSON string. Do not wrap it in markdown code blocks. Do not write your responses outside the following format. ONLY have the output in the JSON template below.
        2. Keep your reasoning CONCISE (maximum 2 sentences per response).
        3. DO NOT QUOTE the code snippets in your reasoning. Refer to errors generally (e.g. say "it has syntax errors" instead of quoting the error).
        4. If the code is repetitive or gibberish, just state "The code is repetitive/gibberish" and move on.

        I want you to return the output as a JSON with the following template:
        {{
          "response_1_reasoning": STRING,
          "response_2_reasoning": STRING,
          "response_1_score": INT,
          "response_2_score": INT,
        }}

        This is the golden response:

        {golden_response}

        This is response 1:
        
        {no_grammar_resp}

        This is response 2:

        {grammar_resp}
        """

        response = ask_qwen_cli(user_query)

        try:
            parsed_json = json.loads(response) # Use the helper function from Step 2
            json_entry = {"idx": i, **parsed_json}
            results.append(json_entry)
            
            # Save immediately (good practice to avoid data loss)
            with open(json_output_path, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            tqdm.write(f"Error parsing index {i}: {e}")
            tqdm.write(f"Raw Response: {response}")

if __name__ == "__main__":
  main()