from argparse import ArgumentParser
import os

from llama_cpp import Llama, LlamaGrammar, llama_chat_format
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

from validate import validate_terraform_files

parser = ArgumentParser(description="Generates Terraform code from the 'iac-eval' dataset using a local LLM.")

parser.add_argument(
    '-c', '--count', 
    type=int, 
    default=-1, 
    help="The limit on the number of dataset examples to process. Default is -1 (process all examples)."
)
parser.add_argument(
    '-g', '--grammar', 
    action="store_true", 
    help="Enable GBNF grammar constraints during generation. Requires 'grammar.gbnf' file to be present."
)
parser.add_argument(
    '-v', '--verbosity', 
    type=int, 
    default=1, 
    help="Control log output level: 0=Silent, 1=Basic progress, 2=Print LLM output to console, 3=Full model debug logs."
)
parser.add_argument(
    '-s', '--start', 
    type=int, 
    default=0, 
    help="The index of the dataset to start generation from; useful for resuming interrupted runs."
)
parser.add_argument(
    '-e', '--end', 
    type=int, 
    default=float('inf'), 
    help="The index of the dataset to stop generation at; useful for creating batches."
)

parser.add_argument('-c', '--count', type=int, default=-1, help="number of lines from the dataset to run, defaults to all (-1)")
parser.add_argument('-g', '--grammar', action="store_true", help="set to use the grammar file")
parser.add_argument('-v', '--verbosity', type=int, default=1, help="how verbose the output is: 0 (no output), 1 (basic logging statements), 2 (output llm while writing), 3 (output model logging)")
parser.add_argument('-s', '--start', type=int, default=0, help="starting idx, default to beginning")
parser.add_argument('-e', '--end', type=int, default=float('inf'), help="ending idx, defaults to min(inf, dataset size)")

args = parser.parse_args()

output_format = \
"""Prompt: {prompt}

Output: {output}
=============
"""

input_format = \
"""You are a code-writer for the HCL language. Report all your answers as plain HCL code. DO NOT wrap the code in a Markdown code block.
Prompt: {prompt}"""

def log(message, verbosity):
	if args.verbosity >= verbosity:
		print(message)

if __name__ == "__main__":
	grammar = None
	if args.grammar:
		try:
			grammar = LlamaGrammar.from_file("grammar.gbnf")
			log("loaded grammar file", 1)
		except Exception as _:
			pass
	else:
		log("skipped loading grammar (enable -g to load grammar)", 1)
		
	
	log("initializing llm", 1)
	llm = Llama(
	    model_path="./models/Phi-3-mini-4k-instruct-q4.gguf", # https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
	    n_ctx=4096, # increase context because we can for some reason
		n_threads=os.cpu_count() - 1,
	    verbose=args.verbosity >= 3, # hide output of model, pollutes terminal
		n_gpu_layers=-1
	)
	
	template = llm.metadata["tokenizer.chat_template"]
	# bos = llm._model.token_get_text(llm.token_bos())
	eos = llm._model.token_get_text(llm.token_eos())
	
	formatter = llama_chat_format.Jinja2ChatFormatter(
	    template=template,
	    bos_token="", # empty bos token to not add <s> tag and remove warning
	    eos_token=eos,
	    add_generation_prompt=True,
	)
	
	log("loading dataset", 1)
	ds = load_dataset("autoiac-project/iac-eval", "default")['test']
	
	if args.count > 0:
		ds = ds.select(range(args.count))

	# creates directory for all generated tf files	
	model_output_path = rf"{os.path.dirname(os.path.abspath(__file__))}/raw_outputs{"_grammar" if args.grammar else ""}"
	if not os.path.exists(model_output_path):
		os.makedirs(model_output_path)
	else:
		log("resetting raw outputs folder", 1)
		for filename in os.listdir(model_output_path):
			filepath = os.path.join(model_output_path, filename)
			os.remove(filepath)
			
	start_idx = 0	

	if args.start:
		start_idx = args.start

	end_idx = len(ds)

	if args.end:
		end_idx = min(end_idx, int(args.end))

	for i, example in tqdm(enumerate(ds.select(range(start_idx, end_idx)), start=start_idx), desc="Generating .tf files"):
		original_prompt = example["Prompt"]
	
		messages = [{"role": "user", "content": input_format.format(prompt=original_prompt)}]
		prompt = formatter(messages=messages).prompt

		resp = llm.create_completion(
			prompt=prompt,
			max_tokens=None,
			temperature=0.0,
		    grammar=grammar
		)
		
		# strip markdown (if necessary)
		# assumes model's response is purely just the tf code
		model_resp = resp["choices"][0]["text"]
		stripped_response = model_resp.removeprefix("```hcl").removeprefix(" ```hcl").removesuffix("```")

    # extensive logging
		if args.verbosity >= 2:
			tqdm.write(output_format.format(prompt=original_prompt, output=stripped_response))
		
		# each row corresponds to i-th idx in original iac-eval dataset
		filename = rf"{model_output_path}/response_{i}.tf"
		with open(filename, "w", encoding="utf-8") as f:
			try:
				f.write(stripped_response)
			except UnicodeEncodeError:
				tqdm.write(f"Unicode error saving file {filename}. Skipping.")
			except IOError:
				tqdm.write(f"IOError saving file {filename}. Skipping.")
			except Exception as e:
				tqdm.write(f"Unknown error saving file {filename}, {e}. Skipping.")

	log("starting validation", 1)
	# then use imported function to validate each generated .tf file
	valid_tf_path = validate_terraform_files(Path(model_output_path), args.grammar)
