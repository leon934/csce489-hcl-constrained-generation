from argparse import ArgumentParser
import json
from llama_cpp import Llama, LlamaGrammar, llama_chat_format
from datasets import load_dataset
from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument('-c', '--count', type=int, default=-1, help="number of lines from the dataset to run, defaults to all (-1)")
parser.add_argument('-g', '--grammar', action="store_true", help="set to use the grammar file")
args = parser.parse_args()

output_format = \
"""Prompt: {prompt}

Output: {output}
=============
"""

if __name__ == "__main__":
	grammar = None
	if args.grammar:
		try:
			grammar = LlamaGrammar.from_file("grammar.gbnf")
		except:
			pass
	
	llm = Llama(
	    model_path="./models/Phi-3-mini-4k-instruct-q4.gguf", #https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
	    n_ctx=4096, # increase context because we can for some reason
		n_threads=8,
	    verbose=False, # hide output of model, pollutes terminal
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
	
	ds = load_dataset("autoiac-project/iac-eval", "default")
	
	if args.count > 0:
		ds = ds['test'].select(range(args.count))
	else:
		ds = ds['test']
	
	for example in tqdm(ds):
		original_prompt = example["Prompt"]
	
		messages = [{"role": "user", "content": "You are a code-writer for the HCL language. Report all your answers as only HCL code.\nPrompt: " + original_prompt}]
		prompt = formatter(messages=messages).prompt
		
		resp = llm.create_completion(
			prompt=prompt,
			max_tokens=None,
			temperature=0.0,
		    grammar=grammar
		)
		
		# print(json.dumps(resp, indent=2))
		#TODO: replace this write with saving the response as a file temporarily for further validation?
		tqdm.write(output_format.format(prompt=original_prompt, output=resp['choices'][0]['text']))
