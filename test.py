import json
from llama_cpp import Llama, LlamaGrammar

grammar = LlamaGrammar.from_file("grammar.gbnf")

llm = Llama(
    model_path="./models/Phi-3-mini-4k-instruct-q4.gguf", #https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
)

resp = llm.create_completion(
	prompt="Is Paris the capitol of France? ",
    grammar=grammar
)

print(json.dumps(resp, indent=2))