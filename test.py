import json
from llama_cpp import Llama

llm = Llama(
    model_path="./models/Phi-3-mini-4k-instruct-q4.gguf", #https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
)

resp = llm.create_chat_completion(
	messages = [
		{
			"role": "user",
			"content": "What is the capital of France?"
		}
	]
)

print(json.dumps(resp, indent=2))