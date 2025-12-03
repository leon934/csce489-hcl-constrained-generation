import json
from llama_cpp import Llama, LlamaGrammar, llama_chat_format

grammar = LlamaGrammar.from_file("grammar.gbnf")
print(grammar)

llm = Llama(
    model_path="./models/Phi-3-mini-4k-instruct-q4.gguf", #https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
)

template = llm.metadata["tokenizer.chat_template"]
bos = llm._model.token_get_text(llm.token_bos())
eos = llm._model.token_get_text(llm.token_eos())

formatter = llama_chat_format.Jinja2ChatFormatter(
    template=template,
    bos_token=bos,
    eos_token=eos,
    add_generation_prompt=True,
)

messages = [{"role": "user", "content": "Write a terrafrom module for an S3 bucket named lyle-test"}]
prompt = """<|user|>
You are a code-writer for the HCL language. Report all your answers as plain HCL code. DO NOT wrap the code in a Markdown code block.
Prompt: Configure a query log that can create a log stream and put log events using Route 53 resources. Name the zone "primary", the cloudwatch log group "aws_route53_example_com", and the cloudwatch log resource policy "route53-query-logging-policy"<|end|>
<|assistant|>"""

print(prompt)

resp = llm.create_completion(
	prompt=prompt,
    grammar=grammar,
    max_tokens=10_000,
    top_p = 1.0,
    top_k = 1_000_000,
)

print(json.dumps(resp, indent=2))