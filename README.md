# CSCE 489 HCL Constrained Generation

## Installation and Running

1. Install python requirements
```
$ python -m venv .venv
# activate virtual environment (.venv/Scripts/activate or source .venv/bin/activate)
$ pip install -r requirements.txt
```

This should install the CPU bound version of `llama-cpp-python`. If you wish to use hardware acceleration, you are [on your own](https://llama-cpp-python.readthedocs.io/en/latest/#installation).


2. Install the model
Check `benchmark.py` for the link to the current model used. Download the `.gguf` file from HuggingFace and add it to the `/models` directory (create it if necessary)

3. Install terraform
Terraform is used to validate the syntatic correctness of the generated HCL code. Install the correct version of terraform for you platfrom from [here](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli). Don't forget to add the executable to your `PATH`, so the python validation code can call it as a subprocess.

4. Run
```
$ python benchmark.py [--count c] [--verbosity v] [--grammar g]
```
- `--count` is the number of lines from the test dataset to run (defaults to all)
- ``--verbosity`` is how much logging to output (0 none, 1 basic logs, 2 output llm output while generating, 3 output model logs) (defaults to 1)
- ``--grammar`` is whether to use the grammar file or not (defaults to false)


## Packages used
- `llama-cpp-python`: to interface with the model and grammar files
- `datasets`: loading our test dataset
- `tqdm`: progress indicator for generation tasks
