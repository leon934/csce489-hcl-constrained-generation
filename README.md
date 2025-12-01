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


3. Run
```
$ python benchmark.py [--count c]
```

Where `--count` is the number of lines from the test dataset to run (defaults to all)


## Packages used
- `llama-cpp-python`: to interface with the model and grammar files
- `datasets`: loading our test dataset
- `tqdm`: progress indicator for generation tasks
