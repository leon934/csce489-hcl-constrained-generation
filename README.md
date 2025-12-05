# CSCE 489 HCL Constrained Generation

Duncan Redheedran, Leon Nguyen, Lyle Morris

## File contents of the `.zip`

1. `raw_outputs` contains the model output without using the grammar file
2. `raw_outputs_grammar` contains the model output with using the grammar file
3. `stripped_outputs` contains the processed `raw_outputs` `.tf` files
4. `stripped_outputs_grammar` contains the processed `raw_outputs_grammar` `.tf` files
5. `*.py` are the necessary Python files to evaluate the LLM
6. `requirements.txt` is used to set up the environment
7. `llm_judge_responses.json` is Qwen3's evaluation of the grammar vs. non-grammar model responses
8. `grammar.gbnf` contains the grammar to constrain the model output

## Installation

1. Install python requirements

```
$ python -m venv .venv
# activate virtual environment (.venv/Scripts/activate or source .venv/bin/activate)
$ pip install -r requirements.txt
```

This should install the CPU bound version of `llama-cpp-python`. If you wish to use hardware acceleration, you are [on your own](https://llama-cpp-python.readthedocs.io/en/latest/#installation).

2. Install the model
   Check `benchmark.py` for the link to the current model used (as of 12/4 we used [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)). Download the `.gguf` file from HuggingFace and add it to the `/models` directory (create it if necessary)

3. Install terraform
   Terraform is used to validate the syntatic correctness of the generated HCL code. Install the correct version of terraform for you platfrom from [here](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli). Don't forget to add the executable to your `PATH`, so the python validation code can call it as a subprocess.

Since `benchmark.py` downloads IaC-Eval from the internet through HuggingFace, running this Python file with a Slurm job will not suffice. You would manually have to install the dataset and point HuggingFace towards it.

## Running the code

There are three key files used to evaluate the LLM:

```
$ python benchmark.py [--count c] [--verbosity v] [--grammar g]
```

-   `--count` is the number of lines from the test dataset to run (defaults to all)
-   `--verbosity` is how much logging to output (0 none, 1 basic logs, 2 output llm output while generating, 3 output model logs) (defaults to 1)
-   `--grammar` is whether to use the grammar file or not (defaults to false)

Note that `benchmark.py` will also run `validate.py`.

```
$ python validate.py [--raw_dir]
```

-   `--raw_dir` is the path that `benchmark.py` created to store all the model's responses to each of the IaC-eval data points.

This function creates:

1. `./stripped_outputs/`: this directory holds the files after they've been preprocessed
2. `./valid_outputs/`: this directory holds the files that have passed both `terraform init` and `terraform validate`

```
$ python llm_judge.py [--tf_no_grammar] [--tf_with_grammar] [--eval]
```

-   `--tf_no_grammar` is the directory that holds the `response_i.tf` files the model generated without the `-g` flag
-   `--tf_with_grammar` is similar to `--tf_no_grammar` but with `-g` flag
-   `--eval` should be used after `llm_judge.py` is ran the first time, since without this flag, it generates the necessary JSON to evaluate the responses side by side.

Since this model uses Qwen3-8B through the CLI, it must be installed first:

1. With `npm`: `npm install -g @qwen-code/qwen-code@latest`
2. With `brew`: `brew install qwen-code`

## Typical Workflow

To fully evaluate the model, a typical workflow after setting up the environment looks like the following:

```bash
# to obtain raw and cleaned response to prompts w/ and w/o grammar
python benchmark.py -g
python benchmark.py

# obtain llm_judge_responses.json
python llm_judge.py --tf_no_grammar ./stripped_outputs --tf_with_grammar stripped_outputs_grammar

# evaluate llm_judge_responses.json
python llm_judge.py --tf_no_grammar ./stripped_outputs --tf_with_grammar stripped_outputs_grammar --eval
```

## Packages used

-   `llama-cpp-python`: to interface with the model and grammar files
-   `datasets`: loading our test dataset
-   `tqdm`: progress indicator for generation tasks

There are a bunch of other smaller packages used; simply installing the `requirements.txt` folder should handle these dependencies.
