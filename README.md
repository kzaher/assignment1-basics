# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### Experiments:

* [Vocabulary size experiment](https://wandb.ai/ante-materija-gmbh/owt_sweep4.training_loop.transformer_llm.vocab_size?nw=nwuserantematerija)
* [Batch size experiment](https://wandb.ai/ante-materija-gmbh/owt_sweep.training_loop.batch_size,training_loop.context_length?nw=nwuserantematerija)
* [Silu experiment](https://wandb.ai/ante-materija-gmbh/owt_sweep.training_loop.transformer_llm.experiments.ff_type?nw=nwuserantematerija)
* [Nope experiment](https://wandb.ai/ante-materija-gmbh/owt_sweep.training_loop.transformer_llm.experiments.use_nope?nw=nwuserantematerija)
* [Rope experiment](https://wandb.ai/ante-materija-gmbh/owt_sweep.training_loop.transformer_llm.rope_theta)
* [Rms experiment](https://wandb.ai/ante-materija-gmbh/owt_sweep.training_loop.transformer_llm.experiments.rms_post_norm)  
* [Mixture experiment](https://wandb.ai/ante-materija-gmbh/owt_introspect.e.ff_type/runs/anbf5iok?nw=nwuserantematerija) 
* [ReluSoft experiment](https://wandb.ai/ante-materija-gmbh/owt_introspect.e.ff_type%2Ce.ff_relu_squeeze_factor%2Ce.ff_relu_min?nw=nwuserantematerija)

NextIdeas
* It appears that the FFReGLU​(x)=(max(0,xW1​)⊗(xV))W2. Try to extend them so that they aren't non linear any longer and that they support multiple inputs.
* Try to make it attract towards 1, 0, or -1 for both inputs and outputs.
* The algorithm is to find the biggest amplitude and then backpropagate.
* Create an experiment. Have a tiny network and try to memorize text. That should reduce the amount of waiting and computation, and then try to perform it on the big one.

* Try to get non linear gradient
* Uncorrelated signals in a layer

Blackwell - install torch
```
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu129
```