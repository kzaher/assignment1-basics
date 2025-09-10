"""
uv run python3 cs336_basics/sampling.py --experiment_path=experiments/owt_sweep4.training_loop.transformer_llm.vocab_size/training_loop.transformer_llm.vocab_size=32000
"""

import sys

from collections import abc
import argparse
from cs336_basics.pretraining import pretrainer, configuration
from cs336_basics.nn import transformer_lm
from cs336_basics.nn import softmax
import json
from jaxtyping import Float
from torch import Tensor
import torch
import logging
from cs336_basics import extensions
from cs336_basics import bpe_tokenizer
from cs336_basics import bpe_constants
import dataclasses
import tqdm

extensions.setup_default_logging()

logger = logging.getLogger(__name__)


def sample(
    model: transformer_lm.TransformerLm,
    in_indices: Float[Tensor, "... batch_size sequence_length"],
    temperature: float,
    nucleus_p: float,
) -> Float[torch.Tensor, "... batch_size"]:
    logits = model(in_indices=in_indices)[..., -1, :]
    if temperature:
        logits = logits / temperature
    logits = softmax.Softmax(dim=-1)(logits)
    max_args = torch.argsort(logits, dim=-1, descending=True)
    top_logits = torch.gather(logits, dim=-1, index=max_args)
    top_mask = (torch.cumsum(top_logits, dim=-1) <= nucleus_p)
    # The first one over is also taken
    top_mask = torch.concat((torch.ones_like(top_mask)[..., :1], top_mask[..., :-1]), dim=-1)
    top_distribution = top_logits * top_mask
    max_arg_indices = torch.multinomial(
        top_distribution, num_samples=1
    )
    tokens = torch.gather(max_args, dim=-1, index=max_arg_indices)
    return tokens


def main(argv: abc.Sequence[str]):
    parser = argparse.ArgumentParser(description="Simplest sampling for transformer.")
    parser.add_argument(
        "--experiment_path",
        type=str,
        default="experiments/owt_sweep4.training_loop.transformer_llm.vocab_size/training_loop.transformer_llm.vocab_size=32000",
        required=False,
        help="Experiment path",
    )
    parser.add_argument(
        "--checkpoint", type=int, required=False, default=None, help="Checkpoint index"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1,
        help="Checkpoint index",
    )
    parser.add_argument(
        "--nucleus_p",
        type=float,
        required=False,
        default=0.2,
        help="Nucleus probability.",
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        required=False,
        default=100,
        help="Number of sampled tokens",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        required=False,
        default=5,
        help="Number of sampled traces",
    )
    args = parser.parse_args()

    with open(
        configuration.PretrainingConfiguration.get_output_metadata_path(
            args.experiment_path
        ),
        "rt",
    ) as f:
        configuration_instance = configuration.PretrainingConfiguration.from_dict(
            json.load(f)
        )
        if args.checkpoint:
            configuration_instance = dataclasses.replace(
                configuration_instance, checkpoint=args.checkpoint
            )
    logger.info('configuration=%s', json.dumps(dataclasses.asdict(configuration_instance), indent=4))
    pretrainer_engine = pretrainer.Pretrainer(configuration=configuration_instance)
    logger.info("Loading checkpoint ...")
    pretrainer_engine.load_latest_checkpoint()
    logger.info("Loading complete")

    device = configuration_instance.training_loop.transformer_llm.device
    while True:
        prompt = input("Input the desired text: ")
        tokenizer: bpe_tokenizer.BpeTokenizer = pretrainer_engine.ensure_tokenizer()
        if not prompt:
            logger.error('Empty prompt, please enter something')
            continue
        input_tokens = torch.tensor(
            [tokenizer.encode(prompt)] * args.sample_count, dtype=torch.int32
        ).to(device=device)
        generated_tokens = torch.tensor([[]] * args.sample_count, dtype=torch.int32).to(
            device=device
        )
        for _ in tqdm.tqdm(range(args.max_output_tokens)):
            with torch.no_grad():
                new_tokens = sample(
                    pretrainer_engine._model,
                    torch.concat((input_tokens, generated_tokens), dim=-1),
                    temperature=args.temperature,
                    nucleus_p=args.nucleus_p,
                )
                assert new_tokens.max().cpu() < configuration_instance.training_loop.transformer_llm.vocab_size
            generated_tokens = torch.concat((generated_tokens, new_tokens), dim=-1)
        eof_token = tokenizer.special_token(bpe_constants.END_OF_TEXT)
        for i in range(generated_tokens.size(0)):
            tokens = torch.concat((input_tokens, generated_tokens), dim=-1)[i, :].cpu().tolist()
            if eof_token in tokens:
                tokens = tokens[:tokens.find(eof_token)]
            print(
                f"Output[{i}]={tokenizer.decode(tokens)}"
            )

if __name__ == "__main__":
    main(sys.argv)
