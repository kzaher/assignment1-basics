"""
uv run python3 cs336_basics/pretrain_llm.py --configuration_path=cs336_basics/pretraining/configurations/tiny_stories.json --input_path=/mnt/fastdisk/data/TinyStoriesV2-GPT4-train.txt --output_path=/mnt/fastdisk/experiments/tiny_stories_1
"""

import sys
from collections import abc
import argparse
from cs336_basics.pretraining import pretrainer, configuration
import json
import logging

logging.basicConfig(
    level=logging.INFO,  # or DEBUG
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,  # default is sys.stderr; pick stdout if you prefer
)


def main(argv: abc.Sequence[str]):
    parser = argparse.ArgumentParser(
        description="Pretraining loop for a transfomer based language model."
    )
    parser.add_argument(
        "--configuration_path", type=str, required=True, help="Configuration path"
    )
    parser.add_argument("--input_path", type=str, required=True, help="Input text path")
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output directory path"
    )
    args = parser.parse_args()

    with open(args.configuration_path, "rt") as f:
        configuration_str = f.read()
    configuration_instance = configuration.PretrainingConfiguration(
        input_path=args.input_path,
        output_path=args.output_path,
        training_loop=configuration.LlmPretrainingTrainingLoopConfiguration.from_dict(
            json.loads(configuration_str)
        ),
    )
    print(configuration_instance)
    pretrainer_engine = pretrainer.Pretrainer(configuration=configuration_instance)
    pretrainer_engine.load_latest_checkpoint()
    pretrainer_engine.train()


if __name__ == "__main__":
    main(sys.argv)
