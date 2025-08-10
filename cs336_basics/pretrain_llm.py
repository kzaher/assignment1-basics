import sys
from collections import abc
import argparse
from cs336_basics.experiments import configuration
from cs336_basics.experiments import pretrainer
import json
import dataclasses


def main(argv: abc.Sequence[str]):
    parser = argparse.ArgumentParser(
        description="Pretraining loop for a transfomer based language model."
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Configuration path"
    )
    args = parser.parse_args()
    llm_configuration = configuration.LlmPretrainingConfiguration.from_dict(
        json.load(args.configuration_path)
    )
    llm_configuration = dataclasses.replace(
        llm_configuration, configuration_path=args.configuration_path
    )
    pretrainer_engine = pretrainer.Pretrainer(configuration=llm_configuration)
    pretrainer_engine.train()


if __name__ == "__main__":
    main(sys.argv)
