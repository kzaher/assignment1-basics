"""
uv run python3 cs336_basics/pretrain_llm.py --configuration_path=cs336_basics/pretraining/configurations/tiny_stories.json
uv run cs336_basics/pretrain_llm.py  --configuration_path=cs336_basics/pretraining/configurations/owt.json
uv run cs336_basics/pretrain_llm.py  --configuration_path=cs336_basics/pretraining/configurations/owt.json --meta_parameters_path=cs336_basics/pretraining/configurations/meta_sweep.json
"""

import sys
from collections import abc
import argparse
from cs336_basics.pretraining import pretrainer, configuration
import json
import extensions
import dataclasses
import logging

extensions.setup_default_logging()


def run_configuration(configuration_instance: configuration.PretrainingConfiguration):
    pretrainer_engine = pretrainer.Pretrainer(configuration=configuration_instance)
    pretrainer_engine.load_latest_checkpoint()
    pretrainer_engine.train()


def main(argv: abc.Sequence[str]):
    parser = argparse.ArgumentParser(
        description="Pretraining loop for a transfomer based language model."
    )
    parser.add_argument(
        "--configuration_path", type=str, required=True, help="Configuration path"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="experiments",
        help="Output directory path",
    )
    parser.add_argument(
        "--meta_parameters_path",
        type=str,
        required=False,
        default=None,
        help="Output directory path",
    )
    parser.add_argument(
        "--checkpoint", type=int, required=False, default=None, help="Checkpoint index"
    )
    args = parser.parse_args()

    with open(args.configuration_path, "rt") as f:
        configuration_instance = configuration.PretrainingConfiguration(
            output_path=args.output_path,
            checkpoint=args.checkpoint,
            training_loop=configuration.LlmPretrainingTrainingLoopConfiguration.from_dict(
                json.load(f)
            ),
        )
    logging.info(
        "main_configuration=%s",
        json.dumps(dataclasses.asdict(configuration_instance), indent=4),
    )
    if meta_parameters_path := args.meta_parameters_path:

        with open(meta_parameters_path, "rt") as f:
            parameter_sweep_configuration = (
                configuration.ParameterSweepConfiguration.from_dict(json.load(f))
            )
        all_configurations = [
            lambda c, replacement_value=replacement_value: (
                dataclasses.replace(
                    extensions.replace_recursively(
                        c,
                        lambda x: eval("x." + override_value.path, locals={"x": x}),
                        replacement_value,
                    ),
                    suffix=f"{override_value.path}={replacement_value}",
                )
            )
            for override_value in parameter_sweep_configuration.values
            for replacement_value in override_value.values
        ]
        for configuration_value_mutation in all_configurations:
            mutated_configuration = configuration_value_mutation(configuration_instance)
            assert mutated_configuration != configuration_instance
            assert mutated_configuration.suffix
            assert (
                mutated_configuration.training_loop.max_iterations
                or mutated_configuration.training_loop.time_limit_in_seconds
            )
            logging.info(
                "modified_configuration=%s",
                json.dumps(dataclasses.asdict(mutated_configuration), indent=4),
            )
            run_configuration(mutated_configuration)
    else:
        run_configuration(configuration_instance=configuration_instance)


if __name__ == "__main__":
    main(sys.argv)
