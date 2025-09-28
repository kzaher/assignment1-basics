"""
uv run python3 cs336_basics/pretrain.py --configuration_path=cs336_basics/pretraining/configurations/tiny_stories.json
uv run cs336_basics/pretrain.py  --configuration_path=cs336_basics/pretraining/configurations/owt.json
uv run cs336_basics/pretrain.py  --configuration_path=cs336_basics/pretraining/configurations/owt.json --meta_parameters_path=cs336_basics/pretraining/configurations/meta_sweep.json

sudo uv run cs336_basics/pretrain.py  --configuration_path=cs336_basics/pretraining/configurations/owt.json --meta_parameters_path=cs336_basics/pretraining/configurations/meta_sweep_vocab_batches.json ;
sudo uv run cs336_basics/pretrain.py  --configuration_path=cs336_basics/pretraining/configurations/owt.json --meta_parameters_path=cs336_basics/pretraining/configurations/meta_sweep_ff.json ;
sudo uv run cs336_basics/pretrain.py  --configuration_path=cs336_basics/pretraining/configurations/owt.json --meta_parameters_path=cs336_basics/pretraining/configurations/meta_sweep_nope.json ;
sudo uv run cs336_basics/pretrain.py  --configuration_path=cs336_basics/pretraining/configurations/owt.json --meta_parameters_path=cs336_basics/pretraining/configurations/meta_sweep_norm.json ;
sudo uv run cs336_basics/pretrain.py  --configuration_path=cs336_basics/pretraining/configurations/owt.json --meta_parameters_path=cs336_basics/pretraining/configurations/meta_sweep_rope_params.json ;

sudo uv run cs336_basics/pretrain.py  --configuration_path=cs336_basics/pretraining/configurations/owt.json --meta_parameters_path=cs336_basics/pretraining/configurations/meta_sweep_relu_soft.json ;
sudo uv run cs336_basics/pretrain.py  --configuration_path=cs336_basics/pretraining/configurations/owt.json --meta_parameters_path=cs336_basics/pretraining/configurations/meta_sweep_ff.json ;

sudo uv run cs336_basics/pretrain.py  --configuration_path=cs336_basics/pretraining/configurations/owt.json --meta_parameters_path=cs336_basics/pretraining/configurations/meta_sweep_ff.json --checkpoint=0;
sudo uv run cs336_basics/pretrain.py  --configuration_path=cs336_basics/pretraining/configurations/owt_10MB.json --meta_parameters_path=cs336_basics/pretraining/configurations/meta_sweep_ff.json --checkpoint=0 ;
sudo uv run cs336_basics/pretrain.py  --configuration_path=cs336_basics/pretraining/configurations/owt_10k.json --meta_parameters_path=cs336_basics/pretraining/configurations/meta_sweep_ff.json  --checkpoint=0 ;

sudo uv run cs336_basics/pretrain.py  --configuration_path=cs336_basics/pretraining/configurations/owt_gemma_2B.json --meta_parameters_path=cs336_basics/pretraining/configurations/meta_sweep_ff.json  --checkpoint=0 

sudo uv run cs336_basics/pretrain.py  --configuration_path=cs336_basics/pretraining/configurations/owt_gemma_270M.json --meta_parameters_path=cs336_basics/pretraining/configurations/meta_sweep_ff.json  --checkpoint=0 

sudo uv run cs336_basics/pretrain.py  --configuration_path=cs336_basics/pretraining/configurations/owt_gemma_270M.json --meta_parameters_path=cs336_basics/pretraining/configurations/meta_sweep_ff_batch.json  --checkpoint=0 

uv run cs336_basics/pretrain.py  --configuration_path=cs336_basics/pretraining/configurations/owt_gemma_270M.json --meta_parameters_path=cs336_basics/pretraining/configurations/meta_sweep_ff_learning_rate.json  --checkpoint=0

RUN="uv run cs336_basics/pretrain.py  --configuration_path=cs336_basics/pretraining/configurations/owt_gemma_270M.json"
pushrun podscreen bash -c  "type down && $RUN 2>&1 | tee last_output.txt; sleep 30 && down"
"""

import sys
from collections import abc
import argparse
from cs336_basics.pretraining import pretrainer, configuration
import json
import extensions
import dataclasses
import logging
import pandas as pd

extensions.setup_default_logging()


def run_configuration(configuration_instance: configuration.PretrainingConfiguration, dry_run: bool):
    if dry_run:
        return
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
        "--dry_run",
        type=bool,
        required=False,
        default=False,
        help="If only the configurations are validated and the model isn't trained",
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
        json.dumps(dataclasses.asdict(configuration_instance), indent=4, default=str),
    )
    if meta_parameters_path := args.meta_parameters_path:
        with open(meta_parameters_path, "rt") as f:
            parameter_sweep_configuration = (
                configuration.ParameterSweepConfiguration.from_dict(json.load(f))
            )

        def update_configuration(paths: list[str], values: list[object]):
            assert len(paths) == len(values)
            c = configuration_instance
            def trimm(path: str):
                if '.' in path:
                    return path.split('.')[-1]
                return path
            compressed_paths = [trimm(p) for p in paths]
            c = extensions.replace_recursively(
                c,
                lambda x: x.training_loop.name,
                f'{c.training_loop.name}.{",".join(compressed_paths)}',
            )
            suffix = ",".join([f"{path}={value}" for path, value in zip(compressed_paths, values)])
            c = dataclasses.replace(c, suffix=suffix)
            for path, value in zip(paths, values):
                c = extensions.replace_recursively(
                    c,
                    lambda x: eval("x." + path, {}, {"x": x}),
                    value,
                )
            return c

        all_configurations = (
            pd.DataFrame(
                [
                    {
                        "i": parameter_index,
                        "path": parameter_override.path,
                        "value": override_value,
                    }
                    for parameter_override in parameter_sweep_configuration.values
                    for parameter_index, override_value in enumerate(parameter_override.values)
                ]
            )
            .groupby("i")
            .apply(
                lambda df: update_configuration(
                    paths=df["path"].tolist(), values=df["value"].tolist()
                )
            )
        )
        for mutated_configuration in all_configurations:
            assert mutated_configuration != configuration_instance
            assert mutated_configuration.suffix
            assert (
                mutated_configuration.training_loop.max_iterations
                or mutated_configuration.training_loop.time_limit_in_seconds
            )
            logging.info(
                "modified_configuration=%s",
                json.dumps(dataclasses.asdict(mutated_configuration), indent=4, default=str),
            )
            run_configuration(mutated_configuration, dry_run=args.dry_run)
    else:
        run_configuration(configuration_instance=configuration_instance, dry_run=args.dry_run)


if __name__ == "__main__":
    main(sys.argv)
