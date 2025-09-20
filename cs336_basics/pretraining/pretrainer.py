from cs336_basics.pretraining import configuration
from cs336_basics.nn import extensions
from cs336_basics.nn import transformer_lm
from cs336_basics.nn import adam_w
from cs336_basics.nn import cross_entropy
from cs336_basics import pretokenization
import os
import logging
import bisect
import numpy as np
import torch
import time
from cs336_basics.train_bpe import train_bpe
from cs336_basics.bpe_tokenizer import BpeTokenizer
from cs336_basics import bpe_constants
import logging
import multiprocessing
import functools
import statistics
import wandb
import datetime
import dataclasses
from numpy.lib.format import open_memmap
import gc
import json
import pandas as pd

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


class Pretrainer:
    def __init__(self, configuration: configuration.PretrainingConfiguration):
        self._configuration = configuration
        lm_configuration = configuration.training_loop.transformer_llm
        self._i = 0
        model = transformer_lm.TransformerLm(lm_configuration)
        model.register_buffer("start_time", torch.tensor(time.time()))
        
        # Keep uncompiled model for validation to avoid cache invalidation
        self._uncompiled_model = model
        
        # Compile the model with optimizations for training
        self._model = torch.compile(model, mode="default")
        optimizer_configuration = (
            configuration.training_loop.adamw_optimizer_configuration
        )
        assert len(optimizer_configuration.betas) == 2
        self._optimizer = adam_w.AdamW(
            self._model.parameters(),
            lr=optimizer_configuration.lr,
            weight_decay=optimizer_configuration.weight_decay,
            betas=(optimizer_configuration.betas[0], optimizer_configuration.betas[1]),
            eps=optimizer_configuration.eps,
        )
        self._run_id: str | None = None

    def _checkpoint_exists(self, i: int) -> str | None:
        return (
            self._configuration.checkpoint_path(i)
            if os.path.exists(self._configuration.checkpoint_written_path(i))
            else None
        )

    def load_latest_checkpoint(self):
        if self._configuration.checkpoint == 0:
            logger.info("Force fresh start")
            return
        elif self._configuration.checkpoint is not None:
            checkpoint_path = self._checkpoint_exists(self._configuration.checkpoint)
        else:
            checkpoint_power = 1
            modulus = self._configuration.training_loop.checkpoint_persist_modulus
            while self._checkpoint_exists(modulus**checkpoint_power):
                checkpoint_power *= 2
            if checkpoint_power == 1:
                logger.info("checkpoint doesn't exist")
                return
            iteration_values = (
                self._configuration.training_loop.checkpoint_persist_modulus
                ** checkpoint_power
            )
            search_space = range(
                0,
                iteration_values,
                self._configuration.training_loop.checkpoint_persist_modulus,
            )
            checkpoint_iteration = search_space[
                bisect.bisect_left(
                    search_space,
                    x=1,
                    key=lambda i: 0 if self._checkpoint_exists(i) else 1,
                )
                - 1
            ]
            checkpoint_path = self._checkpoint_exists(checkpoint_iteration)
        assert checkpoint_path
        metadata = extensions.load_checkpoint(
            src=checkpoint_path, model=self._model, optimizer=self._optimizer
        )
        self._i = int(metadata["i"]) + 1
        self._run_id = str(metadata["run_id"])

    def _persist_checkpoint(self):
        extensions.save_checkpoint(
            self._model,
            self._optimizer,
            metadata={"i": self._i, "run_id": self._run_id},
            out=self._configuration.checkpoint_path(self._i),
        )
        with open(self._configuration.checkpoint_written_path(self._i), "wb") as f:
            pass

    def _set_annealed_learning_rate(self):
        annealing_configuration = (
            self._configuration.training_loop.annealing_configuration
        )
        cosine_lr = extensions.cosine_learning_rate(
            it=self._i,
            zero_iters=annealing_configuration.zero_iters,
            max_learning_rate=annealing_configuration.max_learning_rate,
            min_learning_rate=annealing_configuration.min_learning_rate,
            warmup_iters=annealing_configuration.warmup_iters,
            cosine_cycle_iters=annealing_configuration.cosine_cycle_iters,
            use_cosine_rampup=(
                annealing_configuration.use_cosine_rampup
                if annealing_configuration.use_cosine_rampup is not None
                else False
            ),
        )
        for pg in self._optimizer.param_groups:
            pg["lr"] = cosine_lr

    def train_tokenizer(self):
        vocabulary, merges = train_bpe(
            input_path=self._configuration.training_loop.training_data_path,
            vocab_size=self._configuration.training_loop.transformer_llm.vocab_size,
            special_tokens=[bpe_constants.END_OF_TEXT],
        )

        tokenizer = BpeTokenizer(vocab=vocabulary, merges=merges)
        tokenizer.persist(*self._configuration.tokenizer_path)

    def ensure_tokenizer(self) -> BpeTokenizer:
        if not os.path.exists(self._configuration.tokenizer_path[0]):
            logger.info("Training bpe tokenizer")
            self.train_tokenizer()

        return BpeTokenizer.from_files(*self._configuration.tokenizer_path)

    @classmethod
    def _encode(
        cls,
        file_range: tuple[int, int],
        configuration: configuration.PretrainingConfiguration,
        data_path: str,
    ):
        try:
            tokenizer = BpeTokenizer.from_files(*configuration.tokenizer_path)
            start, end = file_range
            logger.info(
                "[Worker PID %d] Starting worker start=%d, end=%d",
                os.getpid(),
                start,
                end,
            )
            with open(data_path, "rt") as f:
                f.seek(start)
                result = tokenizer.encode(f.read(end - start))
            output_path = f"{data_path}.tokens.start={start},end={end}.npy"
            result_array = np.array(result)
            np.save(output_path, result_array)
            logger.info(
                "[Worker PID %d] Finished tokenizing start=%d, end=%d, tokens=%d",
                os.getpid(),
                start,
                end,
                result_array.shape[0],
            )
            return (output_path, result_array.shape[0])
        except Exception as e:
            logger.exception(
                "[Worker PID %d] Exception in worker for start=%d, end=%d: %s",
                os.getpid(),
                file_range[0],
                file_range[1],
                e,
            )
            return (None, 0)

    def tokenize_data(self, data_path: str):
        self.ensure_tokenizer()

        while True:
            try:
                num_processes = min(multiprocessing.cpu_count(), 12)
                logger.info(
                    "[Main PID %d] Using %d processes for tokenization",
                    os.getpid(),
                    num_processes,
                )
                with open(data_path, "rb") as f:
                    boundaries = pretokenization.find_chunk_boundaries(
                        f,
                        num_processes * 8,
                        bpe_constants.END_OF_TEXT.encode("utf-8"),
                    )

                logger.info("[Main PID %d] Boundaries: %s", os.getpid(), boundaries)
                ctx = multiprocessing.get_context("spawn")
                with ctx.Pool(processes=num_processes) as pool:
                    token_segment_paths_async_result = pool.map_async(
                        functools.partial(
                            Pretrainer._encode,
                            configuration=self._configuration,
                            data_path=data_path,
                        ),
                        [
                            (start, end)
                            for start, end in zip(boundaries[:-1], boundaries[1:])
                        ],
                    )
                    try:
                        token_segment_paths = token_segment_paths_async_result.get(
                            timeout=20 * 60
                        )
                        logger.info("[Main PID %d] Segmentation complete.", os.getpid())
                    except multiprocessing.TimeoutError:
                        logger.error(
                            "[Main PID %d] Timeout waiting for worker processes to finish.",
                            os.getpid(),
                        )
                        pool.terminate()
                        pool.join()
                        raise

                # Filter out failed chunks
                if [x for x in token_segment_paths if x[0] is None]:
                    raise Exception("There are some failed segments.")

                dtype = (
                    np.int16
                    if self._configuration.training_loop.transformer_llm.vocab_size
                    < (1 << 15)
                    else np.int32
                )
                logger.info("Saving input tokens")
                tmp_path = self._configuration.cached_tokens(data_path) + ".tmp"
                mm = open_memmap(
                    tmp_path,
                    mode="w+",
                    dtype=dtype,
                    shape=(sum(size for _, size in token_segment_paths),),
                )
                offset = 0
                for token_path, size in token_segment_paths:
                    mm[offset : offset + size] = np.array(
                        np.load(token_path),
                        dtype=dtype,
                    )
                    offset += size
                    os.remove(token_path)
                del mm
                os.replace(tmp_path, self._configuration.cached_tokens(data_path))
                break
            except Exception as e:
                logger.exception(
                    f"[Main PID {os.getpid()}] Exception while tokenizing: {e}"
                )

    def get_tokenized_training_data(self):
        if not os.path.exists(self._configuration.tokenized_training_data_path):
            logger.info("Creating tokenized training data")
            self.tokenize_data(
                data_path=self._configuration.training_loop.training_data_path
            )

        return np.load(self._configuration.tokenized_training_data_path, mmap_mode="r")

    def get_tokenized_validation_data(self):
        if not os.path.exists(self._configuration.tokenized_validation_data_path):
            logger.info("Creating tokenized validation data")
            self.tokenize_data(
                data_path=self._configuration.training_loop.validation_data_path
            )

        return np.load(
            self._configuration.tokenized_validation_data_path, mmap_mode="r"
        )

    def get_lrs(self):
        return set([pg["lr"] for pg in self._optimizer.param_groups])

    def should_stop(self):
        return (
            self._configuration.training_loop.max_iterations
            and self._i + 1 > self._configuration.training_loop.max_iterations
        ) or (
            self._configuration.training_loop.time_limit_in_seconds
            and (time.time() - self._model.start_time.item())
            > self._configuration.training_loop.time_limit_in_seconds
        )

    def _run_training_loop(self, tokenized_training_data, tokenized_validation_data):
        cross_entropy_loss = cross_entropy.CrossEntropyLoss()
        total_gradient_value = 1e6

        activation_recorder = extensions.ModuleActivationRecorder(self._model)
        gradient_histogram_recorder = extensions.HistogramRecorder()
        weight_histogram_recorder = extensions.HistogramRecorder()
        while True:
            start = time.time()
            with activation_recorder.intercept_activations(
                intercept=self._i % 50 == 0
            ) as recorder:
                self._optimizer.zero_grad()
                self._set_annealed_learning_rate()
                (training_batch, target) = extensions.get_batch(
                    tokenized_training_data,
                    batch_size=self._configuration.training_loop.batch_size,
                    context_length=self._configuration.training_loop.context_length,
                    device=self._configuration.training_loop.transformer_llm.device,
                )
                loss = cross_entropy_loss(
                    self._model(training_batch.to(torch.int64)),
                    target=target.to(torch.int64).to(
                        device=self._configuration.training_loop.transformer_llm.device,
                    ),
                )
                loss.backward()
                (clipped_gradients, total_gradient_value) = (
                    extensions.gradient_clipping_with_gradient_value(
                        self._model.parameters(),
                        max_l2_norm=self._configuration.training_loop.initial_max_l2_norm,
                    )
                )
                self._optimizer.step()
                log_args = {}
                should_stop = self.should_stop()
                if (
                    self._i
                    and self._i
                    % self._configuration.training_loop.checkpoint_persist_modulus
                    == 0
                ) or should_stop:
                    start_save = time.time()
                    self._persist_checkpoint()
                    (validation_batch, validation_target) = extensions.get_batch(
                        tokenized_validation_data,
                        batch_size=self._configuration.training_loop.batch_size,
                        context_length=self._configuration.training_loop.context_length,
                        device=self._configuration.training_loop.transformer_llm.device,
                    )
                    with torch.no_grad():
                        validation_loss = cross_entropy_loss(
                            self._uncompiled_model(validation_batch.to(torch.int64)),
                            target=validation_target.to(torch.int64).to(
                                device=self._configuration.training_loop.transformer_llm.device,
                            ),
                        )
                    log_args |= {
                        "health/checkpoint_save_time": time.time() - start_save,
                        "metrics/loss/validation": validation_loss.item(),
                    }
                    gc.collect()
                    torch.cuda.empty_cache()
                self._i += 1
                if recorder is not None:
                    log_args |= recorder.logs
                    log_args |= extensions.record_weight_gradients(
                        self._uncompiled_model, gradient_histogram_recorder
                    )
                    log_args |= extensions.record_weights(
                        self._uncompiled_model, weight_histogram_recorder
                    )
                    gc.collect()
                    # torch.cuda.empty_cache()
                wandb.log(
                    log_args
                    | {
                        "metrics/loss/training": loss.item(),
                        "metrics/lr0": list(self.get_lrs())[0],
                        "health/i": self._i,
                        "health/step_time": time.time() - start,
                        "health/gradient/clipping": int(clipped_gradients),
                        "health/gradient/value": total_gradient_value,
                    }
                )
                if should_stop:
                    logging.info(
                        f"Training ended: max_iterations={self._configuration.training_loop.max_iterations}, time_limit={self._configuration.training_loop.time_limit_in_seconds}"
                    )
                    break

    def train(self):
        if self.should_stop():
            logger.info("Stop conditions are met.")
            return
        
        count_parameters = pd.DataFrame(
            [
                {"name": name, "params": param.numel()}
                for name, param in self._uncompiled_model.named_parameters()
            ]
        )
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", None,
            "display.max_colwidth", None
        ):
            print(f"# Params\n{count_parameters}")
        total_params = count_parameters["params"].sum()
        print(f'# Total params: {total_params:,}')

        os.makedirs(self._configuration.output_path, exist_ok=True)
        os.makedirs(self._configuration.checkpoint_dir, exist_ok=True)

        with open(self._configuration.output_metadata_path, "wt") as f:
            json.dump(dataclasses.asdict(self._configuration), f, default=lambda x: None)

        tokenized_training_data = self.get_tokenized_training_data()
        tokenized_validation_data = self.get_tokenized_validation_data()
        assert (
            np.max(tokenized_training_data)
            < self._configuration.training_loop.transformer_llm.vocab_size
        )
        assert (
            np.max(tokenized_validation_data)
            < self._configuration.training_loop.transformer_llm.vocab_size
        )
        logger.info("Checked tokens are valid")

        wandb_kw_args = {"id": self._run_id, "resume": "must"} if self._run_id else {}
        with wandb.init(
            # Set the project where this run will be logged
            project=self._configuration.training_loop.name,
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"{self._configuration.suffix or 'experiment'} {datetime.datetime.fromtimestamp(self._model.start_time.item(), datetime.timezone.utc)} timestamp={self._model.start_time.item()}",
            config=dataclasses.asdict(self._configuration) | {'total_params': total_params},
            **wandb_kw_args,
        ) as run:
            self._run_id = run.id
            
            # Print clean, unescaped URLs for better readability
            project_name = self._configuration.training_loop.name
            run_id = run.id
            logger.info(f"🚀 Clean W&B URLs:")
            logger.info(f"   Project: https://wandb.ai/ante-materija-gmbh/{project_name}")
            logger.info(f"   Run: https://wandb.ai/ante-materija-gmbh/{project_name}/runs/{run_id}")
            
            self._run_training_loop(
                tokenized_training_data=tokenized_training_data,
                tokenized_validation_data=tokenized_validation_data,
            )
