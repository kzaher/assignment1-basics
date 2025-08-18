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
from torch import Tensor
from jaxtyping import Int
from cs336_basics.train_bpe import train_bpe
from cs336_basics.bpe_tokenizer import BpeTokenizer
from cs336_basics import bpe_constants
import logging, sys
import multiprocessing
import functools
import statistics
import wandb
import datetime
import dataclasses
from numpy.lib.format import open_memmap
import gc

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


class Pretrainer:
    def __init__(self, configuration: configuration.PretrainingConfiguration):
        self._configuration = configuration
        lm_configuration = configuration.training_loop.transformer_llm
        self._i = 0
        model = transformer_lm.TransformerLm(
            vocab_size=lm_configuration.vocab_size,
            max_sequence_length=lm_configuration.max_sequence_length,
            d_model=lm_configuration.d_model,
            num_layers=lm_configuration.num_layers,
            num_heads=lm_configuration.num_heads,
            d_ff=lm_configuration.d_ff,
            rope_theta=lm_configuration.rope_theta,
            device=lm_configuration.device,
            dtype=getattr(torch, lm_configuration.dtype or "float32"),
        )
        model.register_buffer("start_time", torch.tensor(time.time()))
        self._model = torch.compile(model)
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

    def _checkpoint_exists(self, i: int) -> str | None:
        return (
            self._configuration.checkpoint_path(i)
            if os.path.exists(self._configuration.checkpoint_written_path(i))
            else None
        )

    def load_latest_checkpoint(self):
        if self._configuration.checkpoint:
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
        self._i = (
            extensions.load_checkpoint(
                src=checkpoint_path, model=self._model, optimizer=self._optimizer
            )
            + 1
        )

    def _persist_checkpoint(self):
        extensions.save_checkpoint(
            self._model,
            self._optimizer,
            self._i,
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

    def get_tokenizer(self) -> BpeTokenizer:
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
        tokenizer = BpeTokenizer.from_files(*configuration.tokenizer_path)
        start, end = file_range
        logger.info("Starting worker")
        with open(data_path, "rt") as f:
            f.seek(start)
            result = tokenizer.encode(f.read(end - start))
        output_path = f"{data_path}.tokens.start={start},end={end}.npy"
        result_array = np.array(result)
        np.save(output_path, result_array)
        logger.info("Worker tokenized")
        return (output_path, result_array.shape[0])

    def tokenize_data(self, data_path: str):
        num_processes = min(multiprocessing.cpu_count(), 12)
        with open(data_path, "rb") as f:
            boundaries = pretokenization.find_chunk_boundaries(
                f,
                num_processes * 8,
                bpe_constants.END_OF_TEXT.encode("utf-8"),
            )

        with multiprocessing.Pool(processes=num_processes) as pool:
            token_segment_paths = pool.map(
                functools.partial(
                    Pretrainer._encode,
                    configuration=self._configuration,
                    data_path=data_path,
                ),
                [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])],
            )

        dtype = (
            np.int16
            if self._configuration.training_loop.transformer_llm.vocab_size < (1 << 15)
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

    def train(self):
        wandb.init(
            # Set the project where this run will be logged
            project=self._configuration.training_loop.name,
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"experiment_{self._model.start_time.item()} ({datetime.datetime.fromtimestamp(self._model.start_time.item(), datetime.timezone.utc)})",
            config=dataclasses.asdict(self._configuration),
        )
        os.makedirs(self._configuration.output_path, exist_ok=True)
        os.makedirs(self._configuration.checkpoint_dir, exist_ok=True)

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

        cross_entropy_loss = cross_entropy.CrossEntropyLoss()
        total_gradient_value = 1e6
        distribution_of_gradient_values = []
        while True:
            start = time.time()
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
            distribution_of_gradient_values.append(total_gradient_value)
            distribution_of_gradient_values = distribution_of_gradient_values[-1000:]

            self._optimizer.step()
            grads1, grads2, grads3, grads4, grads5, grads6, grads7 = (
                statistics.quantiles(distribution_of_gradient_values, n=8)
            )
            wandb.log(
                {
                    "loss": loss.item(),
                    "i": self._i,
                    "clipped_gradients": int(clipped_gradients),
                    "lr0": list(self.get_lrs())[0],
                    "grad": total_gradient_value,
                    "grads0": min(distribution_of_gradient_values),
                    "grads1": grads1,
                    "grads2": grads2,
                    "grads3": grads3,
                    "grads4": grads4,
                    "grads5": grads5,
                    "grads6": grads6,
                    "grads7": grads7,
                    "grads8": max(distribution_of_gradient_values),
                    "step_time": time.time() - start,
                }
            )
            if (
                self._i
                and self._i
                % self._configuration.training_loop.checkpoint_persist_modulus
                == 0
            ):
                start_save = time.time()
                self._persist_checkpoint()
                (validation_batch, validation_target) = extensions.get_batch(
                    tokenized_validation_data,
                    batch_size=self._configuration.training_loop.batch_size,
                    context_length=self._configuration.training_loop.context_length,
                    device=self._configuration.training_loop.transformer_llm.device,
                )
                validation_loss = cross_entropy_loss(
                    self._model(validation_batch.to(torch.int64)),
                    target=validation_target.to(torch.int64).to(
                        device=self._configuration.training_loop.transformer_llm.device,
                    ),
                )
                wandb.log(
                    {
                        "checkpoint_save_time": time.time() - start_save,
                        "validation_loss": validation_loss.item(),
                    }
                )
                gc.collect()
                torch.cuda.empty_cache()
            self._i += 1
