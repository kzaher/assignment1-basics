from cs336_basics.experiments import configuration
from cs336_basics.nn import extensions
from cs336_basics.nn import transformer_lm
from cs336_basics.nn import adam_w
from cs336_basics.nn import cross_entropy
import os
import logging
import bisect
import numpy as np
import torch
import time
from torch import Tensor
from jaxtyping import Int

logger = logging.getLogger(__name__)


class Pretrainer:
    def __init__(self, configuration: configuration.LlmPretrainingConfiguration):
        self._configuration = configuration
        lm_configuration = configuration.transformer_llm
        self._i = 0
        self._model = transformer_lm.TransformerLm(
            vocab_size=lm_configuration.vocab_size,
            context_length=lm_configuration.context_length,
            d_model=lm_configuration.d_model,
            num_layers=lm_configuration.num_layers,
            num_heads=lm_configuration.num_heads,
            d_ff=lm_configuration.d_ff,
            rope_theta=lm_configuration.rope_theta,
            device=lm_configuration.device,
        )
        optimizer_configuration = configuration.optimizer_configuration
        assert len(optimizer_configuration.betas) == 2
        self._optimizer = adam_w.AdamW(
            self._model,
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
        checkpoint_power = 1
        modulus = self._configuration.checkpoint_persist_modulus
        while self._checkpoint_exists(modulus**checkpoint_power):
            checkpoint_power *= 2
        if checkpoint_power == 1:
            logger.info("checkpoint doesn't exist")
            return
        iteration_values = (
            self._configuration.checkpoint_persist_modulus**checkpoint_power
        )
        checkpoint_iteration = iteration_values[
            bisect.bisect_left(
                iteration_values,
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
        assert not self._checkpoint_exists(self._i)
        extensions.save_checkpoint(
            self._model,
            self._optimizer,
            self._i,
            out=self._configuration.checkpoint_path(self._i),
        )
        with open(self._configuration.checkpoint_written_path(self._i), "wb") as f:
            pass

    def _set_annealed_learning_rate(self):
        annealing_configuration = self._configuration.annealing_configuration
        self._optimizer.defaults["lr"] = extensions.cosine_learning_rate(
            it=self._i,
            max_learning_rate=annealing_configuration.max_learning_rate,
            min_learning_rate=annealing_configuration.min_learning_rate,
            warmup_iters=annealing_configuration.warmup_iters,
            cosine_cycle_iters=annealing_configuration.cosine_cycle_iters,
        )

    def train(self):
        tokenized_input = np.load(
            self._configuration.tokenized_input_path, mmap_mode="r"
        )
        assert np.max(tokenized_input) < self._configuration.transformer_llm.vocab_size
        logger.info("Checked input file")

        cross_entropy_loss = cross_entropy.CrossEntropyLoss()
        while True:
            start = time.time()
            self._optimizer.zero_grad()
            self._set_annealed_learning_rate()
            (input, target) = extensions.get_batch(
                tokenized_input,
                batch_size=self._configuration.batch_size,
                context_length=self._configuration.context_length,
                device=self._configuration.transformer_llm.device,
            )
            loss = cross_entropy_loss(self._model(input), target=target)
            loss.backward()
            clipped_gradients = extensions.gradient_clipping(
                self._model.parameters(), max_l2_norm=self._configuration.max_l2_norm
            )
            self._optimizer.step()
            logger.info(
                f"Loss: {torch.sum(loss)} time={time.time() - start} clipped_gradients={clipped_gradients}"
            )
            if self._i % self._configuration.checkpoint_persist_modulus == 0:
                self._persist_checkpoint()
