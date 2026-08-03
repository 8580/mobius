#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Gaussian Process Regressor with pLM + embedding cache pre-warming
#
# New file: mobius/surrogate_models/cached_gaussian_process_llm.py
#

import numpy as np

from .gaussian_process_llm import GPLLModel


class CachedGPLLModel(GPLLModel):
    """
    Drop-in replacement for `GPLLModel` that pre-warms a `CachedProteinEmbedding`
    in bounded mini-batches before handing anything to gpytorch.

    The problem it solves
    ---------------------
    gpytorch's `ExactGP.__call__` concatenates the stored training inputs with
    the test inputs in eval mode and evaluates `forward()` on the joint tensor
    (see `_get_test_prior_mean_and_covariances` in gpytorch/models/exact_gp.py:
    `full_inputs.append(torch.cat([train_input, input], dim=-2))`). So every
    `predict()` call reaches `transformer.embed()` with (n_train + n_test)
    sequences.

    SequenceGA calls `predict()` once per GA generation - via
    `Problem._evaluate` -> `acq_fun.forward` -> `surrogate_model.predict` - so
    across a single `Planner.recommand()` that is dozens of calls, each one
    presenting the whole training pool plus the whole GA population (n_pop
    defaults to 500) to the encoder in one tensor.

    What this class does
    --------------------
    Before delegating to the parent, it walks the sequences through
    `CachedProteinEmbedding.add_sequences()` in `embed_batch_size` chunks.
    By the time gpytorch builds its (n_train + n_test) joint tensor and calls
    `embed()` on it, every row is already cached, so that call performs zero
    encoder forward passes and simply gathers rows. Peak activation memory is
    then set by `embed_batch_size`, not by the population size or the size of
    the accumulated training pool.

    It also pins the training sequences in the cache, because gpytorch re-reads
    every training row on every single `predict()` call - letting LRU evict
    them would guarantee a re-embed on the next generation.

    Degrades gracefully: if `pretrained_model` is a plain `ProteinEmbedding`
    (no cache), pre-warming is skipped and behaviour is identical to
    `GPLLModel`.

    Usage
    -----
    plm = CachedProteinEmbedding(pretrained_model_name='esm1b_t33_650M_UR50S',
                                 embedding_type='avg', embed_batch_size=64)
    gpmodel = CachedGPLLModel(kernel=RBFKernel(), pretrained_model=plm,
                              noise_prior=NormalPrior(0, 1))
    # ...then Planner / SequenceGA / ExpectedImprovement as usual.
    """

    def __init__(self, *args, prewarm_batch_size=None, **kwargs):
        """
        Parameters
        ----------
        prewarm_batch_size : int or None, default None
            Chunk size used when pre-warming. Defaults to the embedder's own
            `embed_batch_size`.
        **args, **kwargs
            Passed to `GPLLModel` unchanged.
        """
        super().__init__(*args, **kwargs)
        self._prewarm_batch_size = prewarm_batch_size

    # ------------------------------------------------------------------
    def _prewarm(self, sequences, pin=False):
        """
        Populates the embedding cache for `sequences` in bounded mini-batches.
        No-op when the embedder has no cache, or when the input is not a
        sequence array (e.g. precomputed feature vectors).
        """
        plm = self._pretrained_model

        if not hasattr(plm, 'add_sequences'):
            # Plain ProteinEmbedding - nothing to pre-warm.
            return

        sequences = np.asarray(sequences)

        # Only str-like inputs are tokenizable; anything else is passed
        # straight through to the parent, which will handle (or reject) it.
        if sequences.dtype.kind not in ('O', 'U', 'S'):
            return

        plm.add_sequences(sequences.ravel(),
                          batch_size=self._prewarm_batch_size,
                          pin=pin)

    # ------------------------------------------------------------------
    def fit(self, X_train, y_train, y_noise=None):
        # Pin: these rows are re-read by gpytorch on every subsequent
        # predict() call, so they must never be evicted.
        self._prewarm(X_train, pin=True)
        return super().fit(X_train, y_train, y_noise)

    def predict(self, X_test, y_noise=None):
        self._prewarm(X_test, pin=False)
        return super().predict(X_test, y_noise)

    # ------------------------------------------------------------------
    def cache_info(self):
        """Convenience passthrough to the embedder's cache statistics."""
        plm = self._pretrained_model
        return plm.cache_info() if hasattr(plm, 'cache_info') else {}
