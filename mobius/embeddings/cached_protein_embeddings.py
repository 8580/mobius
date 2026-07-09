#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Cached Protein Embeddings
#
# Drop this file into mobius/embeddings/ alongside protein_embeddings.py.
# To make it importable as `from mobius import CachedProteinEmbedding`, add:
#   - mobius/embeddings/__init__.py:  from .cached_protein_embeddings import CachedProteinEmbedding
#                                      (and add it to __all__)
#   - mobius/__init__.py:             from .embeddings import CachedProteinEmbedding
#                                      (and add it to __all__)
#
# Usage: just swap the constructor used to build `plm` in your notebook -
#   plm = CachedProteinEmbedding(pretrained_model_name='esm1b_t33_650M_UR50S', embedding_type='avg')
# GPLLModel, SequenceGA, Planner etc. don't need to change at all.
#

import numpy as np
import torch

from .protein_embeddings import ProteinEmbedding


class CachedProteinEmbedding(ProteinEmbedding):
    """
    Drop-in replacement for ProteinEmbedding that memoizes tokenization and
    embedding computation per unique sequence.

    Why this matters
    -----------------
    GPLLModel's `_ExactGPLLModel.forward()` calls `self.transformer.embed(tokens)`
    on every single optimizer step of `fit_gpytorch_mll`, and `.fit()` is called
    once per BO round on the *entire* accumulated pool of sequences (which only
    grows round over round). With a frozen encoder - the default, i.e. no
    `layers_to_finetune` - the embedding of a given sequence is mathematically
    identical every time it is computed. Without caching, a fixed set of
    sequences gets pushed through the (potentially very large) transformer
    dozens of times per round just to re-fit GP hyperparameters, and both the
    compute cost and the VRAM footprint of that operation grow every round as
    the pool grows. Caching turns all of that repeated work into O(1) lookups
    after the first time each sequence is seen.

    Two independent caches are kept, both simple dicts on CPU (so they don't
    themselves consume GPU memory):

    - `_token_cache`  : sequence (str)        -> unpadded token id tensor
    - `_embedding_cache` (/ `_probability_cache`) : token-content key -> embedding

    The embedding cache is keyed on the *actual token content* of a row (with
    padding stripped), not on the sequence string directly. This is what lets
    `embed()` benefit from caching even when it's called with pre-tokenized
    batches (as GPLLModel does) rather than through `transform()`.

    IMPORTANT correctness note
    ---------------------------
    Caching is only valid for a *frozen* encoder. If `layers_to_finetune` is
    set (or you later call `.unfreeze()`), embeddings stop being constant
    across calls. This class checks `requires_grad` on the encoder's
    parameters on every `embed()` call and transparently falls back to the
    uncached parent implementation whenever any parameter is trainable, so it
    is always safe to use - it just won't accelerate a fine-tuning run.
    `unfreeze()` also proactively clears the caches, since anything cached
    while frozen is invalid the moment the encoder becomes trainable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._token_cache = {}        # str -> 1D LongTensor (unpadded, incl. special tokens), CPU
        self._embedding_cache = {}    # tuple[int] key -> feature tensor, CPU
        self._probability_cache = {}  # tuple[int] key -> probability tensor, CPU (only if used)

    # ------------------------------------------------------------------
    # Cache bookkeeping
    # ------------------------------------------------------------------
    def __len__(self):
        """Number of unique sequences with a cached embedding."""
        return len(self._embedding_cache)

    def clear_cache(self):
        """Wipe all caches. Call this if the encoder's weights change (e.g.
        after a round of fine-tuning) before relying on caching again."""
        self._token_cache.clear()
        self._embedding_cache.clear()
        self._probability_cache.clear()

    def cache_info(self):
        return {
            'n_tokenized': len(self._token_cache),
            'n_embedded': len(self._embedding_cache),
        }

    @property
    def _is_frozen(self):
        """True only if none of the encoder's parameters require gradients.
        Caching embeddings is only correct while this holds."""
        return not any(p.requires_grad for p in self._model.parameters())

    def unfreeze(self, layers_to_unfreeze=None):
        super().unfreeze(layers_to_unfreeze)
        # Anything cached while frozen is invalid the moment weights can move.
        self.clear_cache()

    # ------------------------------------------------------------------
    # Token-level cache
    # ------------------------------------------------------------------
    def _row_key(self, row):
        """Hashable identity for one tokenized sequence, independent of
        whatever batch-specific padding it currently has: strip the padding
        id and hash what's left (BOS/EOS/residue ids)."""
        ids = row.tolist()
        return tuple(t for t in ids if t != self._padding_token)

    def _pad_batch(self, rows):
        """Re-pads a list of variable-length 1D token tensors into one batch
        tensor, using this model's padding convention."""
        max_len = max(r.shape[0] for r in rows)
        batch = torch.full((len(rows), max_len), self._padding_token, dtype=torch.int64)
        for i, r in enumerate(rows):
            batch[i, :r.shape[0]] = r
        return batch

    def tokenize(self, sequences):
        if not isinstance(sequences, (list, tuple, np.ndarray)):
            sequences = [sequences]
        sequences = list(sequences)

        missing = [s for s in sequences if s not in self._token_cache]
        for seq in missing:
            # Tokenize ONE sequence at a time so no cross-sequence padding
            # contaminates what gets stored; re-uses all of the parent's
            # model-specific logic (ESM alphabet vs. HF tokenizer, extra
            # spaces, padding_length, etc.) unchanged.
            tokens = super().tokenize([seq])
            self._token_cache[seq] = tokens.squeeze(0).detach().cpu()

        rows = [self._token_cache[s] for s in sequences]
        return self._pad_batch(rows).to(self._device)

    # ------------------------------------------------------------------
    # Embedding-level cache
    # ------------------------------------------------------------------
    def embed(self, tokenized_sequences, return_probabilities=False):
        if not self._is_frozen:
            # Encoder is (at least partially) trainable - embeddings are not
            # constant, caching would be silently wrong. Defer to parent.
            return super().embed(tokenized_sequences, return_probabilities)

        rows = list(tokenized_sequences.cpu())
        keys = [self._row_key(r) for r in rows]

        cache = self._probability_cache if return_probabilities else self._embedding_cache
        miss_idx = [i for i, k in enumerate(keys) if k not in cache]

        if miss_idx:
            miss_tokens = self._pad_batch([rows[i] for i in miss_idx]).to(self._device)

            # GPLLModel.fit() puts the (frozen) encoder in .train() mode so
            # dropout etc. are active. For a frozen encoder that just means
            # every call gets a different random mask - which caching would
            # otherwise freeze in place the first time a sequence is seen.
            # Since the weights can't move anyway, force eval() for this
            # forward pass (restoring the prior mode after) so cached
            # embeddings are deterministic, matching best practice for a
            # frozen feature extractor.
            was_training = self._model.training
            self._model.eval()
            try:
                with torch.inference_mode():
                    if return_probabilities:
                        new_features, new_probs = super().embed(miss_tokens, return_probabilities=True)
                        for local_i, global_i in enumerate(miss_idx):
                            self._embedding_cache[keys[global_i]] = new_features[local_i].detach().cpu()
                            self._probability_cache[keys[global_i]] = new_probs[local_i].detach().cpu()
                    else:
                        new_features = super().embed(miss_tokens, return_probabilities=False)
                        for local_i, global_i in enumerate(miss_idx):
                            self._embedding_cache[keys[global_i]] = new_features[local_i].detach().cpu()
            finally:
                if was_training:
                    self._model.train()

        cached_features = [self._embedding_cache[k] for k in keys]
        try:
            features = torch.stack(cached_features).to(self._device)
        except RuntimeError:
            # Variable-length residue embeddings (different sequence lengths) -
            # can't be stacked into one tensor, same as the parent's behaviour.
            features = [f.to(self._device) for f in cached_features]

        if return_probabilities:
            cached_probs = [self._probability_cache[k] for k in keys]
            try:
                probabilities = torch.stack(cached_probs).to(self._device)
            except RuntimeError:
                probabilities = [p.to(self._device) for p in cached_probs]
            return features, probabilities

        return features

    # ------------------------------------------------------------------
    # Convenience: pre-warm the cache in bounded mini-batches
    # ------------------------------------------------------------------
    def add_sequences(self, sequences, batch_size=96):
        """
        Pre-computes and caches tokens + embeddings for `sequences`, processed
        in mini-batches of `batch_size` so a single call never has to run the
        whole (potentially large/growing) pool through the encoder at once.

        Safe and cheap to call repeatedly across BO rounds with the *full*
        accumulated pool each time - sequences already in the cache are
        skipped, so only genuinely new sequences trigger a forward pass.
        """
        sequences = list(sequences)
        new_sequences = [s for s in sequences if s not in self._token_cache]

        for i in range(0, len(new_sequences), batch_size):
            chunk = new_sequences[i:i + batch_size]
            tokens = self.tokenize(chunk)  # fills _token_cache for `chunk`
            self.embed(tokens)             # fills _embedding_cache for `chunk`
