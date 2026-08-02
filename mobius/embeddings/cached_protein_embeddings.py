#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Cached Protein Embeddings (v2)
#
# Drop-in replacement for mobius/embeddings/cached_protein_embeddings.py.
#
# What changed vs. v1 (the version currently on the develop branch):
#
#   1. CHUNKED CACHE-MISS FORWARD PASSES  <-- the actual memory fix
#      v1 pushed *every* cache miss through the encoder in a single forward
#      pass. During SequenceGA's first generation the population is ~n_pop
#      (default 500) freshly-sampled sequences, all of which miss, so v1 ran
#      one ESM2-650M forward with batch=500. Peak activation memory for that
#      single call is what you are still seeing. v2 caps it at
#      `embed_batch_size` (default 64) regardless of how many misses arrive.
#
#   2. BATCHED TOKENIZATION
#      v1 tokenized misses one sequence at a time (to avoid cross-sequence
#      padding contamination). For a 500-strong GA population that is 500
#      separate tokenizer calls per generation. v2 tokenizes in batches and
#      strips padding per row when storing, which is equivalent but much
#      faster.
#
#   3. BOUNDED CACHE (optional)
#      The cache is unbounded in v1. For a 4-position GB1-style design the
#      reachable space is 20^4 = 160k sequences; at 1280 floats each that is
#      ~820 MB of host RAM if the GA explores widely. `max_cache_size` adds
#      LRU eviction, and `pin_sequences()` protects the GP's training set
#      from ever being evicted (it is re-read on every single GP call).
#
#   4. torch.no_grad() INSTEAD OF torch.inference_mode()
#      Tensors created under inference_mode are "inference tensors" and carry
#      permanent autograd restrictions. Stacking them happens to produce a
#      normal tensor, so v1 works, but the variable-length code path returns
#      cached tensors directly and could hand an inference tensor to the
#      kernel during fitting. no_grad gives the same graph-suppression memory
#      benefit without that failure mode.
#
#   5. PADDING-LENGTH FIDELITY (correctness fix)
#      v1's `_pad_batch` always padded to the longest row in the batch. If
#      `padding_length` was set on the embedder, the parent would instead pad
#      to that fixed length. With embedding_type='residue' (flattened output)
#      that changes the feature dimension. v2 honours `padding_length`.
#
#   6. INSTRUMENTATION
#      `cache_info()` now reports hits, misses, forward passes and the
#      largest batch actually sent to the encoder, so you can verify the
#      memory fix is doing what it claims rather than taking it on faith.
#
# Everything else - the public API, return shapes, the frozen-encoder safety
# guard, the eval()-forcing during cache misses - is unchanged, so this stays
# a drop-in replacement for both ProteinEmbedding and v1.
#

from collections import OrderedDict

import numpy as np
import torch

from .protein_embeddings import ProteinEmbedding


class CachedProteinEmbedding(ProteinEmbedding):
    """
    ProteinEmbedding that memoizes tokenization and embedding per unique
    sequence, and bounds the size of every forward pass it makes.

    Why this matters
    -----------------
    gpytorch's `ExactGP.__call__` does something non-obvious in eval mode: it
    concatenates the stored training inputs with the test inputs and runs
    `forward()` on the *joint* tensor. So every `GPLLModel.predict()` call -
    which SequenceGA triggers once per GA generation, via
    `Problem._evaluate` -> `acq_fun.forward` -> `surrogate_model.predict` -
    reaches `transformer.embed()` with (n_train + n_test) sequences, not just
    the n_test trial sequences. Without a cache that is a full re-embedding of
    the entire accumulated training pool on every generation; with a cache the
    training rows are free lookups and only genuinely-new GA candidates cost a
    forward pass.

    Caching alone does not bound *peak* memory though, because the misses
    still have to be embedded, and v1 embedded them all at once. That is what
    `embed_batch_size` fixes here.

    Parameters
    ----------
    embed_batch_size : int, default 64
        Maximum number of sequences sent to the encoder in a single forward
        pass. This is the knob that controls peak activation memory. Lower it
        if you are still tight on VRAM; raise it if you have headroom and want
        throughput. It does not affect results, only memory/speed.
    max_cache_size : int or None, default None
        Maximum number of unique sequences to retain. None means unbounded
        (v1 behaviour). When set, least-recently-used entries are evicted,
        except those protected by `pin_sequences()`.
    **kwargs
        Passed through to ProteinEmbedding unchanged.

    Correctness note
    ----------------
    Caching is only valid for a *frozen* encoder. `embed()` checks
    `requires_grad` on the encoder's parameters on every call and falls back
    to the uncached parent implementation whenever any parameter is trainable,
    so it is always safe to use - it just stops accelerating a fine-tuning
    run. `unfreeze()` additionally clears the caches.
    """

    def __init__(self, *args, embed_batch_size=64, max_cache_size=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.embed_batch_size = int(embed_batch_size)
        self.max_cache_size = max_cache_size

        self._token_cache = OrderedDict()        # str -> 1D int64 tensor (unpadded), CPU
        self._embedding_cache = OrderedDict()    # bytes key -> feature tensor, CPU
        self._probability_cache = OrderedDict()  # bytes key -> probability tensor, CPU
        self._pinned = set()                     # keys exempt from LRU eviction

        self._reset_stats()

    # ------------------------------------------------------------------
    # Bookkeeping / instrumentation
    # ------------------------------------------------------------------
    def _reset_stats(self):
        self._n_hits = 0
        self._n_misses = 0
        self._n_forward_passes = 0
        self._max_forward_batch = 0

    def __len__(self):
        """Number of unique sequences with a cached embedding."""
        return len(self._embedding_cache)

    def clear_cache(self, reset_stats=True):
        """Wipe all caches. Call this if the encoder's weights change."""
        self._token_cache.clear()
        self._embedding_cache.clear()
        self._probability_cache.clear()
        self._pinned.clear()
        if reset_stats:
            self._reset_stats()

    def cache_info(self):
        """
        Returns a dict of cache statistics. `max_forward_batch` is the useful
        one for the memory question: it is the largest batch ever handed to
        the encoder, and should never exceed `embed_batch_size`.
        """
        total = self._n_hits + self._n_misses
        return {
            'n_tokenized': len(self._token_cache),
            'n_embedded': len(self._embedding_cache),
            'n_pinned': len(self._pinned),
            'hits': self._n_hits,
            'misses': self._n_misses,
            'hit_rate': (self._n_hits / total) if total else 0.0,
            'forward_passes': self._n_forward_passes,
            'max_forward_batch': self._max_forward_batch,
            'embed_batch_size': self.embed_batch_size,
        }

    @property
    def _is_frozen(self):
        """Caching embeddings is only correct while no encoder parameter
        requires gradients."""
        return not any(p.requires_grad for p in self._model.parameters())

    def unfreeze(self, layers_to_unfreeze=None):
        super().unfreeze(layers_to_unfreeze)
        # Anything cached while frozen is invalid once weights can move.
        self.clear_cache()

    # ------------------------------------------------------------------
    # Keys, padding, eviction
    # ------------------------------------------------------------------
    def _row_key(self, row):
        """
        Hashable identity for one tokenized sequence, independent of whatever
        batch-specific padding it currently carries. Bytes rather than a tuple
        of ints: same semantics, but much cheaper to build and to hash, which
        matters because this runs on (n_train + n_test) rows every generation.
        """
        row = row.to(torch.int64)
        return row[row != self._padding_token].numpy().tobytes()

    def _pad_batch(self, rows):
        """
        Re-pads a list of variable-length 1D token tensors into one batch
        tensor. Honours `padding_length` when the embedder was configured with
        one, so the parent's feature dimension is reproduced exactly (this
        matters for embedding_type='residue', where the output is flattened).
        """
        longest = max(r.shape[0] for r in rows)
        if self._padding_length is not None:
            max_len = max(int(self._padding_length), longest)
        else:
            max_len = longest

        batch = torch.full((len(rows), max_len), self._padding_token, dtype=torch.int64)
        for i, r in enumerate(rows):
            batch[i, :r.shape[0]] = r
        return batch

    def _store(self, key, features, probabilities=None):
        """Writes one entry to the cache(s) and applies LRU eviction."""
        self._embedding_cache[key] = features.detach().to('cpu', copy=True)
        self._embedding_cache.move_to_end(key)
        if probabilities is not None:
            self._probability_cache[key] = probabilities.detach().to('cpu', copy=True)
            self._probability_cache.move_to_end(key)
        self._evict_if_needed()

    def _evict_if_needed(self):
        if self.max_cache_size is None:
            return
        # Never evict below the pinned set - those are needed on every call.
        while len(self._embedding_cache) > max(self.max_cache_size, len(self._pinned)):
            for key in self._embedding_cache:
                if key not in self._pinned:
                    self._embedding_cache.pop(key, None)
                    self._probability_cache.pop(key, None)
                    break
            else:
                # Everything remaining is pinned; nothing more we can drop.
                break

    def pin_sequences(self, sequences):
        """
        Marks `sequences` as exempt from LRU eviction. Use this for the GP's
        training set: gpytorch re-reads every training row on every single
        predict() call, so evicting them guarantees a re-embed.
        """
        for seq in np.atleast_1d(np.asarray(sequences, dtype=object)).tolist():
            tokens = self._token_cache.get(seq)
            if tokens is not None:
                self._pinned.add(self._row_key(tokens))

    def unpin_all(self):
        self._pinned.clear()

    # ------------------------------------------------------------------
    # Token-level cache
    # ------------------------------------------------------------------
    def tokenize(self, sequences):
        if not isinstance(sequences, (list, tuple, np.ndarray)):
            sequences = [sequences]
        sequences = list(sequences)

        # Deduplicate misses: a GA population can legitimately repeat a
        # sequence, and we only ever want to tokenize each one once.
        missing = list(dict.fromkeys(s for s in sequences if s not in self._token_cache))

        for start in range(0, len(missing), self.embed_batch_size):
            chunk = missing[start:start + self.embed_batch_size]
            # Batch-tokenize, then strip padding per row before storing, so
            # the stored tokens are independent of this batch's padding.
            tokens = super().tokenize(chunk).detach().cpu()
            for seq, row in zip(chunk, tokens):
                row = row.to(torch.int64)
                self._token_cache[seq] = row[row != self._padding_token].clone()

        rows = []
        for s in sequences:
            row = self._token_cache[s]
            self._token_cache.move_to_end(s)
            rows.append(row)

        return self._pad_batch(rows).to(self._device)

    # ------------------------------------------------------------------
    # Embedding-level cache
    # ------------------------------------------------------------------
    def embed(self, tokenized_sequences, return_probabilities=False):
        if not self._is_frozen:
            # Encoder is (at least partially) trainable - embeddings are not
            # constant, so caching would be silently wrong. Defer to parent.
            return super().embed(tokenized_sequences, return_probabilities)

        rows = list(tokenized_sequences.detach().cpu())
        keys = [self._row_key(r) for r in rows]

        cache = self._probability_cache if return_probabilities else self._embedding_cache

        # Unique misses only, preserving first-seen order.
        miss_positions = OrderedDict()
        for i, k in enumerate(keys):
            if k not in cache and k not in miss_positions:
                miss_positions[k] = i
        miss_idx = list(miss_positions.values())

        self._n_hits += len(keys) - len(miss_idx)
        self._n_misses += len(miss_idx)

        if miss_idx:
            # GPLLModel.fit() puts the (frozen) encoder in .train() mode, so
            # dropout is active. For a frozen encoder that only means a fresh
            # random mask per call, which caching would otherwise freeze in
            # place. Force eval() for the forward pass and restore afterwards.
            was_training = self._model.training
            self._model.eval()
            try:
                # THE MEMORY FIX: bounded mini-batches, never one giant call.
                for start in range(0, len(miss_idx), self.embed_batch_size):
                    chunk_idx = miss_idx[start:start + self.embed_batch_size]
                    chunk_tokens = self._pad_batch([rows[i] for i in chunk_idx]).to(self._device)

                    with torch.no_grad():
                        if return_probabilities:
                            new_f, new_p = super().embed(chunk_tokens, return_probabilities=True)
                            for local_i, global_i in enumerate(chunk_idx):
                                self._store(keys[global_i], new_f[local_i], new_p[local_i])
                            del new_f, new_p
                        else:
                            new_f = super().embed(chunk_tokens, return_probabilities=False)
                            for local_i, global_i in enumerate(chunk_idx):
                                self._store(keys[global_i], new_f[local_i])
                            del new_f

                    self._n_forward_passes += 1
                    self._max_forward_batch = max(self._max_forward_batch, len(chunk_idx))
                    del chunk_tokens
            finally:
                if was_training:
                    self._model.train()

        features = self._gather(self._embedding_cache, keys)

        if return_probabilities:
            return features, self._gather(self._probability_cache, keys)

        return features

    def _gather(self, cache, keys):
        """Assembles the requested rows from a cache, refreshing LRU order."""
        out = []
        for k in keys:
            value = cache[k]
            cache.move_to_end(k)
            out.append(value)

        try:
            return torch.stack(out).to(self._device)
        except RuntimeError:
            # Variable-length residue embeddings can't be stacked into one
            # tensor - same behaviour as the parent. clone() so we never hand
            # out a reference into the cache itself.
            return [v.clone().to(self._device) for v in out]

    # ------------------------------------------------------------------
    # Pre-warming
    # ------------------------------------------------------------------
    def add_sequences(self, sequences, batch_size=None, pin=False):
        """
        Pre-computes and caches tokens + embeddings for `sequences` in bounded
        mini-batches, so that a later call which needs all of them at once
        (e.g. the (n_train + n_test) joint tensor gpytorch builds inside
        predict()) is served entirely from cache and triggers no forward pass.

        Cheap to call repeatedly with the full accumulated pool: sequences
        already cached are skipped.

        Parameters
        ----------
        sequences : array-like of str
        batch_size : int or None
            Defaults to `embed_batch_size`.
        pin : bool, default False
            Also protect these sequences from LRU eviction. Use for the GP
            training set.
        """
        batch_size = int(batch_size or self.embed_batch_size)

        sequences = list(np.atleast_1d(np.asarray(sequences, dtype=object)).tolist())
        new_sequences = list(dict.fromkeys(s for s in sequences if s not in self._token_cache))

        for start in range(0, len(new_sequences), batch_size):
            chunk = new_sequences[start:start + batch_size]
            tokens = self.tokenize(chunk)   # fills _token_cache
            self.embed(tokens)              # fills _embedding_cache (chunked internally)
            del tokens

        if pin:
            self.pin_sequences(sequences)
