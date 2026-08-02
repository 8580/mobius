#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Cached Protein Embeddings (v3, multi-backend)
#
# Replaces mobius/embeddings/cached_protein_embeddings.py.
#
# v3 adds ESMC and SaProt support on top of v2's caching/batching. Everything
# v2 did is preserved:
#
#   * per-sequence memoization of tokens and embeddings
#   * cache-miss forward passes chunked at `embed_batch_size` (the memory fix)
#   * batched tokenization
#   * optional bounded LRU cache with pinning
#   * torch.no_grad() rather than inference_mode()
#   * padding_length fidelity
#   * hit/miss/forward-pass instrumentation
#   * automatic fallback to uncached behaviour if the encoder is unfrozen
#
# and v3 adds:
#
#   * `backend=` : 'auto' (default), 'esm2', 'esmc', 'saprot'
#   * one pooling implementation shared by all backends, so 'avg' and
#     'residue' mean the same thing everywhere
#   * a `prepare()` hook so SaProt sees (AA, 3Di) pairs while the caller keeps
#     passing plain amino-acid strings - the GA never needs to know
#   * cache keys namespaced by backend + checkpoint, so two embedders sharing
#     a process can never serve each other's vectors
#   * __getstate__ that drops the cache on pickle, so ray workers do not each
#     receive a copy of it
#
# COMPATIBILITY
# -------------
# The public API is unchanged: tokenize(), embed(), transform(),
# add_sequences(), cache_info(), plus the ProteinEmbedding attributes that
# GPLLModel touches (`.model`, `.device`, `.train()`, `.eval()`, `.freeze()`,
# `.unfreeze()`). It remains a drop-in for ProteinEmbedding and for v2.
#
# For the esm2 backend the numbers are bit-identical to the stock
# ProteinEmbedding - there is a regression test for exactly that.
#

from collections import OrderedDict

import numpy as np
import torch

from .protein_embeddings import ProteinEmbedding
from .plm_backends import resolve_backend, STANDARD_AMINO_ACIDS


class CachedProteinEmbedding(ProteinEmbedding):
    """
    Cached, batched protein embeddings over a pluggable PLM backend.

    Parameters
    ----------
    pretrained_model_name : str
        e.g. 'esm2_t33_650M_UR50D', 'esm1b_t33_650M_UR50S' (esm2 backend);
        'esmc_300m', 'esmc_600m' (esmc backend);
        '/path/to/SaProt_650M_AF2' (saprot backend).
    embedding_type : {'avg', 'residue'}, default 'avg'
    backend : {'auto', 'esm2', 'esmc', 'saprot'}, default 'auto'
        'auto' infers from the checkpoint name, checking 'esmc' and 'saprot'
        before the generic 'esm' substring (which would otherwise swallow
        ESMC - that is the bug in the stock dispatch).
    embed_batch_size : int, default 64
        Max sequences per encoder forward pass. This is the peak-memory knob.
    max_cache_size : int or None, default None
        Max unique sequences retained. None = unbounded. LRU eviction, skipping
        anything protected by `pin_sequences()`.
    padding_length : int or None, default None
    device : str or torch.device or None
    repr_layer : int or None
        esm2 only. Inferred from the checkpoint name when None. (The stock code
        hardcodes 33 and special-cases only t36/t30, so smaller ESM2
        checkpoints silently read a nonexistent layer.)
    structure_sequence : str or None
        saprot only. The 3Di string for the scaffold, from foldseek. Required
        unless `allow_sequence_only=True`.
    allow_sequence_only : bool, default False
        saprot only. Forces AA-only mode with '#' wildcards. The SaProt authors
        state frozen 35M/650M embeddings do not work this way.

    Examples
    --------
    # ESM2 (unchanged behaviour, now cached + batched)
    plm = CachedProteinEmbedding('esm2_t33_650M_UR50D', embed_batch_size=64)

    # ESMC
    plm = CachedProteinEmbedding('esmc_300m', embed_batch_size=64)

    # SaProt with a fixed scaffold structure
    plm = CachedProteinEmbedding('/models/SaProt_650M_AF2',
                                 structure_sequence=foldseek_3di,
                                 embed_batch_size=32)
    """

    def __init__(self, pretrained_model_name='esm2_t33_650M_UR50D',
                 embedding_type='avg', backend='auto',
                 embed_batch_size=64, max_cache_size=None,
                 padding_length=None, device=None,
                 repr_layer=None, structure_sequence=None,
                 allow_sequence_only=False):

        assert embedding_type in ('residue', 'avg'), \
            'Only average (avg) and residue embeddings are supported.'

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # NOTE: ProteinEmbedding.__init__ is deliberately NOT called. It
        # hardcodes the fair-esm / AutoModel dispatch this class exists to
        # replace, and calling it would load a model a second time. Every
        # attribute it would have set is set below, so all inherited methods
        # continue to work.
        self._device = torch.device(device)
        self._pretrained_model_name = pretrained_model_name
        self._embedding_type = embedding_type
        self._padding_length = padding_length
        self._add_extra_space = False
        self._standard_amino_acids = list(STANDARD_AMINO_ACIDS)
        self._layers_to_finetune = None
        self._lora = False

        self._backend = resolve_backend(
            pretrained_model_name, backend=backend,
            repr_layer=repr_layer,
            structure_sequence=structure_sequence,
            allow_sequence_only=allow_sequence_only,
        )
        self._backend.load(self._device)

        self._model = self._backend.model
        self._tokenizer = self._backend.tokenizer
        self._model_type = self._backend.name
        self._bos_token = self._backend.bos_token_id
        self._eos_token = self._backend.eos_token_id
        self._padding_token = self._backend.pad_token_id
        self._vocabulary_idx = self._backend.vocabulary_idx

        self.freeze()
        self.eval()

        # -- caching state ------------------------------------------
        self.embed_batch_size = int(embed_batch_size)
        self.max_cache_size = max_cache_size

        self._token_cache = OrderedDict()        # str -> 1D int64 tensor (unpadded), CPU
        self._embedding_cache = OrderedDict()    # bytes -> feature tensor, CPU
        self._probability_cache = OrderedDict()  # bytes -> probability tensor, CPU
        self._pinned = set()

        # Namespaces cache keys. Two embedders in one process (say an ESM2 one
        # and an ESMC one) must never serve each other's vectors: identical
        # token ids mean different things per backend.
        self._key_prefix = (f'{self._backend.name}:{pretrained_model_name}:'
                            f'{embedding_type}|').encode()

        self._reset_stats()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def backend(self):
        return self._backend

    @property
    def backend_name(self):
        return self._backend.name

    def _reset_stats(self):
        self._n_hits = 0
        self._n_misses = 0
        self._n_forward_passes = 0
        self._max_forward_batch = 0

    def __len__(self):
        return len(self._embedding_cache)

    def clear_cache(self, reset_stats=True):
        self._token_cache.clear()
        self._embedding_cache.clear()
        self._probability_cache.clear()
        self._pinned.clear()
        if reset_stats:
            self._reset_stats()

    def cache_info(self):
        """`max_forward_batch` is the memory-relevant number: the largest batch
        ever handed to the encoder. It must never exceed `embed_batch_size`."""
        total = self._n_hits + self._n_misses
        return {
            'backend': self._backend.name,
            'model': self._pretrained_model_name,
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
        return not any(p.requires_grad for p in self._model.parameters())

    def freeze(self):
        for param in self._model.parameters():
            param.requires_grad = False

    def unfreeze(self, layers_to_unfreeze=None):
        super().unfreeze(layers_to_unfreeze)
        # Anything cached while frozen is invalid once weights can move.
        self.clear_cache()

    # ------------------------------------------------------------------
    # Keys, padding, eviction
    # ------------------------------------------------------------------
    def _row_key(self, row):
        """Backend-namespaced identity for one tokenized sequence, independent
        of this batch's padding. Bytes rather than tuples: same semantics, much
        cheaper to build and hash, and this runs on every row of every call."""
        row = row.to(torch.int64)
        return self._key_prefix + row[row != self._padding_token].numpy().tobytes()

    def _pad_batch(self, rows):
        """Re-pads variable-length token rows into one batch tensor, honouring
        `padding_length` so the feature dimension matches what the backend
        would have produced (this matters for embedding_type='residue', where
        the output is flattened)."""
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
        self._embedding_cache[key] = features.detach().to('cpu', copy=True)
        self._embedding_cache.move_to_end(key)
        if probabilities is not None:
            self._probability_cache[key] = probabilities.detach().to('cpu', copy=True)
            self._probability_cache.move_to_end(key)
        self._evict_if_needed()

    def _evict_if_needed(self):
        if self.max_cache_size is None:
            return
        while len(self._embedding_cache) > max(self.max_cache_size, len(self._pinned)):
            for key in self._embedding_cache:
                if key not in self._pinned:
                    self._embedding_cache.pop(key, None)
                    self._probability_cache.pop(key, None)
                    break
            else:
                break   # everything left is pinned

    def pin_sequences(self, sequences):
        """Exempts `sequences` from LRU eviction. Use for the GP training set:
        gpytorch re-reads every training row on every predict() call."""
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

        # Deduplicate: a GA population legitimately repeats sequences.
        missing = list(dict.fromkeys(s for s in sequences if s not in self._token_cache))

        for start in range(0, len(missing), self.embed_batch_size):
            chunk = missing[start:start + self.embed_batch_size]
            # `prepare` is the identity except for SaProt, where it zips the
            # amino-acid string with the fixed 3Di structure string.
            prepared = [self._backend.prepare(s) for s in chunk]
            tokens = self._backend.tokenize(prepared, padding_length=self._padding_length)
            tokens = tokens.detach().cpu()
            for seq, row in zip(chunk, tokens):
                row = row.to(torch.int64)
                self._token_cache[seq] = row[row != self._padding_token].clone()

        rows = []
        for s in sequences:
            rows.append(self._token_cache[s])
            self._token_cache.move_to_end(s)

        self._warn_if_variable_length(rows)

        return self._pad_batch(rows).to(self._device)

    def _warn_if_variable_length(self, rows):
        """
        Raises a comprehensible error when sequences of differing length are
        used without `padding_length`.

        gpytorch's ExactGP concatenates the stored training tokens with the
        test tokens (`torch.cat([train_input, input], dim=-2)`) before calling
        forward, so both tensors must have the same number of token columns.
        Without a fixed `padding_length` each call pads to the longest sequence
        in *that* call, so a training pool containing a 14-mer and a test batch
        of only 12-mers produce 16 and 14 columns and the cat fails with

            RuntimeError: Sizes of tensors must match except in dimension 0.
            Expected size 16 but got size 14 for tensor number 1 in the list.

        which says nothing about sequence lengths. This is a stock GPLLModel
        limitation, not something caching introduces - the uncached class fails
        identically - but it surfaces the moment you optimize more than one
        scaffold, so it is worth catching here with a message that names the
        fix.
        """
        if self._padding_length is not None or len(rows) < 2:
            return

        lengths = {int(r.shape[0]) for r in rows}
        if len(lengths) > 1:
            raise ValueError(
                f'Sequences of differing tokenized length ({sorted(lengths)}) were passed '
                f'without `padding_length`. GPLLModel cannot handle that: gpytorch '
                f'concatenates the training and test token tensors before the forward '
                f'pass, so they must have the same number of columns, and without a fixed '
                f'padding_length each call pads to its own longest sequence. Set '
                f'padding_length to at least {max(lengths)} (plus room for anything longer '
                f'the GA may propose) when constructing the embedder. This limitation is '
                f'inherited from the stock GPLLModel, which fails the same way with a much '
                f'less obvious error.'
            )

    # ------------------------------------------------------------------
    # Pooling - one implementation for every backend
    # ------------------------------------------------------------------
    def _pool(self, hidden, sequence_mask):
        """Reproduces ProteinEmbedding.embed()'s pooling exactly."""
        if self._embedding_type == 'residue':
            same_length = bool((sequence_mask == sequence_mask[0]).all())
            if same_length:
                selected = hidden[sequence_mask.unsqueeze(-1).expand_as(hidden)]
                n_true = sequence_mask.sum(dim=1).max().item()
                hidden = selected.reshape(hidden.shape[0], n_true, hidden.shape[2])
                return hidden.reshape(hidden.shape[0], hidden.shape[1] * hidden.shape[2])
            return [h[m] for h, m in zip(hidden, sequence_mask)]

        denom = torch.sum(sequence_mask, -1, keepdim=True)
        return torch.sum(hidden * sequence_mask.unsqueeze(-1), dim=1) / denom

    def _pool_probabilities(self, logits, sequence_mask):
        if self._vocabulary_idx is None:
            raise ValueError(
                f"return_probabilities is not supported for the {self._backend.name} "
                f"backend. SaProt's vocabulary is (amino acid, 3Di) pairs, so per-residue "
                f"amino-acid probabilities are undefined without also fixing the 3Di "
                f"symbol. Use backend.amino_acid_token_ids(symbol) to build a "
                f"structurally-conditioned set yourself."
            )

        same_length = bool((sequence_mask == sequence_mask[0]).all())
        if same_length:
            selected = logits[sequence_mask.unsqueeze(-1).expand_as(logits)]
            n_true = sequence_mask.sum(dim=1).max().item()
            logits = selected.reshape(logits.shape[0], n_true, logits.shape[2])
            logits = logits[:, :, self._vocabulary_idx]
        else:
            logits = [l[m][:, self._vocabulary_idx] for l, m in zip(logits, sequence_mask)]

        softmax = torch.nn.Softmax(dim=-1)
        probabilities = [softmax(l) for l in logits]
        try:
            probabilities = torch.stack(probabilities)
        except Exception:
            pass
        return probabilities

    def _raw_embed(self, tokens, return_probabilities=False):
        """Uncached backend forward + pooling."""
        sequence_mask = ((tokens != self._padding_token)
                         & (tokens != self._eos_token)
                         & (tokens != self._bos_token))

        hidden, logits = self._backend.forward(tokens)
        features = self._pool(hidden, sequence_mask)

        if return_probabilities:
            if logits is None:
                raise ValueError(
                    f'Model {self._pretrained_model_name} did not return logits, so '
                    f'probabilities are unavailable. Set return_probabilities=False.'
                )
            return features, self._pool_probabilities(logits, sequence_mask)

        return features

    # ------------------------------------------------------------------
    # Embedding-level cache
    # ------------------------------------------------------------------
    def embed(self, tokenized_sequences, return_probabilities=False):
        if not self._is_frozen:
            # Encoder is (at least partially) trainable, so embeddings are not
            # constant and caching would be silently wrong.
            return self._raw_embed(tokenized_sequences, return_probabilities)

        rows = list(tokenized_sequences.detach().cpu())
        keys = [self._row_key(r) for r in rows]

        cache = self._probability_cache if return_probabilities else self._embedding_cache

        miss_positions = OrderedDict()
        for i, k in enumerate(keys):
            if k not in cache and k not in miss_positions:
                miss_positions[k] = i
        miss_idx = list(miss_positions.values())

        self._n_hits += len(keys) - len(miss_idx)
        self._n_misses += len(miss_idx)

        if miss_idx:
            # GPLLModel.fit() puts the (frozen) encoder in .train() mode, which
            # activates dropout. For a frozen encoder that just means a fresh
            # random mask per call - which caching would otherwise freeze in
            # place. Force eval() for the forward pass, restore afterwards.
            was_training = self._model.training
            self._model.eval()
            try:
                # THE MEMORY FIX: bounded mini-batches, never one giant call.
                for start in range(0, len(miss_idx), self.embed_batch_size):
                    chunk_idx = miss_idx[start:start + self.embed_batch_size]
                    chunk_tokens = self._pad_batch([rows[i] for i in chunk_idx]).to(self._device)

                    with torch.no_grad():
                        if return_probabilities:
                            new_f, new_p = self._raw_embed(chunk_tokens, return_probabilities=True)
                            for local_i, global_i in enumerate(chunk_idx):
                                self._store(keys[global_i], new_f[local_i], new_p[local_i])
                            del new_f, new_p
                        else:
                            new_f = self._raw_embed(chunk_tokens, return_probabilities=False)
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
        out = []
        for k in keys:
            out.append(cache[k])
            cache.move_to_end(k)

        try:
            return torch.stack(out).to(self._device)
        except RuntimeError:
            # Variable-length residue embeddings can't be stacked, same as the
            # parent. clone() so callers never hold a reference into the cache.
            return [v.clone().to(self._device) for v in out]

    # ------------------------------------------------------------------
    # transform / pre-warming
    # ------------------------------------------------------------------
    def transform(self, sequences, return_probabilities=False):
        return self.embed(self.tokenize(sequences), return_probabilities)

    def add_sequences(self, sequences, batch_size=None, pin=False):
        """
        Pre-computes tokens + embeddings in bounded mini-batches, so a later
        call needing all of them at once - such as the (n_train + n_test) joint
        tensor gpytorch builds inside predict() - is served entirely from cache
        and triggers no forward pass.

        Cheap to call repeatedly with the full pool: cached sequences skip.
        """
        batch_size = int(batch_size or self.embed_batch_size)

        sequences = list(np.atleast_1d(np.asarray(sequences, dtype=object)).tolist())
        new_sequences = list(dict.fromkeys(s for s in sequences if s not in self._token_cache))

        for start in range(0, len(new_sequences), batch_size):
            chunk = new_sequences[start:start + batch_size]
            tokens = self.tokenize(chunk)
            self.embed(tokens)
            del tokens

        if pin:
            self.pin_sequences(sequences)

    # ------------------------------------------------------------------
    # Pickling - keep ray from shipping the cache to every worker
    # ------------------------------------------------------------------
    def __getstate__(self):
        """
        SequenceGA dispatches groups to ray workers, and ray pickles the whole
        acquisition function - which reaches the surrogate models, this
        embedder, the PLM weights and the entire cache. Shipping a large cache
        to every worker is worse than the problem this class exists to fix, and
        pointless: a worker's cache additions are discarded when it exits, they
        never travel back.

        So the cache is dropped on pickle. The model is kept, since a worker
        cannot run without it. `CachedSequenceGA` avoids the round trip
        entirely by running groups in one process when a cached embedder is in
        use.
        """
        state = self.__dict__.copy()
        state['_token_cache'] = OrderedDict()
        state['_embedding_cache'] = OrderedDict()
        state['_probability_cache'] = OrderedDict()
        state['_pinned'] = set()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._reset_stats()
