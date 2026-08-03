#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - cache-aware SequenceGA
#
# New file: mobius/optimizers/cached_sequence_ga.py
#

import numpy as np

from .genetic_algorithm import SequenceGA


def _iter_cached_embedders(acquisition_function):
    """
    Yields every distinct CachedProteinEmbedding reachable from an acquisition
    function, i.e. `acq._surrogate_models[i]._pretrained_model`.

    Deduplicated by identity, so a shared embedder (the recommended multi-
    objective setup) is yielded once, not once per objective.
    """
    seen = {}
    models = getattr(acquisition_function, '_surrogate_models', None) or []

    for surrogate in models:
        plm = getattr(surrogate, '_pretrained_model', None)
        if plm is not None and hasattr(plm, 'add_sequences') and id(plm) not in seen:
            seen[id(plm)] = plm

    return list(seen.values())


class CachedSequenceGA(SequenceGA):
    """
    Drop-in replacement for `SequenceGA` that is aware of
    `CachedProteinEmbedding`. Works for single-objective ('GA') and
    multi-objective ('NSGA2', 'SMSEMOA', ...) runs alike - the caching is
    orthogonal to the algorithm.

    Two problems it fixes
    ---------------------

    1. COLD CACHE AT THE START OF EVERY GA RUN.
       `Planner.recommand()` calls `SequenceGA.run()`, which immediately starts
       evaluating populations. The first generation is a full population of
       fresh candidates (n_pop, default 500) plus the training pool that
       gpytorch concatenates onto every predict() - all of it a cache miss at
       once. Pre-warming the input sequences before the GA starts means the
       training half is already resident, so the first generation only pays for
       genuinely new candidates.

    2. RAY WORKERS EACH GET A COPY OF THE MODEL AND CACHE.
       When the input contains more than one scaffold/design group,
       `SequenceGA.run()` dispatches them to ray:

           refs = [parallel_ga.remote(seq_gao, sequences[seq_ids], scores[seq_ids],
                                      acquisition_function, designs[name], filters)
                   for name, seq_ids in group_indices.items()]

       `acquisition_function` reaches the surrogate models, the embedder, the
       PLM weights and the cache, so ray pickles all of it once per group. For
       a 650M-parameter model that is hundreds of MB per worker, and each
       worker starts from a cold cache and throws its additions away on exit -
       they never come back to the parent.

       When a cached embedder is in use, `parallel='auto'` (the default) runs
       the groups sequentially in this process instead, so they share one warm
       cache and the model is never serialized. Given how high the cross-group
       hit rate usually is (groups differ by scaffold but share a training
       pool), that is normally faster in wall-clock terms as well as far
       lighter on memory.

    Parameters
    ----------
    parallel : {'auto', True, False}, default 'auto'
        'auto'  - serial when a cached embedder is detected, ray otherwise.
        False   - always serial.
        True    - always ray. The cache still helps within each worker, and
                  `CachedProteinEmbedding.__getstate__` drops the cache before
                  pickling so at least the cache is not copied, but the model
                  still is. Use only if the groups are genuinely heavy and you
                  have the memory.
    prewarm : bool, default True
        Pre-warm and pin the input sequences before the GA starts.
    **kwargs
        Passed to `SequenceGA` unchanged (algorithm, n_gen, n_pop, period,
        design_protocol_filename, ...).

    Examples
    --------
    # single-objective
    optimizer = CachedSequenceGA(algorithm='GA', period=15,
                                 design_protocol_filename='design.yaml')

    # multi-objective - identical, only the algorithm changes
    optimizer = CachedSequenceGA(algorithm='SMSEMOA', period=15)
    """

    def __init__(self, *args, parallel='auto', prewarm=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._parallel = parallel
        self._prewarm = prewarm
        self._last_cache_info = {}

    # ------------------------------------------------------------------
    def run(self, sequences, scores, acquisition_function):
        sequences = np.asarray(sequences)
        scores = np.asarray(scores)

        embedders = _iter_cached_embedders(acquisition_function)

        # -- 1. pre-warm ------------------------------------------------
        if self._prewarm and embedders:
            for plm in embedders:
                # pin: gpytorch re-reads every training row on every predict()
                # call, so these must survive LRU eviction for the whole run.
                plm.add_sequences(sequences.ravel(), pin=True)

        # -- 2. choose serial vs ray -----------------------------------
        if self._parallel == 'auto':
            force_serial = bool(embedders)
        elif self._parallel:
            force_serial = False
        else:
            force_serial = True

        if force_serial:
            # n_process = 1 is not enough on its own: SequenceGA.run() branches
            # on `len(groups) == 1`, so any multi-group input still goes to
            # ray. `_run_serial` reproduces the dispatch loop in-process.
            results = self._run_serial(sequences, scores, acquisition_function)
        else:
            results = super().run(sequences, scores, acquisition_function)

        if embedders:
            self._last_cache_info = embedders[0].cache_info()

        return results

    # ------------------------------------------------------------------
    def _run_serial(self, sequences, scores, acquisition_function):
        """
        Same as `SequenceGA.run()` but every group is optimized in this
        process, against one shared warm cache, with no pickling of the model.

        The group-splitting logic is imported from the stock module rather than
        reimplemented, so this tracks upstream changes to how designs and
        scaffolds are parsed.
        """
        from .genetic_algorithm import (
            SerialPolymerGA, SerialBioPolymerGA,
            _generate_design_protocol_from_polymers,
            _load_polymer_design_from_config, _load_biopolymer_design_from_config,
            _load_filters_from_config,
            _prepare_polymers, _prepare_biopolymers,
            _group_polymers_by_scaffold, _group_biopolymers_by_design,
        )
        from ..utils import guess_input_formats

        if self._optimization_type == 'single':
            if scores.shape[1] != 1:
                raise ValueError('Only one score per sequence is allowed for '
                                 'single-objective optimization.')
        else:
            if scores.shape[1] < 2:
                raise ValueError('Only one score per sequence provided for '
                                 'multi-objective optimization.')

        sequence_formats = guess_input_formats(sequences)
        unique_formats = np.unique(sequence_formats)

        if len(unique_formats) > 1:
            raise ValueError(f'The input contains sequences in multiple formats: {unique_formats}')
        if 'unknown' in unique_formats:
            raise ValueError('The input contains sequences in an unknown format '
                             '(only HELM or FASTA allowed).')

        if unique_formats[0] == 'HELM':
            serial_seq_ga = SerialPolymerGA

            if self._parameters['design_protocol_filename'] is None:
                design_protocol = _generate_design_protocol_from_polymers(sequences)
                designs = _load_polymer_design_from_config(design_protocol)
                filters = {}
            else:
                designs = _load_polymer_design_from_config(self._parameters['design_protocol_filename'])
                filters = _load_filters_from_config(self._parameters['design_protocol_filename'])

            sequences, scores = _prepare_polymers(sequences, scores, acquisition_function, designs)
            groups, group_indices = _group_polymers_by_scaffold(sequences, return_index=True)
        else:
            serial_seq_ga = SerialBioPolymerGA

            if self._parameters['design_protocol_filename'] is None:
                raise ValueError('A design protocol must be provided for biopolymers optimization.')

            designs = _load_biopolymer_design_from_config(self._parameters['design_protocol_filename'])
            filters = _load_filters_from_config(self._parameters['design_protocol_filename'])

            sequences, scores = _prepare_biopolymers(sequences, scores, acquisition_function, designs)
            groups, group_indices = _group_biopolymers_by_design(sequences, designs, return_index=True)

        seq_gao = serial_seq_ga(**self._parameters)

        if len(groups) == 1:
            indices = list(group_indices.values())[0]
            design = designs[list(group_indices.keys())[0]]
            return seq_gao.run(sequences[indices], scores[indices],
                               acquisition_function, design, filters)

        # Multiple groups, one process, one shared cache.
        return [seq_gao.run(sequences[seq_ids], scores[seq_ids],
                            acquisition_function, designs[name], filters)
                for name, seq_ids in group_indices.items()]

    # ------------------------------------------------------------------
    def cache_info(self):
        """Cache statistics as of the end of the last `run()`."""
        return self._last_cache_info
