#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - protein language model backends
#
# New file: mobius/embeddings/plm_backends.py
#
# WHY THIS EXISTS
# ---------------
# `ProteinEmbedding.__init__` has exactly two code paths:
#
#     if 'esm' in pretrained_model_name:
#         self._model, alphabet = esm.pretrained.load_model_and_alphabet(...)   # fair-esm
#     else:
#         self._model = AutoModel.from_pretrained(...)                          # HuggingFace
#
# Neither works for ESMC, and the HF path does not work for SaProt either.
# Three concrete problems, all verified against the published packages:
#
# 1. ESMC IS MIS-ROUTED BY THE NAME CHECK.
#    'esmc_300m' contains the substring 'esm', so it takes the fair-esm branch
#    and calls `esm.pretrained.load_model_and_alphabet('esmc_300m')`. That
#    function does not exist in the EvolutionaryScale package - its
#    `esm/pretrained.py` exposes ESMC_300M_202412, ESMC_600M_202412,
#    ESM3_sm_open_v0, load_local_model and register_local_model, and no
#    `load_model_and_alphabet` at all.
#
# 2. `fair-esm` AND `esm` COLLIDE ON THE SAME TOP-LEVEL MODULE NAME.
#    Verified from the wheels: fair-esm 2.0.0 and esm 3.2.3 both install a
#    top-level package called `esm`. ESM2-via-fair-esm and ESMC-via-
#    EvolutionaryScale cannot both be importable in one environment; whichever
#    is installed second wins. No amount of code works around this - see the
#    install notes for the two supported layouts. The backends here import
#    lazily inside `load()` so that importing mobius does not fail merely
#    because the other package is the one installed.
#
# 3. SAPROT IS NOT AN AA-SEQUENCE MODEL.
#    Its vocabulary is structure-aware: every token is an (amino acid, 3Di)
#    pair - 'Ev', 'Vp', 'M#' - for a vocabulary of 446. Feeding it a plain
#    'MEVVQ...' string does not produce the representation it was trained for.
#    The SaProt README is explicit:
#
#        "SaProt (35M and 650M) requires structural (SA token) input for
#         optimal performance. Its AA-only sequence mode works but should be
#         finetuned - its frozen embeddings work only for SA input, not AA
#         sequences!"
#
#    Frozen embeddings are precisely what a GP surrogate consumes here, so the
#    SaProt backend requires a structure string by default and refuses AA-only
#    mode unless you opt in explicitly.
#
# Every backend exposes the same small interface, so `CachedProteinEmbedding`
# stays backend-agnostic:
#
#     .load(device)               -> populates .model / .tokenizer
#     .prepare(sequence)          -> str actually handed to the tokenizer
#     .tokenize(sequences, ...)   -> LongTensor [B, L]
#     .forward(tokens)            -> (hidden [B, L, D], logits [B, L, V] or None)
#     .pad_token_id / .bos_token_id / .eos_token_id
#     .vocabulary_idx             -> indices of the 20 AAs in the logits
#                                    vocabulary, or None if not well defined
#

import re

import numpy as np
import torch


STANDARD_AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


class _PLMBackend:
    """Interface every backend implements."""

    name = 'base'
    #: True when `prepare()` is not the identity, i.e. the string fed to the
    #: tokenizer differs from the amino-acid sequence the caller passed in.
    transforms_sequence = False

    def __init__(self, pretrained_model_name):
        self.pretrained_model_name = pretrained_model_name
        self.model = None
        self.tokenizer = None
        self.device = None

    # -- lifecycle ---------------------------------------------------
    def load(self, device):
        raise NotImplementedError

    # -- sequence preparation ----------------------------------------
    def prepare(self, sequence):
        """Maps a plain amino-acid string to whatever the model consumes.
        Identity for everything except SaProt."""
        return sequence

    # -- tokenization / forward --------------------------------------
    def tokenize(self, sequences, padding_length=None):
        raise NotImplementedError

    def forward(self, tokens):
        """Returns (hidden_states [B, L, D], logits [B, L, V] or None)."""
        raise NotImplementedError

    # -- special tokens ----------------------------------------------
    @property
    def pad_token_id(self):
        raise NotImplementedError

    @property
    def bos_token_id(self):
        raise NotImplementedError

    @property
    def eos_token_id(self):
        raise NotImplementedError

    @property
    def vocabulary_idx(self):
        """Indices of the 20 standard amino acids in the logits vocabulary, in
        STANDARD_AMINO_ACIDS order. None when the vocabulary is not a plain
        amino-acid vocabulary (SaProt)."""
        return None

    # -- convenience -------------------------------------------------
    def parameters(self):
        return self.model.parameters()

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    @property
    def training(self):
        return self.model.training


# =====================================================================
# ESM2 / ESM-1b / ESM-1v via fair-esm
# =====================================================================
class ESM2Backend(_PLMBackend):
    """facebookresearch/esm (PyPI: fair-esm)."""

    name = 'esm2'

    def __init__(self, pretrained_model_name, repr_layer=None):
        super().__init__(pretrained_model_name)
        self.alphabet = None
        self._repr_layer = repr_layer

    def load(self, device):
        import esm  # lazy: see note 2 at the top of this file

        if not hasattr(esm, 'pretrained') or not hasattr(esm.pretrained, 'load_model_and_alphabet'):
            raise ImportError(
                "The installed `esm` package does not provide "
                "`esm.pretrained.load_model_and_alphabet`, which the esm2 backend needs. "
                "You most likely have EvolutionaryScale's `esm` (ESM3/ESMC) installed, "
                "which shadows `fair-esm`. Both ship a top-level module called `esm` and "
                "cannot coexist. Install `fair-esm` for the esm2 backend, or use "
                "backend='esmc'."
            )

        self.device = torch.device(device)
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(self.pretrained_model_name)
        self.model.to(self.device)
        self.model.eval()

        from .protein_embeddings import BatchConverter
        self.tokenizer = BatchConverter(self.alphabet)

        if self._repr_layer is None:
            self._repr_layer = self._infer_repr_layer()

        return self

    def _infer_repr_layer(self):
        """
        Derives the representation layer from the checkpoint name.

        The stock code hardcodes 33 and special-cases only t36 and t30:

            self._repr_layers = 33
            if 't36' in name:   self._repr_layers = 36
            elif 't30' in name: self._repr_layers = 30

        so esm2_t6_8M, esm2_t12_35M and esm2_t48_15B silently ask for a layer
        that does not exist. Parsing `_t<N>_` covers every published
        checkpoint; the model's own `num_layers` is the fallback.
        """
        match = re.search(r'_t(\d+)_', self.pretrained_model_name)
        if match:
            return int(match.group(1))

        num_layers = getattr(self.model, 'num_layers', None)
        if num_layers is not None:
            return int(num_layers)

        raise ValueError(
            f'Cannot determine the representation layer for '
            f'{self.pretrained_model_name!r}. Pass repr_layer=<int> explicitly.'
        )

    @property
    def repr_layer(self):
        return self._repr_layer

    def tokenize(self, sequences, padding_length=None):
        if padding_length is None:
            padding, max_length = 'longest', None
        else:
            padding, max_length = 'max_length', padding_length

        _, _, tokens = self.tokenizer([('sequence', s) for s in sequences],
                                      padding=padding, truncation=False,
                                      max_length=max_length)
        return tokens.to(self.device)

    def forward(self, tokens):
        results = self.model(tokens, repr_layers=[self._repr_layer])
        hidden = results['representations'][self._repr_layer]
        logits = results.get('logits')
        return hidden, logits

    @property
    def pad_token_id(self):
        return self.alphabet.padding_idx

    @property
    def bos_token_id(self):
        return self.alphabet.cls_idx

    @property
    def eos_token_id(self):
        return self.alphabet.eos_idx

    @property
    def vocabulary_idx(self):
        toks = self.alphabet.all_toks
        return np.array([toks.index(aa) for aa in STANDARD_AMINO_ACIDS if aa in toks])


# =====================================================================
# ESMC via EvolutionaryScale esm
# =====================================================================
class ESMCBackend(_PLMBackend):
    """
    EvolutionaryScale ESMC (PyPI: esm >= 3).

    Notes
    -----
    * `ESMC.from_pretrained` casts the model to bfloat16 on any non-CPU device.
      bfloat16 carries ~3 decimal digits, which is fine inside the model but
      poor as GP input features, so `forward()` casts hidden states back to
      float32. The cache therefore always stores float32, and results do not
      silently change between a CPU and a GPU run.
    * The documented path (`client.encode(ESMProtein(...))` then
      `client.logits(...)`) handles one protein per call. Batching goes through
      `model._tokenize(list_of_str)` and `model(sequence_tokens=...)`, which is
      what is used here - batching is the entire point of this exercise.
    """

    name = 'esmc'

    def load(self, device):
        try:
            from esm.models.esmc import ESMC  # lazy: see note 2 at the top
        except ImportError as exc:
            raise ImportError(
                "Could not import `esm.models.esmc.ESMC`, which the esmc backend needs. "
                "Install EvolutionaryScale's package (`pip install esm`). It shares the "
                "top-level module name `esm` with `fair-esm`, so the two cannot be "
                "installed side by side."
            ) from exc

        self.device = torch.device(device)
        self.model = ESMC.from_pretrained(self.pretrained_model_name, device=self.device)
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        return self

    def tokenize(self, sequences, padding_length=None):
        # ESMC pads to the longest sequence in the batch; honour an explicit
        # padding_length by right-padding afterwards.
        tokens = self.model._tokenize(list(sequences))

        if padding_length is not None and tokens.shape[1] < padding_length:
            pad = torch.full((tokens.shape[0], padding_length - tokens.shape[1]),
                             self.pad_token_id, dtype=tokens.dtype, device=tokens.device)
            tokens = torch.cat([tokens, pad], dim=1)

        return tokens.to(self.device)

    def forward(self, tokens):
        output = self.model(sequence_tokens=tokens)
        hidden = output.embeddings.float()          # bfloat16 -> float32
        logits = output.sequence_logits
        if logits is not None:
            logits = logits.float()
        return hidden, logits

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self):
        return self.tokenizer.cls_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def vocabulary_idx(self):
        vocab = self.tokenizer.get_vocab()   # token -> id
        return np.array([vocab[aa] for aa in STANDARD_AMINO_ACIDS if aa in vocab])


# =====================================================================
# SaProt via HuggingFace
# =====================================================================
class SaProtBackend(_PLMBackend):
    """
    westlake-repl/SaProt - structure-aware vocabulary (AA + 3Di).

    SaProt does not consume amino-acid strings. Every token is an (AA, 3Di)
    pair: sequence 'MEV' with 3Di string 'dvp' tokenizes as 'Md', 'Ev', 'Vp'.
    '#' is the wildcard 3Di symbol, used by the authors for low-pLDDT regions
    and, here, for AA-only mode.

    Structure handling in a BO loop
    -------------------------------
    A design campaign mutates a fixed scaffold, so the backbone - and hence the
    3Di string - is essentially constant across variants. Pass the wild-type
    3Di string once as `structure_sequence` and it is zipped with each variant.
    This mirrors SaProt's own `predict_mut`, which holds the structure fixed
    and substitutes the amino-acid letter.

    That is an approximation: good for point mutations on a stable scaffold,
    poor if your designs change the fold. It is also the only tractable option
    inside a GA loop, since folding every candidate would cost far more than
    the PLM forward pass this caching exists to avoid.

    Getting the 3Di string
    ----------------------
        from utils.foldseek_util import get_struc_seq
        seq, foldseek_seq, combined = get_struc_seq('bin/foldseek', 'wt.pdb', ['A'])['A']
        # `foldseek_seq` is the 3Di string to pass as structure_sequence

    AA-only mode
    ------------
    `structure_sequence=None, allow_sequence_only=True` uses '#' at every
    position. The SaProt authors state frozen 35M/650M embeddings do not work
    this way. If you need AA-only, use SaProt_1.3B_AF2 (documented to handle
    AA-only well) or ESM2/ESMC. This backend raises unless you opt in.
    """

    name = 'saprot'
    transforms_sequence = True

    def __init__(self, pretrained_model_name, structure_sequence=None,
                 allow_sequence_only=False):
        super().__init__(pretrained_model_name)
        self.structure_sequence = structure_sequence
        self.allow_sequence_only = allow_sequence_only

        if structure_sequence is None and not allow_sequence_only:
            raise ValueError(
                "SaProt needs a 3Di structure string. Its frozen embeddings are only "
                "meaningful for structure-aware (SA) input - the authors state that for "
                "the 35M and 650M checkpoints, frozen AA-only embeddings do not work. "
                "Pass structure_sequence='<3Di string for your scaffold>' (from foldseek), "
                "or set allow_sequence_only=True to force AA-only mode with '#' wildcards "
                "(only advisable for SaProt_1.3B_*), or use backend='esm2' / 'esmc'."
            )

    def load(self, device):
        from transformers import EsmTokenizer, EsmForMaskedLM

        self.device = torch.device(device)
        self.tokenizer = EsmTokenizer.from_pretrained(self.pretrained_model_name)
        self.model = EsmForMaskedLM.from_pretrained(self.pretrained_model_name)
        self.model.to(self.device)
        self.model.eval()
        return self

    def prepare(self, sequence):
        """Interleaves an amino-acid string with the fixed 3Di string."""
        if self.structure_sequence is None:
            return ''.join(f'{aa}#' for aa in sequence)

        if len(sequence) != len(self.structure_sequence):
            raise ValueError(
                f'Sequence length ({len(sequence)}) does not match the structure string '
                f'length ({len(self.structure_sequence)}). SaProt needs one 3Di symbol '
                f'per residue; indels change the backbone and need a new structure string.'
            )

        return ''.join(f'{aa}{s}' for aa, s in zip(sequence, self.structure_sequence))

    def tokenize(self, sequences, padding_length=None):
        if padding_length is None:
            kwargs = {'padding': 'longest'}
        else:
            kwargs = {'padding': 'max_length', 'max_length': padding_length}

        output = self.tokenizer(list(sequences), add_special_tokens=True,
                                return_tensors='pt', truncation=False, **kwargs)
        return output['input_ids'].to(self.device)

    def forward(self, tokens):
        # EsmForMaskedLM returns logits only unless hidden states are requested.
        results = self.model(input_ids=tokens, output_hidden_states=True)
        return results.hidden_states[-1], results.logits

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self):
        return self.tokenizer.cls_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def vocabulary_idx(self):
        """
        SaProt's vocabulary is (AA, 3Di) pairs, so there is no single token per
        amino acid, and per-residue amino-acid probabilities are undefined
        without also fixing the 3Di symbol. Returning None makes
        `return_probabilities=True` raise a clear error rather than silently
        indexing whatever tokens happen to be at those positions.
        Use `amino_acid_token_ids(symbol)` for a structurally-conditioned set.
        """
        return None

    def amino_acid_token_ids(self, structure_symbol):
        """Token ids for the 20 amino acids paired with one fixed 3Di symbol."""
        vocab = self.tokenizer.get_vocab()
        return np.array([vocab[f'{aa}{structure_symbol}'] for aa in STANDARD_AMINO_ACIDS
                         if f'{aa}{structure_symbol}' in vocab])


# =====================================================================
# Factory
# =====================================================================
def resolve_backend(pretrained_model_name, backend='auto', **kwargs):
    """
    Builds the right backend for a checkpoint name.

    Parameters
    ----------
    pretrained_model_name : str
    backend : {'auto', 'esm2', 'esmc', 'saprot'}, default 'auto'
        'auto' dispatches on the name. A plain substring test for 'esm' cannot
        separate ESM2 from ESMC ('esmc_300m' contains 'esm') - that is exactly
        the bug in the stock code - so ESMC is matched first and more
        specifically here.
    **kwargs
        Backend-specific: `repr_layer` (esm2); `structure_sequence`,
        `allow_sequence_only` (saprot).
    """
    backend = (backend or 'auto').lower()
    name = pretrained_model_name.lower()

    if backend == 'auto':
        if 'esmc' in name:
            backend = 'esmc'
        elif 'saprot' in name:
            backend = 'saprot'
        elif 'esm' in name:
            backend = 'esm2'
        else:
            raise ValueError(
                f"Cannot infer a backend for {pretrained_model_name!r}. Pass "
                f"backend='esm2' | 'esmc' | 'saprot' explicitly. Generic HuggingFace "
                f"models remain supported by the stock ProteinEmbedding class."
            )

    if backend == 'esm2':
        return ESM2Backend(pretrained_model_name, repr_layer=kwargs.get('repr_layer'))
    if backend == 'esmc':
        return ESMCBackend(pretrained_model_name)
    if backend == 'saprot':
        return SaProtBackend(pretrained_model_name,
                             structure_sequence=kwargs.get('structure_sequence'),
                             allow_sequence_only=kwargs.get('allow_sequence_only', False))

    raise ValueError(f"Unknown backend {backend!r}. Expected 'esm2', 'esmc' or 'saprot'.")
