Mobius
======

Docker build & run
------------------
```bash
docker build -f GPU.Dockerfile -t mobius-gpu .

docker run -it mobius-gpu /bin/bash
docker run --runtime=nvidia --gpus all -it mobius-gpu /bin/bash
```

# for real
----------
```bash
docker network create --internal privatenet
docker run --runtime=nvidia --gpus all -v $(pwd)/models:/home/app/models -v $(pwd)/data:/home/app/data --env HF_HOME='/home/app/models' -it mobius-gpu python3 ./data/run.py
docker run --network=privatenet --runtime=nvidia --gpus all -v $(pwd)/models:/home/app/models -v $(pwd)/data:/home/app/data --env HF_HOME='/home/app/models' -it mobius-gpu /bin/bash
```


```bash
docker run --gpus all -it --rm -p 8888:8888 -v "$(pwd)":/workspace mobius-gpu
```

# Cache-aware PLM embeddings for mobius - ESM2, ESMC, SaProt

Complete install for single- and multi-objective optimisation with cached, batched protein language model embeddings.

This supersedes the earlier versions (v2).

## `fair-esm` and `esm` cannot coexist

Verified from the published wheels: **fair-esm 2.0.0 and esm 3.2.3 both install a top-level package called `esm`.** Whichever you install second shadows the other.

* ESM2 needs `esm.pretrained.load_model_and_alphabet` → that is **fair-esm**.
* ESMC needs `esm.models.esmc.ESMC` → that is **EvolutionaryScale's esm**.
  (Its `esm/pretrained.py` exposes `ESMC_300M_202412`, `ESMC_600M_202412`,
  `ESM3_sm_open_v0`, `load_local_model`, `register_local_model` — and no
  `load_model_and_alphabet` at all.)

So there are two supported environments:

```bash
# Environment A — ESM2 (and SaProt, which is pure HuggingFace)
pip install fair-esm transformers

# Environment B — ESMC (and SaProt)
pip install esm transformers
```

SaProt works in either, since it only needs `transformers`.

The backends import lazily inside `load()`, so `import mobius` succeeds whichever package is present; you only get an error, naming the conflict, if you actually ask for a backend the environment cannot serve.

If you need ESM2 and ESMC in one workflow, run them in separate environments and exchange embeddings via files, or use two conda envs behind a queue. There is no in-process fix.


# Usage of CachedProteinEmbedding, CachedGPLLModel and CachedSequenceGA

## Single objective

```python
from mobius import CachedProteinEmbedding, CachedGPLLModel, CachedSequenceGA
from mobius import ExpectedImprovement, Planner
from gpytorch.kernels import RBFKernel
from gpytorch.priors import NormalPrior

plm = CachedProteinEmbedding('esm2_t33_650M_UR50D',      # or 'esmc_300m'
                             embedding_type='avg',
                             embed_batch_size=64,        # peak-VRAM knob
                             max_cache_size=50_000)      # host-RAM knob

gpmodel = CachedGPLLModel(kernel=RBFKernel(), pretrained_model=plm,
                          noise_prior=NormalPrior(0, 1))
ei = ExpectedImprovement(gpmodel, maximize=True)
optimizer = CachedSequenceGA(algorithm='GA', period=15,
                             design_protocol_filename='design.yaml')
ps = Planner(ei, optimizer)
```

## Multiple objectives — one shared embedder

```python
plm = CachedProteinEmbedding('esmc_300m', embed_batch_size=64)

gp_a = CachedGPLLModel(kernel=RBFKernel(), pretrained_model=plm,
                       noise_prior=NormalPrior(0, 1))
gp_b = CachedGPLLModel(kernel=RBFKernel(), pretrained_model=plm,   # SAME plm
                       noise_prior=NormalPrior(0, 1))

acq = ExpectedImprovement([gp_a, gp_b], maximize=[False, False])
optimizer = CachedSequenceGA(algorithm='SMSEMOA', period=15,
                             design_protocol_filename='design.yaml')
ps = Planner(acq, optimizer)
```

Passing the **same** `plm` to every surrogate is the point: the acquisition
function fits and queries all of them on identical sequence sets, so encoder
cost becomes independent of the number of objectives. Measured: two objectives
cost exactly one encoder pass per unique sequence, not two.

## SaProt

```python
# 1. get the 3Di string for your scaffold, once
from utils.foldseek_util import get_struc_seq
seq, foldseek_seq, combined = get_struc_seq('bin/foldseek', 'wt.pdb', ['A'])['A']

# 2. hand it to the embedder; the GA still passes plain AA strings
plm = CachedProteinEmbedding('/models/SaProt_650M_AF2',
                             structure_sequence=foldseek_seq,
                             embed_batch_size=32)


## Known limitation: variable-length sequences need `padding_length`

`GPLLModel` cannot handle sequences of differing length unless you fix
`padding_length`. gpytorch concatenates the stored training tokens with the
test tokens before the forward pass:

```python
full_inputs.append(torch.cat([train_input, input], dim=-2))
```

so both must have the same number of token columns. Without a fixed
`padding_length`, each call pads to its own longest sequence, and a training
pool containing a 14-mer next to a test batch of 12-mers gives 16 and 14
columns:

```
RuntimeError: Sizes of tensors must match except in dimension 0.
Expected size 16 but got size 14 for tensor number 1 in the list.
```

This is inherited from stock `GPLLModel` (verified: the uncached class fails
identically), but it occurs as soon as more than one scaffold is optimised. v3
raises a message naming the cause and the fix instead of the raw torch error:

```python
plm = CachedProteinEmbedding('esm2_t33_650M_UR50D',
                             padding_length=64)   # >= longest + headroom
```

## Tuning of CachedGPLLModel and CachedProteinEmbedding

| symptom | change |
|---|---|
| still OOM | lower `embed_batch_size` (32, then 16) |
| VRAM headroom, want speed | raise `embed_batch_size` (128, 256) |
| host RAM climbing | set `max_cache_size` (e.g. 50_000) |
| fine-tuning the encoder (`layers_to_finetune`) | caching auto-disables; expect stock memory behaviour |


## Verify CachedGPLLModel is working

```python
from mobius import CachedProteinEmbedding, CachedGPLLModel

plm = CachedProteinEmbedding(pretrained_model_name='esm1b_t33_650M_UR50S',
                             embedding_type='avg',
                             embed_batch_size=64)
gpmodel = CachedGPLLModel(kernel=RBFKernel(), pretrained_model=plm,
                          noise_prior=NormalPrior(0, 1))
# ... Planner / SequenceGA / ExpectedImprovement unchanged ...

print(plm.backend_name)       # 'esm2'
print(plm.cache_info())
# after a round: max_forward_batch must never exceed embed_batch_size
```


## The stock multi-objective bug

`Problem._evaluate` does `if self._acq_fun.maximize:`. For a multi-objective
acquisition function `maximize` is an ndarray, so this raises:

```
ValueError: The truth value of an array with more than one element is
ambiguous. Use a.any() or a.all()
```

It fires on the first GA generation, so **every** multi-objective run dies
immediately — including `examples/multi_objectives.ipynb`. Confirmed against
the stock code with a stock uncached embedder, so it is not caused by caching.

The replacement also fixes a quieter issue on the same lines: the pre-evaluation
shift took `min`/`max` over the whole array rather than per column, mixing the
scales of different objectives. Single-objective output is bit-identical to
stock (verified).
