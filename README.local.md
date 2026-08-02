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

print(plm.cache_info())
# max_forward_batch must never exceed embed_batch_size
