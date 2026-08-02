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

## Tuning of CachedGPLLModel and CachedProteinEmbedding

| symptom | change |
|---|---|
| still OOM | lower `embed_batch_size` (32, then 16) |
| VRAM headroom, want speed | raise `embed_batch_size` (128, 256) |
| host RAM climbing | set `max_cache_size` (e.g. 50_000) |
| fine-tuning the encoder (`layers_to_finetune`) | caching auto-disables; expect stock memory behaviour |
