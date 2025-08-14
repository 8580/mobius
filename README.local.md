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
