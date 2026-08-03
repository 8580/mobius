#FROM nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

#ENV DEBIAN_FRONTEND=noninteractive
#ENV PYTHONUNBUFFERED=1
##ENV PIP_NO_CACHE_DIR=1
#
RUN apt-get update && apt-get install -y --no-install-recommends \
    git
    #&& rm -rf /var/lib/apt/lists/*

WORKDIR /opt

#RUN python3 -m pip install --upgrade pip setuptools
RUN python3 -m pip install jupyter-packaging versioneer
#RUN python3 -m pip install nglview

COPY requirements.txt .
RUN python -m pip install -r requirements.txt --no-build-isolation


COPY . /opt/mobius/
WORKDIR /opt/mobius
RUN python -m pip install . 
WORKDIR /

RUN pip install jupyterlab matplotlib scikit-learn
## Expose the default Jupyter port
EXPOSE 8888
#
## Start JupyterLab on container launch
## --ip=0.0.0.0 allows connections from outside the container
## --allow-root allows it to run if you don't map a custom non-root user
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]

## docker build -f GPU.Dockerfile -t gpu-mobius .
## docker run --gpus all -it --rm gpu-mobius /bin/bash
## docker run --gpus all -it --rm -p 8888:8888 -v "$(pwd)":/workspace gpu-mobius
