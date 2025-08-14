FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 as BASE

RUN apt update && \
    apt upgrade -y

RUN apt install -y --no-install-recommends ca-certificates wget unzip python3 pip python3-pip vim

# with 12.4 ->
RUN pip3 install torch torchvision torchaudio
RUN pip3 install torchinfo

#RUN rm -rf /var/lib/apt/lists/* \
#    && apt-get autoremove -y \
#    && apt-get clean

ENV APP=/app
RUN mkdir -p $APP
WORKDIR $APP

RUN apt install -y --no-install-recommends python3-dev
RUN apt install -y --no-install-recommends git

# upgrade pip & install
RUN pip3 install --upgrade pip

#RUN pip3 install botorch biotite==0.41 fair-esm grakel gpytorch matplotlib numba numpy numpydoc pandas parmed
#RUN pip3 install openmm rdkit seaborn sentencepiece scikit-learn scipy sphinx==6.2.1 sphinx_rtd_theme tqdm transformers mhfp meeko ray pymoo mapchiral vina
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt

RUN git clone https://github.com/prody/ProDy.git && cd ProDy && python3 setup.py build_ext --inplace --force && pip install -Ue .

COPY . $APP
RUN pip install -e .

#RUN python3 setup.py install
#
##RUN apt install -y --no-install-recommends inetutils-ping
#
#RUN addgroup --gid 9999 --system app && adduser --system --uid 9999 --group app
#RUN chown -R app:app /home/app
#ENV APP_HOME=/home/app
#WORKDIR $APP_HOME
#RUN chown -R app:app $APP_HOME
#USER app
