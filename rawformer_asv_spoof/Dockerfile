FROM nvcr.io/nvidia/pytorch:22.01-py3

RUN apt-get update

RUN pip install torch==1.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchaudio==0.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install pip --upgrade
RUN pip install wandb

COPY ./ /app/
WORKDIR /app

RUN pip install git+https://github.com/asteroid-team/torch-audiomentations

ENV https_proxy=http://192.168.101.101:3128
ENV http_proxy=http://192.168.101.101:3128

ENTRYPOINT ["bash"]