FROM nvidia/cuda:11.6.0-devel-ubuntu20.04 as base
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y build-essential
RUN apt-get install -y curl
RUN apt-get install -y wget
RUN mkdir /home/installs
RUN curl -o /home/installs/miniconda_install.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh
RUN sh /home/installs/miniconda_install.sh -b
RUN rm -f /home/installs/miniconda_install.sh
WORKDIR /home/
RUN git clone https://github.com/benmatok/nglod.git
WORKDIR /home/nglod
ENV PATH /root/miniconda3/bin:$PATH
RUN conda create -n nglod python=3.8
ENV PATH /root/miniconda3/envs/nglod/bin:$PATH
ENV CONDA_DEFAULT_ENV nglod
SHELL ["conda", "run", "-n", "nglod", "/bin/bash", "-c"]
RUN pip install --upgrade pip
RUN pip install -r ./infra/requirements.txt
RUN apt-get install -y libopenexr-dev 
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install dialog apt-utils -y
RUN apt-get install -y python3-opencv
RUN pip install pyexr
RUN pip install laspy
RUN pip install open3d

WORKDIR /home/nglod/sdf-net/lib/extensions/mesh2sdf_cuda
RUN python setup.py clean --all install --user
WORKDIR /home/nglod/sdf-net/lib/extensions/sol_nglod
RUN python setup.py clean --all install --user
WORKDIR /home/nglod/sdf-net
