FROM nvidia/cuda:11.6.0-devel-ubuntu20.04 as base
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y build-essential
RUN apt-get install -y curl


