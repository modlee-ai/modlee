# FROM python:3.10-bullseye
FROM ubuntu:22.04

WORKDIR /root

RUN apt update && \
    apt -y install sudo
RUN apt -y install python3.10 python3-pip
# RUN curl -0 https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz && \
#     tar -xf Python-3.10.13.tgz
RUN alias python3=python3.10

RUN mkdir modlee_pypi
COPY . ./modlee_pypi
RUN pip3 install --upgrade pip setuptools
RUN pip3 install ./modlee_pypi/
ENTRYPOINT [ "/bin/bash" ]
# https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz
