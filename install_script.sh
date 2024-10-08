#!/bin/sh

set -e

sudo apt-get update
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-dev python3.8-venv
sudo apt-get install -y build-essential unzip

python3.8 -m venv venv
. venv/bin/activate

python3.8 -m pip uninstall -y pip
python3.8 -m ensurepip --default-pip
python3.8 -m pip install --upgrade 'pip<24.1'
python3.8 -m pip --version

python3.8 -m pip install -r requirements.txt
python3.8 -m pip install zenodo_get

zenodo_get https://doi.org/10.5281/zenodo.10890078
unzip Datasets.zip -d Datasets/
