#!/bin/sh
sudo apt-get install -y python3.8-venv
sudo apt-get install -y libpython3.8-dev
sudo apt-get install -y python3-dev
sudo apt-get install -y build-essential

python3.8 -m venv venv
. venv/bin/activate
python3.8 -m pip install --upgrade pip
pip install -r requirements.txt
