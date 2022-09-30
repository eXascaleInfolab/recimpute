#!/bin/sh
sudo apt-get update
sudo apt-get install -y python3.8-venv
sudo apt-get install -y libpython3.8-dev
sudo apt-get install -y python3-dev
sudo apt-get install -y build-essential
sudo apt-get install -y unzip

python3.8 -m venv venv
. venv/bin/activate
python3.8 -m pip install --upgrade pip
pip install -r requirements.txt

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Z8cU5S5v5hITrP_jD56UzqfUTPRjQrxI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Z8cU5S5v5hITrP_jD56UzqfUTPRjQrxI" -O datasets.zip && rm -rf /tmp/cookies.txt
unzip datasets.zip -d Datasets/
