#!/bin/sh

module add python3/3.12.1
module add gcc/8.3.0-shared

python3 -m venv --system-site-packages .venv
source .venv/bin/activate

export PIP_INDEX_URL="https://$USER:$(cat ~/token.txt)@devops.saab.se/Python/_packaging/packages/pypi/simple"

pip3 install --upgrade pip
pip3 install wheel
pip3 install -r requirements.txt

