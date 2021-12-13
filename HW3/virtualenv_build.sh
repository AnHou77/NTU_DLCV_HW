#!/bin/bash

virtualenv --python=python3.8 venv

source ./venv/bin/activate

pip list

python --version

pip install -r requirements.txt

pip list

echo 'virtual environment build completed !'