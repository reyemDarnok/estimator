#!/usr/bin/env bash
git clone git@github.com:reyemDarnok/estimator.git
cd estimator || exit 1
python3 -m pip install -r estimator/requirements.txt
./estimator.py -h