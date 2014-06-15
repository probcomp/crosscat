#!/bin/bash


bash install_cxfreeze_user.sh
pip install -r ../../requirements.txt

mkdir -p ~/.matplotlib
echo backend: Agg > ~/.matplotlib/matplotlibrc

