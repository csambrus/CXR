#!/usr/bin/env bash
set -e

cd /workspace/CXR

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -U pip
  pip install -r requirements.txt
  pip install ipykernel jupyterlab
fi
source .venv/bin/activate
python -m ipykernel install --user --name cxr_venv --display-name "CXR (.venv)"
ln -s /workspace/kaggle /root/.kaggle
