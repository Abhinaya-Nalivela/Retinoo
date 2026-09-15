#!/usr/bin/env bash
# uninstall existing torch and torchvision
pip uninstall -y torch torchvision
# install specific CPU-only versions compatible with the project
pip install torch==2.3.0+cpu torchvision==0.18.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
# verify installation
python - <<EOF
import torch, torchvision
print('torch version:', torch.__version__)
print('torchvision version:', torchvision.__version__)
EOF
