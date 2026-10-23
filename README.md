# Bridging Image Restoration and Recognition via Causal Mediated Unrolling

Official implementation of **Bridging Image Restoration and Recognition via Causal Mediated Unrolling**, NeurIPS 2026.

## Environment

The code was tested with Python 3.10, CUDA 11.8, and an NVIDIA RTX 3090.

```bash
conda create -n cameu python=3.10 -y
conda activate cameu

pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmcv==2.2.0
pip install -r requirements.txt
```
