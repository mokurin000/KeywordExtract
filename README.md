
# Installation

## Main program

```bash
pip install -e .
```

## CUDA

### To check:

```bash
python -c 'import torch; print(torch.cuda.is_available())'
```

### To Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

```bash
conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
```