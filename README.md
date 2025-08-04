# DeepRWCap: Neural-Guided Random-Walk Capacitance Solver for IC Design

DeepRWCap is a machine learning-guided random walk solver that accelerates capacitance extraction by predicting the transition quantities required to guide each step of the walk. This repository contains the implementation described in the paper "DeepRWCap: Neural-Guided Random-Walk Capacitance Solver for IC Design."

![OVERVIEW](./figures/overview.png)

## Index
- [Prerequisites](#prerequisites)
- [Datasets](#datasets)
- [Python Training](#python-training)
- [C++ Inference](#c-inference)
- [DeepRWCap](#deepwrcap)
- [GGFT](#ggft)

## Prerequisites
- Python 3.10+
- CUDA 12.6+ (for GPU support)
- CMake 3.18+
- GCC/G++ compiler

Containerized approach (recommended):
```bash
singularity pull pytorch-24.12-py3.sif docker://nvcr.io/nvidia/pytorch:24.12-py3
singularity shell --nv pytorch-24.12-py3.sif
```

## Datasets

Download:
```bash
cd datasets
source download_datasets.sh
```

Each dataset file is a binary file with the following format:
**Header** (2 values):
- `N`: Grid resolution (e.g., 16, 21, 23)
- `block_w`: Block width parameter (set to 1)

**Body** (repeated samples):
Each sample contains:

- Dielectric data: `(N/block_w)³` values representing the permittivity distribution
- Structure data: `7 × n_structures` values (geometric structure parameters, unused)
- Green's function/Gradient data: `6 × N²` values for the 6 faces of the cube

## Python Training

### Setup

Download pre-trained models:
Models:
```bash
cd models
source download_models.sh
```

Install the core dependencies with:
```bash
pip install -r requirements.txt
```

Note: The requirements.txt is configured for PyTorch 2.6+ with CUDA 12.6 support (supported by the recommended container). If you are using a different CUDA version, you may need to install PyTorch separately following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Features

`pytorch_training/main.py` manages training and optimization of the presented models using PyTorch and TensorRT.

- Trains multiple predefined models on GPU(s) with multiprocessing
- Automatically measures FLOPs and parameter counts
- Exports best models in TorchScript format
- Benchmarks and compiles models with TensorRT (FP32 & FP16)
- Reports latency and throughput improvements after compilation


### Usage

Run from the command line:

```bash
python pytorch_training/main.py [train] [compile]
```

- `train` → Run training only
- `compile` → Run TensorRT compilation only
- `No arguments` → Run both training and compilation

Model configurations and datasets are predefined in the script (see `MODELS_TO_TRAIN` and `DATASET_BASE_CONFIGS`).

### Outputs

- Trained models saved in: `/workspace/models/`
- Logs saved in: `/workspace/runs/`


## C++ Inference

### Build

The C++ backend provides high-performance inference using LibTorch and TensorRT with CUDA acceleration.


__Containerized approach__ (recommended):
```bash
singularity pull pytorch-24.12-py3.sif docker://nvcr.io/nvidia/pytorch:24.12-py3
singularity shell --nv pytorch-24.12-py3.sif
```

__Build:__
```bash
cd inference_cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```


## DeepRWCap




## GGFT


