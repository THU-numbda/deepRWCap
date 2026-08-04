# DeepRWCap: Neural-Guided Random-Walk Capacitance Solver for IC Design

DeepRWCap is a machine-learning-guided random-walk solver that accelerates
capacitance extraction by predicting the transition quantities used to guide
each step of the walk. This repository contains the training, inference, and
evaluation code for the
[AAAI 2026 paper](https://doi.org/10.1609/aaai.v40i2.37066).

## Quick Start

### Requirements

- Python 3.10+
- CUDA 12.6+ for GPU support
- CMake 3.18+
- GCC/G++

The reported environment can be reproduced with the NVIDIA PyTorch container and
a `/workspace` bind mount:

```bash
singularity pull pytorch-24.12-py3.sif docker://nvcr.io/nvidia/pytorch:24.12-py3
singularity shell --nv --bind /path/to/deepRWCap:/workspace pytorch-24.12-py3.sif
```

Replace `/path/to/deepRWCap` with the path to this repository. The bind mount
makes the repository available inside the container at `/workspace`.

Install the remaining Python dependencies inside the container:

```bash
pip install thop neuraloperator
```

## Generate Training Data

```bash
cd ggft
./run_ggft.sh
```

> [!WARNING] See the [GGFT documentation](ggft/README.md) for details on
> generating finite-difference training data. Alternatively, run the commands
> above to use the provided generation script.

Each dataset is a binary file with the following format:

- **Header:** Two values:
  - `N`: Grid resolution, such as 16, 21, or 23
  - `block_w`: Block-width parameter, set to 1
- **Body:** Repeated samples containing:
  - Dielectric data: `N³` values representing the permittivity distribution
  - Structure data: `7 × n_structures` geometric values, currently unused
  - Poisson[^1]/gradient data: `6 × N²` values for the six faces of the cube

[^1]: The surface Green's function is equivalent to the Poisson kernel.

## Train and Compile Models

Run the provided scripts from `training_pytorch/`:

```bash
cd training_pytorch
./run_training.sh # train the models from scratch
./run_compilation.sh # compile with TensorRT and copy to `/workspace/models/`
```

`training_pytorch/src/main.py` manages training and optimization of the
presented models using PyTorch and TensorRT. It:

- Trains multiple predefined models on GPUs with multiprocessing
- Measures FLOPs and parameter counts
- Exports the best models in TorchScript format
- Benchmarks and compiles models with TensorRT in FP32 and FP16
- Reports latency and throughput improvements after compilation

The entry point can also run individual stages:

```bash
python src/main.py [train] [compile]
```

- `train`: Run training only
- `compile`: Run TensorRT compilation only
- No arguments: Run both training and compilation

Model configurations and datasets are predefined in `MODELS_TO_TRAIN` and
`DATASET_BASE_CONFIGS`.

Training produces:

- Models in `/workspace/training_pytorch/models/`
- Logs in `/workspace/training_pytorch/runs/`

## Build the C++ Inference Library

The C++ backend provides high-performance inference using LibTorch, TensorRT,
and CUDA. The build produces `inference_cpp/build/lib/dnnsolver.so`:

```bash
unset CUDACXX
cd inference_cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

The DeepRWCap binary expects `dnnsolver.so` in `/workspace/executable`.

## Running DeepRWCap

### Setup

1. Activate the Singularity container.
2. Ensure that `dnnsolver.so` and `models.txt` are in the `executable/`
   directory.
3. Use a single GPU to ensure correct CUDA stream synchronization:
   `export CUDA_VISIBLE_DEVICES=0`.

### Direct Usage

Run a capacitance-extraction task with:

```bash
/path/to/binary -f <input_file.cap3d> -n <num_cores> [accuracy_options]
```

Required arguments:

- `-f <input_file.cap3d>`: Input file containing the 3D capacitance structure
- `-n <num_cores>`: Number of CPU cores to use

Accuracy-control options:

- `-p <value>`: Convergence threshold for self-capacitance
- `-c <value>`: Convergence threshold for the capacitance matrix
- `--c-ratio <value>`: Fraction of capacitance-matrix elements that must meet
  the convergence threshold

For example, from `executable/`:

```bash
./bin/deepRWCap -f /workspace/testcases/cap3d/case3.cap3d -n 16 -p 0.01 -c 0.01 --c-ratio 0.95
```

The command produces:

- `case3.cap3d.out`: Capacitance-extraction results
- `case3.cap3d.log`: Detailed execution log

## Reproduce the Paper Results

Use `executable/run_script.py` to reproduce the paper's capacitance-extraction
results. The script provides:

- Automated repeated testing for statistical analysis
- Multi-core scaling with 1, 2, 4, 8, and 16 cores
- Relative-error analysis against reference solutions

```bash
python run_script.py /path/to/binary <number_of_runs> [test_cases...]
```

Parameters:

- `/path/to/binary`: `./bin/deepRWCap` for DeepRWCap, or
  `./baselines/rwcap_agf`, `./baselines/rwcap_microwalk`, or
  `./baselines/rwcap_fdm` for a baseline
- `<number_of_runs>`: Number of iterations per test case, such as 10
- `[test_cases...]`: Optional test cases such as `case1` and `case2`, or `all`

For example, from `executable/`:

```bash
python run_script.py ./bin/deepRWCap 10 all
```

## Citation

If you use DeepRWCap in academic work, please cite the
[AAAI 2026 paper](https://doi.org/10.1609/aaai.v40i2.37066):

```bibtex
@article{rodriguez2026deeprwcap,
  title   = {DeepRWCap: Neural-Guided Random-Walk Capacitance Solver for IC Design},
  author  = {Rodriguez, Hector R. and Huang, Jiechen and Yu, Wenjian},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume  = {40},
  number  = {2},
  pages   = {971--979},
  year    = {2026},
  doi     = {10.1609/aaai.v40i2.37066},
  url     = {https://doi.org/10.1609/aaai.v40i2.37066}
}
```
