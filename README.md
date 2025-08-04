# DeepRWCap: Neural-Guided Random-Walk Capacitance Solver for IC Design

# dnn-assisted-frw-impl
## Set up
Locate and set up Pytorch directory
```python
import torch
print(torch.utils.cmake_prefix_path)
```
```cmake
find_package(Torch REQUIRED PATHS /path/to/torch)
```

Set up CUDA installation directory on CMakeLists.txt:
```cmake
set(CUDA_TOOLKIT_ROOT_DIR /path/to/cuda)
```

## Compile the project

```sh
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
```

## Run the project

```sh
./dnnsolver path/to/dataset [Number of samples] [X, Y, Z] (optional)
```

Examples for testing Green's function and Gradient predictions:
```sh
./dnnsolver_exec /data/hector/datasets/greens100000samples5structures/0.bin 2048 
./dnnsolver_exec /data/hector/datasets/gradientX100000samples5structures/0.bin 2048 X
./dnnsolver_exec /data/hector/datasets/gradientY100000samples5structures/0.bin 2048 Y
./dnnsolver_exec /data/hector/datasets/gradientZ100000samples5structures/0.bin 2048 Z
```



## Model configuration

__Legacy__ model configuration:
```bash
# Model configuration file
# Format: path # description

/data/hector/models6elements_normalized/DirectFaceSelectorSmallerHarder_0.0022.pt.jit # Face selector Green's
/data/hector/models6elements_normalized/DirectFacePredictorAttentionSmallerHarder_0.0000.pt.jit # Green predictor
/data/hector/models6elements_normalized/DirectGradientFaceAndWeightSmallerHarder_0.0057.pt.jit  # Gradient face selector weight predictor
/data/hector/models6elements_normalized/DirectGradientAndSignFace2SmallerHarder_0.0000.pt.jit # Gradient face2 predictor with sign
/data/hector/models6elements_normalized/DirectGradientPredictorFace1Smaller_0.0000.pt.jit # Gradient face1 predictor
```


| Case	| Cap Mean ± StdDev	   | Time Mean ± StdDev 	| RelErr Mean ± StdDev |
--------|----------------------|------------------------|----------------------|
| case1	| 1.843e-16 ± 2.07e-18 |	16.55 ± 0.29s       |	1.84 ± 1.10%       |
| case2	| 1.232e-15 ± 1.15e-17 |	12.93 ± 0.34s       | 	0.89 ± 0.72%       |



__Small__ model configuration:
```bash
# Model configuration file
# Format: path # description

/data/hector/final_models/GreensFaceSelector3D_0.0014.pt.jit # Face selector Green's
/data/hector/final_models/GreensFacePredictor_5.5e-8.pt.jit # Green predictor
/data/hector/final_models/GradientFaceSelectorWeight_0.0128.pt.jit  # Gradient face selector weight predictor
/data/hector/final_models/GradientFace2Predictor_5.1e-7.pt.jit # Gradient face2 predictor with sign
/data/hector/final_models/GradientFace1Predictor_7.8e-8.pt.jit # Gradient face1 predictor
```



| Case  | Cap (μ±σ)            | Time (μ±σ) | MatrixErr (μ±σ) | SelfCapErr (μ±σ)  | Walks (μ±σ)       | Total Tasks (μ±σ)   | GPU Tasks (Grad/Green)      | GPU Batch (Grad/Green)      |
|-------|----------------------|------------|-----------------|-------------------|-------------------|---------------------|-----------------------------|-----------------------------|
| case1 | 1.588e-14±1.47e-16   | 9.9±0.7s   | 1.4±0.6%        | 0.7±0.6%          | 56730±4044        | 457720±33107        | 53486±3819/250705±17819     | 371±24/380±17               |
| case2 | 1.677e-14±1.73e-16   | 10.6±0.3s  | 2.5±1.0%        | 1.8±1.1%          | 88371±2602        | 717046±21334        | 31921±991/196942±5929       | 138±67/243±21               |
| case3 | 2.119e-14±1.39e-16   | 6.5±0.2s   | 2.5±0.6%        | 1.8±0.6%          | 25293±1919        | 187371±14314        | 24443±1861/104497±8159      | 483±51/293±17               |
| case9 | 1.793e-16±1.54e-18   | 52.8±1.3s  | 4.3±0.7%        | 4.5±0.8%          | 1681306±27867     | 13453562±221783     | 302272±4971/1814021±28442   | 53±3/187±2                  |

__Balanced__ model configuration:

```bash
# Model configuration file
# Format: path # description

/data/hector/final_models/GreensFaceSelector3D_0.0014.pt.jit # Face selector Green's
/data/hector/final_models/GreensFacePredictor_5.5e-8.pt.jit # Green predictor
/data/hector/models6elements_normalized/DirectGradientFaceAndWeightSmallerHarder_0.0057.pt.jit  # Gradient face selector weight predictor
/data/hector/models6elements_normalized/DirectGradientAndSignFace2SmallerHarder_0.0000.pt.jit # Gradient face2 predictor with sign
/data/hector/final_models/GradientFace1Predictor_7.8e-8.pt.jit # Gradient face1 predictor
```

| Case  | Cap (μ±σ)            | Time (μ±σ) | MatrixErr (μ±σ) | SelfCapErr (μ±σ)  | Walks (μ±σ)       | Total Tasks (μ±σ)   | GPU Tasks (Grad/Green)      | GPU Batch (Grad/Green)  |
|-------|----------------------|------------|-----------------|-------------------|-------------------|---------------------|-----------------------------|-----------------------------|
| case1 | 1.566e-14±1.10e-16   | 10.5±0.5s  | 2.9±0.6%        | 1.6±0.7%          | 60826±3263        | 492155±26747        | 57343±3091/270856±14545     | 333±20/429±12               |
| case2 | 1.666e-14±1.77e-16   | 10.7±0.4s  | 1.9±0.9%        | 1.2±1.1%          | 87859±4465        | 714909±34474        | 31719±1652/196603±9145      | 70±35/280±16                |
| case3 | 2.148e-14±1.47e-16   | 6.2±0.3s   | 1.6±0.6%        | 0.6±0.5%          | 24218±1602        | 177810±11845        | 23396±1552/99410±6959       | 424±30/330±18               |
| case9 | 1.901e-16±8.61e-19   | 49.0±1.4s  | 1.4±0.4%        | 1.3±0.5%          | 1505997±14005     | 11934921±116638     | 270886±2484/1641570±16368   | 50±3/189±1                  |


__Huge__ model configuration:
```bash
# Model configuration file
# Format: path # description

/data/hector/huge_models/HugeFaceSelector_0.0002.pt.jit # Face Selector (Classification) - Error: 0.0002s
/data/hector/huge_models/HugeGreensPredictor_0.0000.pt.jit # Green's Function Predictor - Error: ~0
/data/hector/huge_models/HugeFaceSelectorWeight_0.0045.pt.jit # Face Selector + Weight (Dual Head) - Error: 0.0035
/data/hector/huge_models/HugeGradient2Predictor_0.0000.pt.jit # Gradient Face2 Predictor
/data/hector/huge_models/HugeGradient1Predictor_0.0000.pt.jit # Gradient Face1 Predictor - Error: ~0
```

Summary Table:                                                                                                                                                              
| Case  | Cap (μ±σ)            | Time (μ±σ) | MatrixErr (μ±σ) | SelfCapErr (μ±σ)  | Walks (μ±σ)       | Total Tasks (μ±σ)   | GPU Tasks (Grad/Green)      | GPU Batch (Grad/Green)      |
|-------|----------------------|------------|-----------------|-------------------|-------------------|---------------------|-----------------------------|-----------------------------|
| case1 | 1.586e-14±1.57e-16   | 31.1±0.8s  | 1.4±0.5%        | 0.8±0.6%          | 58624±1794        | 468775±15147        | 55277±1695/257213±8176      | 382±23/459±14   |       
| case2 | 1.665e-14±1.46e-16   | 31.6±1.1s  | 1.8±0.5%        | 1.3±0.6%          | 88218±4358        | 719068±35740        | 31854±1564/197368±9652      | 84±34/330±12    |         
| case3 | 2.152e-14±1.51e-16   | 18.1±1.1s  | 2.0±0.6%        | 0.6±0.4%          | 24678±3061        | 178274±22131        | 23824±2954/99855±12422      | 587±71/359±18   |        
| case9 | 1.885e-16±1.33e-18   | 121.9±2.2s | 1.0±0.4%        | 0.6±0.6%          | 1513523±16634     | 12069705±78711      | 272129±3166/1646109±17432   | 61±1/210±1      |