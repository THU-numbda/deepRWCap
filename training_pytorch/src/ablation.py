from __future__ import annotations
import os, time, torch, torch.multiprocessing as mp
import torch.nn as nn
import numpy as np
import sys
from torch.utils.data import DataLoader, Subset
from dataset import GreenDataset  
from architecture_ablation import *
from train import train_model_improved, test_model_improved
from thop import profile
import torch_tensorrt

def count_params(model, verbose=False):
    """Counts the total number of trainable parameters in a model."""
    total_params = 0
    if verbose:
        print("--- Trainable Parameters ---")
    for name, p in model.named_parameters():
        if p.requires_grad:
            num_params = p.numel()
            # For complex parameters, count real and imaginary parts separately
            if p.is_complex():
                num_params *= 2
            if verbose:
                print(f'{name:<60} -> {str(list(p.shape)):<20} -> {num_params} params')
            total_params += num_params
    if verbose:    
        print("----------------------------")
    return total_params


# ───────────────────────────────── configuration ─────────────────────────────────
TRAINING_CONFIG = {
    "batch_size":      16,
    "epochs":          200,
    "device_count":    torch.cuda.device_count(),   # auto detect
    "model_save_dir":  "models_ablation",
    "tensorrt_save_dir":"models_ablation",
    "log_dir":         "runs_ablation",
    "optimizer_params":{'lr': 1e-3, "weight_decay": 1e-10}
}

MODELS_TO_TRAIN = [
    {
        "name": "PoissonReLUKL",
        "class": FacePredictor,
        "kwargs": {"head_mode": "relu"},
        "dataset": "greens_harder",
        "target_face": 0,
        "mainloss": "kl_div",
    },
    {
        "name": "PoissonReLUKLLearnable",
        "class": FacePredictor,
        "kwargs": {"head_mode": "relu", "pe_mode": "learnable"},
        "dataset": "greens_harder",
        "target_face": 0,
        "mainloss": "kl_div",
    },
    {
        "name": "PoissonReLUKLNoPE",
        "class": FacePredictor,
        "kwargs": {"head_mode": "relu", "pe_mode": "none"},
        "dataset": "greens_harder",
        "target_face": 0,
        "mainloss": "kl_div",
    },
    {
        "name": "PoissonReLUKLNoPEConv",
        "class": FacePredictor,
        "kwargs": {"head_mode": "relu", "pe_mode": "none", "solver": "conv"},
        "dataset": "greens_harder",
        "target_face": 0,
        "mainloss": "kl_div",
    },
    {
        "name": "PoissonReLUKLNoPEMLP",
        "class": FacePredictor,
        "kwargs": {"head_mode": "relu", "pe_mode": "none", "solver": "mlp"},
        "dataset": "greens_harder",
        "target_face": 0,
        "mainloss": "kl_div",
    },
    {
        "name": "PoissonReLUKLNoPE3D",
        "class": FacePredictor,
        "kwargs": {"head_mode": "relu", "pe_mode": "none", "solver": "3d"},
        "dataset": "greens_harder",
        "target_face": 0,
        "mainloss": "kl_div",
    },
]

# Dataset configurations
DATASET_BASE_CONFIGS = {
    "greens_harder": {
        "path": ["../ggft/dataset/poisson.bin"],
        "input_mode": "dielectric",
        "N": 23,
        "n_structures": 16,
        "dtype": np.float64
    },
    "gradient_harder": {
        "path": ["../ggft/dataset/gradient.bin"],
        "input_mode": "dielectric",
        "N": 23,
        "n_structures": 16,
        "dtype": np.float64
    }
}

# ───────────────────────────── helper: choose dataset_type ──────────────────────
def _dataset_type(model_cfg) -> str:
    """greens_function  vs  gradient   (for GreenDataset)."""
    return "greens_function" if "greens" in model_cfg["dataset"] else "gradient" 

# ───────────────────────────── training of one model ────────────────────────────
def train_single_model(model_cfg, gpu_id: int):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    print(f"\n{'='*60}"
          f"\n   {model_cfg['name']}  →  {device}"
          f"\n{'='*60}")
    
    # ------------------------------ model / loss -------------------------------
    model    = model_cfg["class"](name=model_cfg['name'], **model_cfg["kwargs"]).to(device)

    # Measure FLOPs and parameters
    try:
        print(f"Model Name: {model_cfg['name']}")
        custom_count = count_params(model)
        print(f"Model Parameters (custom counting): {custom_count / 1e3:.2f} K")
        dummy_input = torch.randn(1, 1, 1, 23, 23, 23, device=device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        print(f"Model FLOPs (thop): {flops / 1e6:.2f} MFLOPs")
        print(f"Model Parameters (thop): {params / 1e3:.2f} K")
    except Exception as e:
        print(f"Could not measure FLOPs and parameters: {e}")
    
    sys.stdout.flush()
    
    # ------------------------------ dataset ------------------------------------
    base_cfg = DATASET_BASE_CONFIGS[model_cfg["dataset"]]
    ds = GreenDataset(files=base_cfg["path"],
                      dataset_type=_dataset_type(model_cfg),
                      N=base_cfg["N"],
                      dtype=base_cfg["dtype"],
                      n_structures=base_cfg["n_structures"])

    # splits (85/10/5)
    tot = len(ds)
    i1 = int(tot*0.90)
    idx = list(range(tot))
    train_ld = DataLoader(Subset(ds, idx[:i1]), batch_size=TRAINING_CONFIG["batch_size"], shuffle=True)
    test_ld   = DataLoader(Subset(ds, idx[i1:]),  batch_size=TRAINING_CONFIG["batch_size"], shuffle=False)

    _, best = train_model_improved(
        model, train_ld, test_ld,
        device=device,
        optimizer_params=TRAINING_CONFIG["optimizer_params"],
        target_face=model_cfg.get("target_face"),
        epochs=TRAINING_CONFIG["epochs"],
        log_dir=TRAINING_CONFIG["log_dir"],
        model_save_dir=TRAINING_CONFIG["model_save_dir"],
        mainloss=model_cfg["mainloss"],
    )
    return model_cfg["name"], best

def benchmark_model_performance(model, input_shape=[512,1,1,23,23,23], iterations=1000, model_cfg=None):
    """Measures model latency and throughput with better TensorRT compatibility."""
    try:
        # Determine device more safely
        device = torch.device("cuda:0")
        
        # Try to move model to device if it's a regular PyTorch model
        try:
            if hasattr(model, 'parameters'):
                param_list = list(model.parameters())
                if param_list:
                    if not param_list[0].is_cuda:
                        model = model.to(device)
                    else:
                        device = param_list[0].device
            else:
                model = model.to(device)
        except:
            pass
        
        # Detect if this is an FP16 model by testing with different input types
        model.eval()
        
        # Try FP32 first, then FP16 if it fails
        input_dtype = torch.float32
        dummy_input = torch.randn(input_shape, device=device, dtype=input_dtype)
        
        print(f"    Detecting input precision...")
        with torch.no_grad():
            try:
                _ = model(dummy_input)
                print(f"    Model accepts FP32 inputs")
            except Exception as e:
                if "Expected inputs" in str(e) or "dtype" in str(e).lower():
                    print(f"    Model requires FP16 inputs, switching...")
                    input_dtype = torch.half
                    dummy_input = torch.randn(input_shape, device=device, dtype=input_dtype)
                    try:
                        _ = model(dummy_input)
                        print(f"    Model accepts FP16 inputs")
                    except Exception as e2:
                        print(f"    Model failed with both FP32 and FP16: {str(e2)[:100]}")
                        return None, None
                else:
                    raise e

        # Warm-up with correct dtype
        print(f"    Warming up model with {input_dtype} inputs...")
        with torch.no_grad():
            for i in range(3):
                try:
                    _ = model(dummy_input)
                except Exception as e:
                    print(f"    Warmup iteration {i} failed: {str(e)[:100]}...")
                    if i == 2:
                        return None, None
        
        torch.cuda.synchronize()

        # Measure latency
        print(f"    Running {iterations} inference iterations...")
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(dummy_input)
        torch.cuda.synchronize()
        end_time = time.time()

        latency = (end_time - start_time) / iterations * 1000  # ms
        throughput = iterations * input_shape[0] / (end_time - start_time)  # samples/s
        print(f"\n📊 Model Evaluation Report for {model_cfg['name']}")
        print(f"    Input Precision: {input_dtype}")
        sys.stdout.flush()

        if model_cfg is not None:
            base_cfg = DATASET_BASE_CONFIGS[model_cfg["dataset"]]
            ds = GreenDataset(files=base_cfg["path"],
                            dataset_type=_dataset_type(model_cfg),
                            N=base_cfg["N"],
                            dtype=base_cfg["dtype"],
                            n_structures=base_cfg["n_structures"])
            
            tot = len(ds)
            i1 = int(tot*0.90)
            idx = list(range(tot))
            def dtype_collate(batch):
                xs, ys = zip(*batch)
                xs = torch.stack(xs).to(input_dtype)
                ys_dict = {}
                for key in ys[0].keys():
                    ys_dict[key] = torch.stack([y[key] for y in ys]).to(input_dtype)
                ys = ys_dict
                return xs, ys
            
            train_ld = DataLoader(Subset(ds, idx[:i1]), batch_size=TRAINING_CONFIG["batch_size"], shuffle=False, collate_fn=dtype_collate)
            test_ld   = DataLoader(Subset(ds, idx[i1:]),  batch_size=TRAINING_CONFIG["batch_size"], shuffle=False, collate_fn=dtype_collate)
            model_type = "FacePredictor" if model_cfg["target_face"] is not None else ("FaceSelector3D" if "greens" in model_cfg["dataset"] else "FaceSelectorWeight3D")
            train_loss = test_model_improved(model=model, 
                                loader=train_ld,
                                device=device,
                                target_face=model_cfg["target_face"],
                                model_type=model_type,
                                mainloss=model_cfg["mainloss"],)
            test_loss = test_model_improved(model=model, 
                                loader=test_ld,
                                device=device,
                                target_face=model_cfg["target_face"],
                                model_type=model_type,
                                mainloss=model_cfg["mainloss"],)
            
            # Print comprehensive evaluation report
            print(f"    Dataset: {model_cfg['dataset']}")
            print(f"    Target Face: {model_cfg.get('target_face', 'N/A')}")
            print(f"    Loss Function: {model_cfg['mainloss']}")
            print(f"    Batch Size: {TRAINING_CONFIG['batch_size']}")
            print(f"    Train Data Count: {len(train_ld.dataset)}")
            print(f"    Test Data Count: {len(test_ld.dataset)}")
            print(f"    Train Loss: {train_loss:.6e}")
            print(f"    Test Loss: {test_loss:.6e}")
        return latency, throughput
        
    except Exception as e:
        print(f"    Benchmarking failed: {str(e)[:200]}")
        return None, None

@torch.no_grad
def compile_and_benchmark_model(model_name, model, config, model_cfg):
    """Compile model with TensorRT and benchmark performance."""
    print(f"\n🔧 Compiling {model_name} with TensorRT...")
    
    # Set device to GPU 0 for compilation consistency
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    
    try:
        model.eval()
        model = model.to(device)
        
        # Create TensorRT variants
        os.makedirs(config['tensorrt_save_dir'], exist_ok=True)
        compile_args = dict(
            ir='torchscript',
            workspace_size=2 << 30,
            min_block_size=1,
            require_full_compilation=True,
            allow_shape_tensors=True,
        )
        if not isinstance(model, torch.jit.ScriptModule):
            compile_args = dict(
                ir                          = 'dynamo',
                optimization_level          = 5,
                require_full_compilation    = True,
                min_block_size              = 1,
                enable_experimental_decompositions = True,
                # use_fast_partitioner        = False,     # global partitioner
                assume_dynamic_shape_support= True,
                allow_shape_tensors         = True,
                # sparse_weights              = True,
                # use_explicit_typing         = True,
                tiling_optimization_level   = "full",
                num_avg_timing_iters        = 5,
                workspace_size              = 2 << 30,
                # cache_built_engines         = True,
                # reuse_cached_engines        = True,
                # timing_cache_path           = "/tmp/torchtrt_timing_cache.bin",
            )
        
        print(f"\n🔧 Start Compiling with {compile_args['ir']}…")
        # Initialize TensorRT models as None
        trt_fp32 = None
        trt_fp16 = None
        trt_fp32_path = os.path.join(config['tensorrt_save_dir'], f"{model_name}_tensorrt_fp32.jit")
        trt_fp16_path = os.path.join(config['tensorrt_save_dir'], f"{model_name}_tensorrt_fp16.jit")
        if os.path.exists(trt_fp32_path):
            print(f"  ✅ TensorRT FP32 already exists: {trt_fp32_path}")
            trt_fp32 = torch.jit.load(trt_fp32_path)
        else:
            # FP32 TensorRT
            try:
                print(f"  Creating TensorRT FP32 for {model_name}...")
                trt_fp32 = torch_tensorrt.compile(
                    model,
                    inputs=[torch_tensorrt.Input(
                        min_shape=[1, 1, 1, 23, 23, 23],
                        opt_shape=[512, 1, 1, 23, 23, 23],
                        max_shape=[2048, 1, 1, 23, 23, 23],
                        dtype=torch.float32
                    )],
                    enabled_precisions=[torch.float32],
                    **compile_args,
                )
                if compile_args['ir'] == 'torchscript':
                    torch.jit.save(trt_fp32, trt_fp32_path)
                else:
                    torch_tensorrt.save(trt_fp32, trt_fp32_path, output_format="torchscript", inputs=[torch.randn((512, 1, 1, 23, 23, 23)).cuda()])
                trt_fp32 = torch.jit.load(trt_fp32_path)
                print(f"  ✅ TensorRT FP32 saved: {trt_fp32_path}")
                
            except Exception as e:
                print(f"  ❌ TensorRT FP32 failed: {str(e)}")
        
        if os.path.exists(trt_fp16_path):
            print(f"  ✅ TensorRT FP16 already exists: {trt_fp16_path}")
            trt_fp16 = torch.jit.load(trt_fp16_path)
        else:
            # FP16 TensorRT
            try:
                print(f"  Creating TensorRT FP16 for {model_name}...")
                trt_fp16 = torch_tensorrt.compile(
                    model,
                    inputs=[torch_tensorrt.Input(
                        min_shape=[1, 1, 1, 23, 23, 23],
                        opt_shape=[512, 1, 1, 23, 23, 23],
                        max_shape=[2048, 1, 1, 23, 23, 23],
                        dtype=torch.float32
                    )],
                    enabled_precisions=[torch.half],
                    **compile_args,
                )
                if compile_args['ir'] == 'torchscript':
                    torch.jit.save(trt_fp16, trt_fp16_path)
                else:
                    torch_tensorrt.save(trt_fp16, trt_fp16_path, output_format="torchscript", inputs=[torch.randn((512, 1, 1, 23, 23, 23)).cuda()])
                trt_fp16 = torch.jit.load(trt_fp16_path)
                print(f"  ✅ TensorRT FP16 saved: {trt_fp16_path}")
                
            except Exception as e:
                print(f"  ❌ TensorRT FP16 failed: {str(e)[:200]}")

        # Baseline: TorchScript model
        print(f"  Benchmarking Base model for {model_name}...")
        torchscript_latency, torchscript_throughput = benchmark_model_performance(model, model_cfg=model_cfg)
        
        if torchscript_latency is not None and torchscript_throughput is not None:
            print(f"    Base: Latency={torchscript_latency:.2f}ms, "
                  f"Throughput={torchscript_throughput:.2f} samples/s")
        else:
            print(f"    Base benchmarking failed - skipping TensorRT compilation")
            return
        # Benchmark TensorRT models and compare
        if trt_fp32 is not None:
            print(f"  Benchmarking TensorRT FP32 for {model_name}...")
            fp32_latency, fp32_throughput = benchmark_model_performance(trt_fp32, model_cfg=model_cfg)
            
            if fp32_latency is not None and fp32_throughput is not None:
                print(f"    TensorRT FP32: Latency={fp32_latency:.2f}ms, "
                      f"Throughput={fp32_throughput:.2f} samples/s")
                latency_improvement = (torchscript_latency - fp32_latency) / torchscript_latency
                throughput_improvement = (fp32_throughput - torchscript_throughput) / torchscript_throughput
                print(f"      Latency Improvement: {latency_improvement:.2%}")
                print(f"      Throughput Improvement: {throughput_improvement:.2%}")
            else:
                print(f"    TensorRT FP32 benchmarking failed")

        if trt_fp16 is not None:
            print(f"  Benchmarking TensorRT FP16 for {model_name}...")
            fp16_latency, fp16_throughput = benchmark_model_performance(trt_fp16, model_cfg=model_cfg)
            
            if fp16_latency is not None and fp16_throughput is not None:
                print(f"    TensorRT FP16: Latency={fp16_latency:.2f}ms, "
                      f"Throughput={fp16_throughput:.2f} samples/s")
                latency_improvement = (torchscript_latency - fp16_latency) / torchscript_latency
                throughput_improvement = (fp16_throughput - torchscript_throughput) / torchscript_throughput
                print(f"      Latency Improvement: {latency_improvement:.2%}")
                print(f"      Throughput Improvement: {throughput_improvement:.2%}")
            else:
                print(f"    TensorRT FP16 benchmarking failed")
                
    except Exception as e:
        print(f"❌ Failed to compile {model_name}: {str(e)[:200]}")

def worker(model_cfg, gpu_id, ret):
    name, loss = train_single_model(model_cfg, gpu_id)
    ret[name] = loss    # store in shared dict

def run_training():
    """Run the training phase."""
    mp.set_start_method('spawn', force=True)
    
    print("🚀 Starting parallel model training")
    print(f"📊 Training {len(MODELS_TO_TRAIN)} models across {TRAINING_CONFIG['device_count']} GPUs")
    print(f"⏰ Epochs per model: {TRAINING_CONFIG['epochs']}")

    os.makedirs(TRAINING_CONFIG["model_save_dir"], exist_ok=True)

    n_gpu = max(1, TRAINING_CONFIG["device_count"])
    print(f"🚀 {len(MODELS_TO_TRAIN)} models  →  {n_gpu} GPU(s) (round-robin)")

    start_time = time.time()
    manager, results, procs = mp.Manager(), mp.Manager().dict(), []
    for idx, cfg in enumerate(MODELS_TO_TRAIN):
        gpu_id = idx % n_gpu
        p = mp.Process(target=worker, args=(cfg, gpu_id, results))
        p.start();  procs.append(p)

    for p in procs: p.join()
    
    # ---------------- summary --------------------------------------------------
    print("\n📊 Training results")
    for cfg in MODELS_TO_TRAIN:
        name, best = cfg["name"], results.get(cfg["name"])
        if best is not None:
            print(f"  ✅ {name:25s}  best={best:.6f}")
        else:
            print(f"  ❌ {name:25s}  failed")

    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time/60:.2f} minutes")

def run_compilation(use_dynamo=False):
    """Run the TensorRT compilation phase."""
    print("\n🔧 Starting TensorRT compilation...")
    
    os.makedirs(TRAINING_CONFIG["tensorrt_save_dir"], exist_ok=True)
    
    for model_config in MODELS_TO_TRAIN:
        model_name = model_config['name']
        
        model    = model_config["class"](name=model_name, **model_config["kwargs"])

        # Measure FLOPs and parameters
        try:
            print(f"Model Name: {model_config['name']}")
            custom_count = count_params(model)
            print(f"Model Parameters (custom counting): {custom_count / 1e3:.2f} K")
            dummy_input = torch.randn(1, 1, 1, 23, 23, 23)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            print(f"Model FLOPs (thop): {flops / 1e6:.2f} MFLOPs")
            print(f"Model Parameters (thop): {params / 1e3:.2f} K")
        except Exception as e:
            print(f"Could not measure FLOPs and parameters: {e}")
        
        sys.stdout.flush()
        if use_dynamo:
            ckpt_path = os.path.join(TRAINING_CONFIG['model_save_dir'], f"{model_name}_best.pt")
            if os.path.exists(ckpt_path):
                state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=True)
                compile_and_benchmark_model(model_name, model, TRAINING_CONFIG, model_config)
            else:
                print(f"  ⚠️  .pt file not found for {model_name}")
        else:
            jit_path = os.path.join(TRAINING_CONFIG['model_save_dir'], f"{model_name}_best.jit")            
            if os.path.exists(jit_path):
                model = torch.jit.load(jit_path)
                compile_and_benchmark_model(model_name, model, TRAINING_CONFIG, model_config)
            else:
                print(f"  ⚠️  JIT file not found for {model_name}")
    
    print("\n🎉 All models compiled successfully!")

def main():
    """Main function to orchestrate training and/or compilation based on arguments."""
    args = sys.argv[1:]  # Get command line arguments (excluding script name)
    
    # Check for valid arguments
    valid_args = {"train", "compile", "dynamo"}
    provided_args = set(args)
    
    if not provided_args.issubset(valid_args):
        invalid_args = provided_args - valid_args
        print(f"❌ Invalid arguments: {invalid_args}")
        print("Usage: python script.py [train] [compile] [dynamo]")
        print("  - No arguments: run both training and compilation")
        print("  - train: run only training")
        print("  - compile: run only compilation")
        print("  - train compile: run both training and compilation")
        return
    
    # Determine what to run based on arguments
    if len(args) == 0:
        # No arguments provided - run everything
        print("🚀 Running both training and compilation")
        run_training()
        run_compilation()
    if "train" in args:
        print("🚀 Running training")
        run_training()
    if "compile" in args:
        print("🚀 Running compilation")
        run_compilation("dynamo" in args)

if __name__ == "__main__":
    main()