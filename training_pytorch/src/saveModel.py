import sys
import os
import torch
import torch.nn as nn

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dataset import *  # Assuming the dataset class is defined in dataset.py
from architecture import *  # Assuming model classes are in model_definitions.py
from train import train_model, load_model  # Assuming the generic train_model is in training_utils.py

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

def visualize_3d_mask(model, save_path="mask_visualization.png", figsize=(20, 15)):
    """
    Visualize the learned 3D mask from ZAxisSoftMask
    
    Args:
        model: Your trained model containing ZAxisSoftMask
        save_path: Where to save the visualization
        figsize: Figure size for the plot
    """
    
    # Extract the mask from your model
    if hasattr(model, 'mask'):
        mask_module = model.mask
    else:
        # If mask is nested deeper, adjust this
        mask_module = None
        for module in model.modules():
            if isinstance(module, ZAxisSoftMask):
                mask_module = module
                break
    
    if mask_module is None:
        print("Error: Could not find ZAxisSoftMask in the model")
        return
    
    # Get the learned mask values
    with torch.no_grad():
        mask_logits = mask_module.mask_logits  # (23, 23, 23)
        mask_values = torch.sigmoid(mask_logits).cpu().numpy()  # Apply sigmoid and convert to numpy
    
    # Create the subplot grid: 5 rows x 5 columns (25 subplots for 23 depth slices)
    fig, axes = plt.subplots(5, 5, figsize=figsize)
    fig.suptitle('Learned 3D Mask Visualization (Depth Slices)', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Custom colormap for better visualization
    colors = ['darkblue', 'blue', 'lightblue', 'white', 'orange', 'red', 'darkred']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('mask_cmap', colors, N=n_bins)
    
    # Plot each depth slice
    for depth_idx in range(23):
        ax = axes_flat[depth_idx]
        
        # Get the slice
        slice_data = mask_values[depth_idx, :, :]  # (23, 23)
        
        # Create the heatmap
        im = ax.imshow(slice_data, cmap=cmap, vmin=0, vmax=1, aspect='equal')
        
        # Add title with statistics
        slice_mean = np.mean(slice_data)
        slice_std = np.std(slice_data)
        # ax.set_title(f"Depth {depth_idx}, mean={slice_mean:.3f}, std={slice_std:.3f}")
        
        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add grid lines for better readability
        ax.grid(True, alpha=0.3, color='gray', linewidth=0.5)
    
    # Hide unused subplots (we have 23 slices but 25 subplots)
    for idx in range(23, 25):
        axes_flat[idx].set_visible(False)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=30)
    cbar.set_label('Mask Weight (0=ignored, 1=full attention)', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"3D mask visualization saved to: {save_path}")
    
    # Display statistics
    print("\n=== 3D Mask Statistics ===")
    print(f"Overall mean: {np.mean(mask_values):.4f}")
    print(f"Overall std:  {np.std(mask_values):.4f}")
    print(f"Min value:    {np.min(mask_values):.4f}")
    print(f"Max value:    {np.max(mask_values):.4f}")
    print(f"Sparsity:     {np.mean(mask_values < 0.1):.1%} (values < 0.1)")
    
    plt.show()
    
    return fig, mask_values


def quantize_model(model: nn.Module, dtype: torch.dtype, device='cpu') -> nn.Module:
    """
    Convert model weights to a given precision (float16, bfloat16, etc.)
    """
    model = model.to(dtype=dtype, device=device)
    for param in model.parameters():
        param.data = param.data.to(dtype)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(dtype)
    return model

def main():
    model_path = "/data/hector/model_ablation/GreensFacePredictor_5.5e-8.pt"
    output_path = "/data/hector/final_models/GreensFacePredictor_5.5e-8.pt.jit"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    target_dtype = torch.float32  # Change to torch.bfloat16 or torch.float32 as needed

    # Load the model (FP32)
    loaded_model = load_model(
        model_class=FacePredictor,
        model_path=model_path,
        device=device
    )

    # Quantize weights to desired precision
    # quantized_model = quantize_model(loaded_model, dtype=target_dtype, device=device)

    # visualize_3d_mask(loaded_model, save_path="mask_visualization_6.png")

    # Script and save the quantized model
    scripted_model = torch.jit.script(loaded_model)
    scripted_model.save(output_path)

    print(f"✅ Quantized model saved at: {output_path} (dtype={target_dtype})")

if __name__ == "__main__":
    main()