import torch
import torch.nn as nn
import time
import os
import sys
import json
import argparse
import numpy as np
from thop import profile
from thop import clever_format

# Add parent directory to path to allow importing mambavision and models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ladder_net import MambaLadder

def measure_efficiency():
    # 0. Parse Arguments
    parser = argparse.ArgumentParser(description='Measure Model Efficiency')
    parser.add_argument('--output', type=str, default='model_efficiency.json', help='Path to save results JSON')
    args = parser.parse_args()

    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on: {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    else:
        gpu_name = "CPU"
    
    # 2. Instantiate Model
    model = MambaLadder(pretrained=False).to(device)
    model.eval()
    
    # 3. Create Dummy Inputs
    # Based on config: Batch Size 1 for latency, Image 512x512, Seq 40
    B = 1
    C, H, W = 3, 512, 512
    T, T_C = 40, 7 
    
    dummy_image = torch.randn(B, C, H, W).to(device)
    dummy_weather = torch.randn(B, T, T_C).to(device)
    
    # Dictionary to store results
    results = {
        "device": gpu_name,
        "input_resolution": f"{H}x{W}"
    }

    # ---------------------------------------------------------
    # A. FLOPs and Parameters (using thop)
    # ---------------------------------------------------------
    print("\n--- Calculating FLOPs and Params ---")
    # thop.profile needs inputs passed as a tuple
    macs, params = profile(model, inputs=(dummy_image, dummy_weather), verbose=False)
    
    # Convert to formatted strings (for display)
    flops_str, params_str = clever_format([macs, params], "%.3f")
    
    # Store raw values in JSON for plotting later
    results["params_M"] = params / 1e6  # Millions
    results["flops_G"] = macs / 1e9     # GFLOPs (approx)
    results["params_display"] = params_str
    results["flops_display"] = flops_str
    
    print(f"Total Parameters: {params_str}")
    print(f"Total MACs (GFLOPs): {flops_str}")
    
    # ---------------------------------------------------------
    # B. Inference Latency (FPS)
    # ---------------------------------------------------------
    print("\n--- Measuring Latency (GPU Warmup 10 iter) ---")
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_image, dummy_weather)
            
    # Measure
    iterations = 100
    print(f"Running {iterations} iterations...")
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(dummy_image, dummy_weather)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / iterations
    fps = 1 / avg_time
    
    results["latency_ms"] = avg_time * 1000
    results["throughput_fps"] = fps
    
    print(f"Average Inference Time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    
    # ---------------------------------------------------------
    # C. Model File Size
    # ---------------------------------------------------------
    print("\n--- Model Size on Disk ---")
    # Save a temporary checkpoint to measure size
    temp_path = "temp_model_size.pth"
    torch.save(model.state_dict(), temp_path)
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    
    results["model_size_MB"] = size_mb
    print(f"Checkpoint Size: {size_mb:.2f} MB")
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    # ---------------------------------------------------------
    # D. Save to JSON
    # ---------------------------------------------------------
    # Ensure directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    measure_efficiency()