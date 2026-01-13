#!/usr/bin/env python3
"""
Test GPU memory cleanup functionality.

This script verifies that GPU memory is properly freed after image generation.
Run this with nvidia-smi in another terminal to watch memory usage.
"""

import torch
import time
from model_registry import initialize_default_models

def get_gpu_memory_info():
    """Get current GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0}
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - allocated
    
    return {
        "allocated": allocated,
        "reserved": reserved,
        "free": free,
        "total": total
    }

def main():
    print("=" * 70)
    print("GPU Memory Cleanup Test")
    print("=" * 70)
    
    # Check initial GPU state
    print("\n1. Initial GPU state (before loading model):")
    mem = get_gpu_memory_info()
    print(f"   Allocated: {mem['allocated']:.2f}GB")
    print(f"   Reserved:  {mem['reserved']:.2f}GB")
    print(f"   Free:      {mem['free']:.2f}GB")
    print(f"   Total:     {mem['total']:.2f}GB")
    
    # Load model
    print("\n2. Loading SD3 Large model...")
    registry = initialize_default_models(device="cuda", verbose=True, load_all=False)
    
    print("\n3. GPU state after model loading:")
    mem = get_gpu_memory_info()
    print(f"   Allocated: {mem['allocated']:.2f}GB")
    print(f"   Reserved:  {mem['reserved']:.2f}GB")
    print(f"   Free:      {mem['free']:.2f}GB")
    
    # Generate image
    print("\n4. Generating test image...")
    image = registry.generate_image(
        model_name="sd3_large",
        prompt="A serene mountain landscape at sunset, digital art, highly detailed",
        num_inference_steps=20,  # Quick test
        guidance_scale=4.5,
        resolution=(1024, 768),
        seed=42
    )
    
    print(f"\n5. Image generated: {image.size}")
    
    # Check memory after generation (cleanup happens automatically)
    print("\n6. GPU state after generation (with automatic cleanup):")
    mem = get_gpu_memory_info()
    print(f"   Allocated: {mem['allocated']:.2f}GB")
    print(f"   Reserved:  {mem['reserved']:.2f}GB")
    print(f"   Free:      {mem['free']:.2f}GB")
    
    # Wait and check again
    print("\n7. Waiting 5 seconds and checking again...")
    time.sleep(5)
    mem = get_gpu_memory_info()
    print(f"   Allocated: {mem['allocated']:.2f}GB")
    print(f"   Reserved:  {mem['reserved']:.2f}GB")
    print(f"   Free:      {mem['free']:.2f}GB")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("\nExpected behavior:")
    print("- Step 3: High memory usage (model loaded)")
    print("- Step 6: Memory should be significantly reduced")
    print("- Step 7: Memory should remain low")
    print("\nIf memory doesn't decrease in steps 6-7, GPU cleanup may not be working.")
    print("=" * 70)
    
    # Optionally save test image
    image.save("test_gpu_cleanup.png")
    print("\nTest image saved as: test_gpu_cleanup.png")

if __name__ == "__main__":
    main()
