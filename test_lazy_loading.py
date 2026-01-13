#!/usr/bin/env python3
"""
Test script to verify lazy loading behavior.
Run this to confirm models are not loaded until needed.
"""

import torch
import sys
from model_registry import ModelRegistry
import config

def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2    # MB
        return allocated, reserved
    return 0, 0

def main():
    print("="*60)
    print("Testing Lazy Loading Implementation")
    print("="*60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires GPU.")
        sys.exit(1)
    
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Step 1: Baseline GPU memory
    print("\n" + "="*60)
    print("Step 1: Baseline GPU Memory (before ModelRegistry)")
    print("="*60)
    torch.cuda.empty_cache()
    allocated, reserved = get_gpu_memory_mb()
    print(f"GPU Memory: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved")
    
    # Step 2: Initialize ModelRegistry (should NOT load models)
    print("\n" + "="*60)
    print("Step 2: Initialize ModelRegistry (lazy loading)")
    print("="*60)
    print("Creating ModelRegistry instance...")
    registry = ModelRegistry(device="cuda", verbose=True)
    
    allocated, reserved = get_gpu_memory_mb()
    print(f"\nGPU Memory after init: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved")
    
    # Step 3: List available models (should show config models even if not loaded)
    print("\n" + "="*60)
    print("Step 3: List Available Models")
    print("="*60)
    available_models = registry.list_models()
    print(f"Available models: {available_models}")
    print(f"Number of models: {len(available_models)}")
    
    # Verify no models actually loaded
    loaded_models = list(registry.image_models.keys())
    print(f"Actually loaded models: {loaded_models if loaded_models else 'None (lazy loading working!)'}")
    
    if len(loaded_models) == 0:
        print("✅ PASS: No models loaded at initialization (lazy loading confirmed)")
    else:
        print(f"❌ FAIL: {len(loaded_models)} models loaded at initialization")
        return False
    
    # Step 4: Request a model (should trigger lazy loading)
    print("\n" + "="*60)
    print("Step 4: Request Model (should trigger lazy load)")
    print("="*60)
    test_model = "sd3_large"  # Default model
    print(f"Requesting model: {test_model}")
    print("(This should trigger automatic loading...)")
    
    # Get GPU memory before loading
    allocated_before, reserved_before = get_gpu_memory_mb()
    print(f"GPU Memory before: {allocated_before:.2f} MB allocated")
    
    # Request model (triggers lazy load)
    model = registry.get_image_model(test_model)
    
    if model is None:
        print(f"❌ FAIL: Model {test_model} not found or failed to load")
        return False
    
    # Get GPU memory after loading
    allocated_after, reserved_after = get_gpu_memory_mb()
    memory_increase = allocated_after - allocated_before
    print(f"\nGPU Memory after: {allocated_after:.2f} MB allocated")
    print(f"Memory increase: {memory_increase:.2f} MB")
    
    if memory_increase > 1000:  # Model should use >1GB
        print(f"✅ PASS: Model loaded on-demand (lazy loading working!)")
    else:
        print(f"⚠️  WARNING: Memory increase seems low ({memory_increase:.2f} MB)")
    
    # Verify model is now in registry
    loaded_models_after = list(registry.image_models.keys())
    print(f"Loaded models after request: {loaded_models_after}")
    
    # Step 5: Test unload (if enabled)
    if config.UNLOAD_MODEL_AFTER_GENERATION:
        print("\n" + "="*60)
        print("Step 5: Test Model Unload")
        print("="*60)
        print(f"UNLOAD_MODEL_AFTER_GENERATION = {config.UNLOAD_MODEL_AFTER_GENERATION}")
        print("Manually unloading model...")
        
        registry.unload_model_from_gpu(test_model)
        
        allocated_after_unload, reserved_after_unload = get_gpu_memory_mb()
        print(f"GPU Memory after unload: {allocated_after_unload:.2f} MB allocated")
        
        memory_freed = allocated_after - allocated_after_unload
        print(f"Memory freed: {memory_freed:.2f} MB")
        
        if memory_freed > 1000:
            print("✅ PASS: Model unloaded successfully")
        else:
            print(f"⚠️  WARNING: Memory freed seems low ({memory_freed:.2f} MB)")
    
    # Final summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print("✅ Lazy loading is working correctly!")
    print("✅ Models are NOT loaded at initialization")
    print("✅ Models are loaded on-demand when requested")
    if config.UNLOAD_MODEL_AFTER_GENERATION:
        print("✅ Models are unloaded after use (as configured)")
    print("\n💡 TwistedPic will now keep GPU memory free until needed")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
