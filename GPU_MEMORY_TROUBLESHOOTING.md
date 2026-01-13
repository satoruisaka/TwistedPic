# GPU Memory Troubleshooting Guide

## Problem: Ollama Running at 100% CPU After TwistedPic Use

### Symptoms
- TwistedPic generates images successfully
- After image generation, Ollama stops using GPU
- Ollama falls back to CPU mode and runs at 100% CPU usage
- Only way to fix is restarting both TwistedPic and Ollama

### Root Cause
When TwistedPic loads large models like `sd3_large` (Stable Diffusion 3.5 Large), it allocates significant GPU memory. Without proper cleanup, PyTorch/CUDA keeps this memory allocated even after image generation completes, preventing Ollama from accessing the GPU.

### Solution 1: Automatic GPU Cleanup (Recommended)

**Status**: Now enabled by default as of the latest update.

The system automatically clears GPU memory after each image generation:
- Clears PyTorch CUDA cache
- Runs garbage collection
- Frees GPU memory for Ollama to use

**Configuration in `config.py`:**
```python
GPU_MEMORY_CLEANUP = True  # Default: enabled
```

**Verification:**
When running TwistedPic with `verbose=True`, you'll see:
```
[ModelRegistry] GPU memory after cleanup: 2.45GB allocated, 2.50GB reserved
```

### Solution 2: Advanced Model Unloading (If Solution 1 Insufficient)

If basic cleanup doesn't free enough GPU memory, enable aggressive model unloading:

**Configuration in `config.py`:**
```python
UNLOAD_MODEL_AFTER_GENERATION = True  # Default: disabled
```

**Trade-offs:**
- ✅ Completely frees GPU memory between generations
- ✅ Ensures Ollama can always access GPU
- ❌ Slower: Each generation requires reloading model to GPU (~5-10 seconds overhead)
- ❌ More disk I/O and wear on hardware

**When to use:**
- You primarily use Ollama and occasionally generate images
- You have limited VRAM (<12GB)
- Basic cleanup isn't freeing enough memory

### Solution 3: Use Smaller Models

Switch from `sd3_large` to smaller models in the web UI:

**Model comparison:**
| Model | VRAM Usage | Speed | Quality |
|-------|-----------|-------|---------|
| `sd3_large` (SD 3.5 Large) | ~10GB | Fast on RTX 5090 | Excellent |
| `sdxl_base` (SDXL 1.0) | ~6GB | Very Fast | Very Good |
| `flux_dev` (Flux.1-dev) | ~12GB | Slow (not optimized) | Excellent |

**Recommendation**: Use `sdxl_base` if you need frequent image generation alongside Ollama.

### Solution 4: Monitor GPU Memory

Check GPU memory usage in real-time:

```bash
# Watch GPU memory every 2 seconds
watch -n 2 nvidia-smi

# Check current allocation
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
```

**Expected behavior after fix:**
- **During image generation**: High GPU usage (8-12GB depending on model)
- **After generation**: GPU memory drops significantly (1-3GB)
- **Ollama can then use GPU**: GPU usage increases when Ollama processes requests

### Testing the Fix

1. **Start TwistedPic**:
   ```bash
   cd TwistedPic
   python server.py
   ```

2. **Generate an image** via the web UI

3. **Check GPU memory** after generation:
   ```bash
   nvidia-smi
   ```
   Should show significantly reduced memory usage

4. **Test Ollama** (should use GPU, not CPU):
   ```bash
   curl http://localhost:11434/api/generate -d '{
     "model": "mistral:latest",
     "prompt": "Why is the sky blue?",
     "stream": false
   }'
   ```
   Monitor `nvidia-smi` - you should see GPU usage increase

5. **Check Ollama isn't using 100% CPU**:
   ```bash
   top  # or htop
   ```
   Ollama should use minimal CPU when GPU is working

### Configuration Summary

**Default settings (recommended for most users):**
```python
# config.py
GPU_MEMORY_CLEANUP = True
UNLOAD_MODEL_AFTER_GENERATION = False
```

**Aggressive settings (if Ollama still can't access GPU):**
```python
# config.py
GPU_MEMORY_CLEANUP = True
UNLOAD_MODEL_AFTER_GENERATION = True
```

### Additional Tips

1. **Restart order matters**: If both services get stuck:
   ```bash
   # Kill both services
   pkill -f "python server.py"
   pkill -f "ollama serve"
   
   # Start Ollama first
   ollama serve &
   
   # Then start TwistedPic
   cd TwistedPic && python server.py
   ```

2. **Monitor logs**: Run TwistedPic with verbose logging to see GPU cleanup:
   ```python
   # In server.py, line ~105
   model_registry = initialize_default_models(device="cuda", verbose=True, load_all=False)
   ```

3. **Check CUDA version compatibility**: Ensure your PyTorch and CUDA versions are compatible:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
   ```

### When to Contact Support

If after implementing these fixes you still experience:
- Ollama consistently using 100% CPU
- GPU memory not being freed after image generation
- Both services unable to share GPU access

Then open an issue with:
- GPU model and VRAM
- PyTorch version
- CUDA version
- Output of `nvidia-smi` before/after image generation
- TwistedPic logs with verbose=True
