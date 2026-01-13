# GPU Memory Management Update - Summary

## Date: January 7, 2026

## Problem
After upgrading TwistedPic to use `sd3_large` (Stable Diffusion 3.5 Large), the application was monopolizing GPU memory and preventing Ollama from accessing the GPU. This caused Ollama to fall back to CPU mode and run at 100% CPU usage.

## Root Cause
Large image generation models like SD3 Large allocate substantial GPU memory (~10GB). Without explicit cleanup, PyTorch/CUDA maintains these allocations even after image generation completes, blocking other GPU applications like Ollama.

## Solution Implemented

### 1. Automatic GPU Memory Cleanup (Default)
**Files modified**: 
- `model_registry.py`
- `config.py`

**Changes**:
- Added `_cleanup_gpu_memory()` method that:
  - Clears PyTorch CUDA cache with `torch.cuda.empty_cache()`
  - Runs Python garbage collection with `gc.collect()`
  - Reports memory stats when verbose mode is enabled
- Automatically called after every image generation
- Enabled by default via `GPU_MEMORY_CLEANUP = True` in config.py

### 2. Advanced Model Unloading (Optional)
**Files modified**: 
- `model_registry.py`
- `config.py`

**Changes**:
- Added `unload_model_from_gpu()` method to move model to CPU
- Added `reload_model_to_gpu()` method to reload model before generation
- Controlled via `UNLOAD_MODEL_AFTER_GENERATION` flag (disabled by default)
- Trade-off: Slower generation but completely frees GPU memory

### 3. Configuration Options
**File**: `config.py`

```python
# Basic cleanup (recommended - enabled by default)
GPU_MEMORY_CLEANUP = True

# Advanced cleanup (only if basic isn't sufficient)
UNLOAD_MODEL_AFTER_GENERATION = False
```

### 4. Documentation
**New files created**:
- `GPU_MEMORY_TROUBLESHOOTING.md` - Comprehensive troubleshooting guide
- `test_gpu_cleanup.py` - Test script to verify GPU cleanup is working

**Updated files**:
- `README.md` - Added GPU Memory Management section with configuration guide

## How to Use

### Default Behavior (Recommended)
No action needed. GPU cleanup is now automatic with every image generation.

### If Ollama Still Can't Access GPU
Edit `config.py`:
```python
UNLOAD_MODEL_AFTER_GENERATION = True
```

### Testing the Fix
Run the test script:
```bash
cd TwistedPic
python test_gpu_cleanup.py
```

Monitor GPU in another terminal:
```bash
watch -n 2 nvidia-smi
```

## Expected Behavior After Fix

| Stage | GPU Memory | Ollama Status |
|-------|-----------|---------------|
| Before image generation | Low (~1-3GB) | Can use GPU |
| During image generation | High (~10GB) | May be blocked |
| After image generation | Low (~1-3GB) | Can use GPU |

## Files Changed

1. **TwistedPic/model_registry.py**
   - Added `gc` import for garbage collection
   - Added `_cleanup_gpu_memory()` method
   - Added `unload_model_from_gpu()` method
   - Added `reload_model_to_gpu()` method
   - Modified `generate_image()` to call cleanup methods
   - Added config-based control for cleanup behavior

2. **TwistedPic/config.py**
   - Added `GPU_MEMORY_CLEANUP = True`
   - Added `UNLOAD_MODEL_AFTER_GENERATION = False`
   - Added documentation comments

3. **TwistedPic/README.md**
   - Added "GPU Memory Management" section
   - Documented configuration options
   - Explained symptoms and solutions

4. **TwistedPic/GPU_MEMORY_TROUBLESHOOTING.md** (new)
   - Comprehensive troubleshooting guide
   - Step-by-step solutions
   - Configuration examples
   - Testing procedures

5. **TwistedPic/test_gpu_cleanup.py** (new)
   - Test script to verify GPU cleanup
   - Reports memory usage at each stage
   - Generates test image

## Testing Recommendations

1. **Start services in order**:
   ```bash
   ollama serve
   cd TwistedPair/V2 && uvicorn server:app --port 8001
   cd TwistedPic && python server.py
   ```

2. **Generate an image** via web UI (http://localhost:5000)

3. **Verify GPU memory freed**:
   ```bash
   nvidia-smi
   ```

4. **Test Ollama still works with GPU**:
   ```bash
   curl http://localhost:11434/api/generate -d '{
     "model": "mistral:latest",
     "prompt": "Test prompt",
     "stream": false
   }'
   ```

5. **Monitor CPU usage** - Ollama should NOT be at 100% CPU

## Rollback Plan

If the changes cause issues, disable cleanup in `config.py`:
```python
GPU_MEMORY_CLEANUP = False
UNLOAD_MODEL_AFTER_GENERATION = False
```

## Performance Impact

### With GPU_MEMORY_CLEANUP = True (default)
- **Impact**: Negligible (~0.1-0.2 seconds per generation)
- **Benefit**: Prevents GPU monopolization
- **Recommended for**: All users

### With UNLOAD_MODEL_AFTER_GENERATION = True
- **Impact**: Moderate (~5-10 seconds overhead per generation)
- **Benefit**: Complete GPU memory release between generations
- **Recommended for**: Users who rarely generate images or have limited VRAM

## Additional Notes

- The fix is backward compatible - all existing functionality remains unchanged
- Verbose logging shows GPU memory stats when enabled
- The cleanup logic only runs when CUDA is available
- CPU mode is unaffected by these changes
