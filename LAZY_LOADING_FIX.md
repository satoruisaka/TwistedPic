# Lazy Loading Fix for GPU Memory Management

## Problem Statement

**Issue**: TwistedPic occupies GPU memory immediately at startup and doesn't release it until an image is generated.

**Impact**: 
- Other applications (like Ollama) cannot use GPU memory while TwistedPic is running
- Ollama falls back to CPU mode, causing 100% CPU usage and slow responses
- GPU memory is wasted even when TwistedPic is idle

## Root Cause

The original implementation in `server.py` loaded the default image model during startup:

```python
@app.on_event("startup")
async def startup_event():
    # This immediately loads SD 3.5 Large (~10GB) into GPU memory!
    model_registry = initialize_default_models(device="cuda", verbose=True, load_all=False)
```

Even though `GPU_MEMORY_CLEANUP` and `UNLOAD_MODEL_AFTER_GENERATION` were enabled in `config.py`, these only took effect **after** the first image generation. The model remained in GPU memory from startup until first use.

## Solution: Lazy Loading

We implemented **lazy loading** - models are only loaded into GPU memory when actually needed for image generation.

### Changes Made

#### 1. Server Startup ([server.py](server.py#L98-L123))
```python
@app.on_event("startup")
async def startup_event():
    # Initialize empty model registry (no models loaded)
    from model_registry import ModelRegistry
    model_registry = ModelRegistry(device="cuda", verbose=True)
    print("[Startup] Model registry initialized (no models loaded yet - lazy loading enabled)")
    # ... rest of initialization
    print("💡 GPU memory will remain free until first image generation request")
```

**Result**: TwistedPic starts up without loading any models into GPU memory.

#### 2. ModelRegistry.get_image_model() ([model_registry.py](model_registry.py#L162-L197))
Added lazy loading logic:
```python
def get_image_model(self, name: str):
    # 1. Check if model is currently loaded
    if name in self.image_models:
        return self.image_models[name]
    
    # 2. Check if model was unloaded (reload it)
    if hasattr(self, '_unloaded_models') and name in self._unloaded_models:
        self.reload_model_to_gpu(name)
        return self.image_models.get(name)
    
    # 3. LAZY LOADING: Check if model exists in config but hasn't been loaded yet
    if name in config.IMAGE_MODELS:
        model_info = config.IMAGE_MODELS[name]
        self.register_image_model(
            name=name,
            model_id=model_info["model_id"],
            pipeline_type=model_info.get("pipeline_type", "sdxl"),
            parameters=config.DEFAULT_IMAGE_PARAMS
        )
        return self.image_models.get(name)
    
    # Model not found
    return None
```

**Result**: Models are loaded on-demand when first requested for generation.

#### 3. ModelRegistry.list_models() ([model_registry.py](model_registry.py#L199-L213))
Updated to show all available models (loaded, unloaded, and config-defined):
```python
def list_models(self) -> list[str]:
    loaded_models = set(self.image_models.keys())
    
    # Include unloaded models
    if hasattr(self, '_unloaded_models'):
        unloaded_models = set(self._unloaded_models.keys())
        loaded_models = loaded_models | unloaded_models
    
    # Include models from config (lazy loading)
    config_models = set(config.IMAGE_MODELS.keys())
    
    return list(loaded_models | config_models)
```

**Result**: Web UI shows all available models even when none are loaded yet.

## Behavior After Fix

### Startup Phase
```
Initializing TwistedPic components...
[Startup] Model registry initialized (no models loaded yet - lazy loading enabled)
✅ All components initialized successfully
💡 GPU memory will remain free until first image generation request
```

**GPU Memory**: ~0-500MB (just PyTorch/CUDA overhead)  
**Ollama**: Can use GPU immediately

### First Image Generation Request
```
[ModelRegistry] Model sd3_large not loaded yet, loading now (lazy loading)...
[ModelRegistry] Loading SD3 model: stabilityai/stable-diffusion-3.5-large...
[ModelRegistry] ✅ Loaded sd3_large on cuda
```

**GPU Memory**: ~10GB (model loaded)  
**Generation Time**: +10-15 seconds overhead for first generation (model loading)

### After Image Generation Completes
With `UNLOAD_MODEL_AFTER_GENERATION = True` (default):
```
[ModelRegistry] Unloading sd3_large from GPU to free memory...
[ModelRegistry] GPU memory after model unload: 0.45GB allocated, 0.50GB reserved
```

**GPU Memory**: ~500MB (model unloaded)  
**Ollama**: Can use GPU again

### Subsequent Image Generation Requests
```
[ModelRegistry] Model sd3_large was unloaded, reloading automatically...
[ModelRegistry] Loading SD3 model: stabilityai/stable-diffusion-3.5-large...
```

**Behavior**: Model is reloaded on each generation (consistent overhead)

## Configuration Options

### Option 1: Maximum GPU Availability (Default - Recommended)
```python
# config.py
GPU_MEMORY_CLEANUP = True
UNLOAD_MODEL_AFTER_GENERATION = True  # Aggressive unloading
```

**Use case**: You primarily use Ollama and occasionally generate images  
**Trade-off**: +10-15 seconds overhead per generation (model reload)  
**GPU freed**: ~9.5GB between generations

### Option 2: Balanced Performance
```python
# config.py
GPU_MEMORY_CLEANUP = True
UNLOAD_MODEL_AFTER_GENERATION = False  # Keep model loaded
```

**Use case**: Frequent image generation sessions  
**Trade-off**: GPU memory stays occupied after first generation  
**GPU freed**: ~2-3GB (cache cleanup only)

### Option 3: Maximum Speed
```python
# config.py
GPU_MEMORY_CLEANUP = False
UNLOAD_MODEL_AFTER_GENERATION = False
```

**Use case**: Dedicated image generation workstation  
**Trade-off**: GPU memory monopolized by TwistedPic  
**GPU freed**: None

## Testing the Fix

### 1. Test Startup (GPU Should Be Free)
```bash
# Terminal 1: Start TwistedPic
cd /home/sator/project/TwistedPic
uvicorn server:app --host 0.0.0.0 --port 5000

# Terminal 2: Check GPU memory
watch -n 2 nvidia-smi

# Expected: ~500MB used (just PyTorch overhead)
```

### 2. Test Ollama Access (Before First Generation)
```bash
# Ollama should use GPU immediately
curl http://localhost:11434/api/generate -d '{
  "model": "mistral:latest",
  "prompt": "What is the capital of France?",
  "stream": false
}'

# Check nvidia-smi: Ollama should be using GPU
```

### 3. Test First Image Generation
```bash
# Generate an image via web UI at http://localhost:5000
# Expected console output:
# [ModelRegistry] Model sd3_large not loaded yet, loading now (lazy loading)...
# [ModelRegistry] Loading SD3 model...
# [ModelRegistry] ✅ Loaded sd3_large on cuda
# ... generation happens ...
# [ModelRegistry] Unloading sd3_large from GPU to free memory...
# [ModelRegistry] GPU memory after model unload: 0.45GB allocated

# Check nvidia-smi: Memory should drop back to ~500MB
```

### 4. Test Ollama Access (After Generation)
```bash
# Ollama should regain GPU access
curl http://localhost:11434/api/generate -d '{
  "model": "mistral:latest",
  "prompt": "Explain quantum computing",
  "stream": false
}'

# Check nvidia-smi: Ollama should be using GPU again
```

## Performance Impact

| Scenario | Before Fix | After Fix (Default) |
|----------|-----------|---------------------|
| **Startup GPU usage** | ~10GB (model loaded) | ~500MB (no model) |
| **Idle GPU usage** | ~10GB (model stays loaded) | ~500MB (model unloaded) |
| **First generation** | Fast (model ready) | +10-15s (model loading) |
| **Subsequent generations** | Fast | +10-15s each (model reload) |
| **Ollama availability** | Blocked after startup | Always available |

## Monitoring GPU Memory

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Memory usage breakdown
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# Process-specific memory
nvidia-smi pmon -c 1

# Detailed breakdown
nvidia-smi -q -d MEMORY
```

## Compatibility

- ✅ Works with all model types (Flux, SD3, SDXL)
- ✅ Compatible with existing GPU_MEMORY_CLEANUP settings
- ✅ Compatible with UNLOAD_MODEL_AFTER_GENERATION settings
- ✅ No breaking changes to API or web UI
- ✅ Backward compatible with existing workflows

## Future Enhancements

1. **Smart Model Caching**: Keep model loaded for N minutes after last use
2. **Multi-model LRU Cache**: Keep 2-3 recently used models loaded
3. **Model Priority System**: Never unload certain "hot" models
4. **GPU Memory Threshold**: Auto-unload only if memory pressure detected

## Related Files

- [server.py](server.py) - Startup initialization
- [model_registry.py](model_registry.py) - Model management and lazy loading
- [config.py](config.py) - Configuration options
- [GPU_MEMORY_TROUBLESHOOTING.md](GPU_MEMORY_TROUBLESHOOTING.md) - General troubleshooting
- [GPU_QUICK_REFERENCE.md](GPU_QUICK_REFERENCE.md) - Quick commands

## Summary

**Before**: TwistedPic loaded models at startup and held GPU memory hostage  
**After**: TwistedPic starts with empty GPU, loads models on-demand, frees GPU after use  
**Benefit**: Ollama and other applications can use GPU when TwistedPic is idle  
**Cost**: +10-15 seconds per image generation (model loading overhead)

**Recommended for**: Systems where GPU is shared between multiple applications (Ollama + TwistedPic + others).
