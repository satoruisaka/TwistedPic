# Quick Reference: GPU Memory Management

## Problem
TwistedPic monopolizes GPU after generating images, causing Ollama to run at 100% CPU.

## Solution
Automatic GPU cleanup now enabled by default.

## Configuration (config.py)

```python
# ✅ RECOMMENDED: Basic cleanup (enabled by default)
GPU_MEMORY_CLEANUP = True
UNLOAD_MODEL_AFTER_GENERATION = False

# 🔧 IF ISSUES PERSIST: Aggressive cleanup (slower but frees more memory)
GPU_MEMORY_CLEANUP = True
UNLOAD_MODEL_AFTER_GENERATION = True
```

## Quick Test

1. Generate an image in TwistedPic
2. Check GPU memory freed:
   ```bash
   nvidia-smi
   ```
3. Test Ollama still uses GPU (not CPU):
   ```bash
   curl http://localhost:11434/api/generate -d '{"model":"mistral:latest","prompt":"test"}'
   ```

## Expected Results

| Status | GPU Memory | Ollama CPU |
|--------|-----------|------------|
| ✅ Working | Drops after generation | Normal (~10-30%) |
| ❌ Not working | Stays high | 100% |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Ollama at 100% CPU | Enable `GPU_MEMORY_CLEANUP = True` |
| Still 100% CPU | Enable `UNLOAD_MODEL_AFTER_GENERATION = True` |
| Slow generation | Disable `UNLOAD_MODEL_AFTER_GENERATION` |
| Out of memory | Use `sdxl_base` instead of `sd3_large` |

## Documentation
- Full guide: [GPU_MEMORY_TROUBLESHOOTING.md](GPU_MEMORY_TROUBLESHOOTING.md)
- Update summary: [GPU_CLEANUP_UPDATE_SUMMARY.md](GPU_CLEANUP_UPDATE_SUMMARY.md)
- Test script: [test_gpu_cleanup.py](test_gpu_cleanup.py)

## Monitor GPU in Real-Time

```bash
watch -n 2 nvidia-smi
```

Look for memory usage to drop after each image generation.
