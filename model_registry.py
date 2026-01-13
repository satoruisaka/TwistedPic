"""
model_registry.py - Image Model Registry for TwistedPic

Manages Stable Diffusion XL models for image generation.
Adapted from DreamSprout with configurable parameters.
"""

import torch
import gc
from diffusers import StableDiffusionXLPipeline, FluxPipeline, StableDiffusion3Pipeline
from compel import Compel, ReturnedEmbeddingsType
from typing import Dict, Any, Optional
import config


class ModelRegistry:
    """Registry for managing image generation models."""
    
    def __init__(self, device: str = "cuda", verbose: bool = False):
        """
        Initialize model registry.
        
        Args:
            device: Device to run models on ("cuda" or "cpu")
            verbose: Enable debug logging
        """
        self.device = device
        self.verbose = verbose
        self.image_models: Dict[str, Any] = {}
        
        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA not available, falling back to CPU (will be slow)")
            self.device = "cpu"
    
    def register_image_model(
        self,
        name: str,
        model_id: str,
        pipeline_type: str = "sdxl",
        parameters: Optional[Dict[str, Any]] = None,
        dtype: torch.dtype = None
    ) -> None:
        """
        Register an image generation model (Flux, SD3, or SDXL).
        
        Args:
            name: Registry name for the model
            model_id: HuggingFace model ID
            pipeline_type: Type of pipeline ("flux", "sd3", or "sdxl")
            parameters: Generation parameters (steps, CFG, resolution, etc.)
            dtype: torch dtype (default: float16 for CUDA, float32 for CPU)
        """
        if self.verbose:
            print(f"[ModelRegistry] Loading {pipeline_type.upper()} model: {model_id}...")
        
        # Set dtype based on device
        if dtype is None:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Select pipeline class based on type
        if pipeline_type == "flux":
            pipeline_class = FluxPipeline
        elif pipeline_type == "sd3":
            pipeline_class = StableDiffusion3Pipeline
        elif pipeline_type == "sdxl":
            pipeline_class = StableDiffusionXLPipeline
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}. Must be 'flux', 'sd3', or 'sdxl'")
        
        # Load pipeline
        try:
            if self.device == "cuda":
                # Set device_map based on pipeline type
                # Flux and SD3 support "balanced" or "cuda", SDXL works best with direct .to()
                if pipeline_type in ["flux", "sd3"]:
                    pipe = pipeline_class.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        use_safetensors=True
                    ).to(self.device)
                    
                    # Enable memory efficient attention for Flux/SD3
                    if hasattr(pipe, 'enable_model_cpu_offload'):
                        # Don't use CPU offload - keep everything on GPU
                        pass
                    if hasattr(pipe, 'enable_vae_slicing'):
                        pipe.enable_vae_slicing()
                    if hasattr(pipe, 'enable_vae_tiling'):
                        pipe.enable_vae_tiling()
                else:
                    # SDXL
                    pipe = pipeline_class.from_pretrained(
                        model_id,
                        torch_dtype=dtype,
                        use_safetensors=True,
                        variant="fp16"
                    ).to(self.device)
                    
                    # Enable optimizations for SDXL
                    if hasattr(pipe, 'enable_vae_slicing'):
                        pipe.enable_vae_slicing()
                    if hasattr(pipe, 'enable_vae_tiling'):
                        pipe.enable_vae_tiling()
                
                # Use PyTorch's native scaled_dot_product_attention
                # xformers doesn't support RTX 5090 (compute capability 12.0) yet
                if self.verbose:
                    print(f"[ModelRegistry] Using PyTorch native attention (RTX 5090 optimized)")
                    print(f"[ModelRegistry] Enabled VAE optimizations for faster generation")
            else:
                # CPU mode - simpler loading
                pipe = pipeline_class.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    low_cpu_mem_usage=False
                ).to(self.device)
            
            # Store generation parameters
            default_params = config.DEFAULT_IMAGE_PARAMS.copy()
            if parameters:
                default_params.update(parameters)
            
            pipe.generation_config = default_params
            pipe.pipeline_type = pipeline_type  # Store for later use
            pipe.model_id = model_id  # Store model ID for reloading
            
            # Initialize Compel only for SDXL (Flux and SD3 have their own long prompt support)
            if pipeline_type == "sdxl":
                try:
                    compel = Compel(
                        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                        requires_pooled=[False, True]
                    )
                    pipe.compel = compel
                    if self.verbose:
                        print(f"[ModelRegistry] Compel enabled for long prompts (>77 tokens)")
                except Exception as e:
                    print(f"[ModelRegistry] ⚠️  Compel initialization failed: {e}")
                    print(f"[ModelRegistry] Long prompts will be truncated to 77 tokens")
                    pipe.compel = None
            else:
                # Flux and SD3 have native long prompt support (no Compel needed)
                pipe.compel = None
                if self.verbose:
                    print(f"[ModelRegistry] Native long prompt support (no Compel needed)")
            
            # Register model
            self.image_models[name] = pipe
            
            if self.verbose:
                print(f"[ModelRegistry] ✅ Loaded {name} on {self.device}")
                print(f"[ModelRegistry] Parameters: {pipe.generation_config}")
        
        except Exception as e:
            print(f"[ModelRegistry] ❌ Failed to load {model_id}: {e}")
            raise
    
    def get_image_model(self, name: str):
        """
        Get registered image model by name.
        If model was unloaded, it will be automatically reloaded.
        If model was never loaded (lazy loading), it will be loaded now.
        
        Args:
            name: Registry name of the model
            
        Returns:
            Diffusers pipeline or None if not found
        """
        # Check if model is currently loaded
        if name in self.image_models:
            return self.image_models[name]
        
        # Check if model was unloaded and can be reloaded
        if hasattr(self, '_unloaded_models') and name in self._unloaded_models:
            if self.verbose:
                print(f"[ModelRegistry] Model {name} was unloaded, reloading automatically...")
            self.reload_model_to_gpu(name)
            return self.image_models.get(name)
        
        # Lazy loading: Check if model exists in config but hasn't been loaded yet
        if name in config.IMAGE_MODELS:
            if self.verbose:
                print(f"[ModelRegistry] Model {name} not loaded yet, loading now (lazy loading)...")
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
    
    def list_models(self) -> list[str]:
        """
        Get list of available model names.
        Includes loaded models, unloaded models, and models defined in config (lazy loading).
        """
        loaded_models = set(self.image_models.keys())
        
        # Include unloaded models (can be reloaded on demand)
        if hasattr(self, '_unloaded_models'):
            unloaded_models = set(self._unloaded_models.keys())
            loaded_models = loaded_models | unloaded_models
        
        # Include models from config (lazy loading - not loaded yet but available)
        config_models = set(config.IMAGE_MODELS.keys())
        
        # Return union of all available models
        return list(loaded_models | config_models)
    
    def generate_image(
        self,
        model_name: str,
        prompt: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        resolution: Optional[tuple[int, int]] = None,
        seed: Optional[int] = None,
        use_random_seed: bool = False,
        negative_prompt: Optional[str] = None
    ) -> Any:
        """
        Generate image using registered model.
        
        Args:
            model_name: Name of registered model
            prompt: Text prompt for image generation
            num_inference_steps: Number of denoising steps (quality)
            guidance_scale: Classifier-free guidance scale (prompt adherence)
            resolution: (width, height) tuple
            seed: Random seed for reproducibility
            use_random_seed: If True, ignore seed parameter
            negative_prompt: Things to avoid in image
            
        Returns:
            Generated image (PIL Image)
            
        Raises:
            ValueError: If model not found
        """
        pipe = self.get_image_model(model_name)
        if pipe is None:
            raise ValueError(f"Model '{model_name}' not registered")
        
        # get_image_model() already handles reloading if model was unloaded
        
        # Use model's config as defaults, override with provided params
        params = pipe.generation_config.copy()
        
        if num_inference_steps is not None:
            params["num_inference_steps"] = num_inference_steps
        
        if guidance_scale is not None:
            params["guidance_scale"] = guidance_scale
        
        if resolution is not None:
            params["resolution"] = resolution
        
        if negative_prompt is not None:
            params["negative_prompt"] = negative_prompt
        
        # Handle seed
        if use_random_seed:
            generator = None
        else:
            seed_value = seed if seed is not None else params.get("seed", 42)
            generator = torch.Generator(device=self.device).manual_seed(seed_value)
        
        # Extract width and height
        width, height = params["resolution"]
        
        if self.verbose:
            print(f"[ModelRegistry] Generating image...")
            print(f"  Prompt: {prompt[:100]}...")
            print(f"  Steps: {params['num_inference_steps']}")
            print(f"  CFG Scale: {params['guidance_scale']}")
            print(f"  Resolution: {width}x{height}")
            if not use_random_seed:
                print(f"  Seed: {seed_value}")
        
        # Check if prompt exceeds token limits and handle accordingly
        pipeline_type = getattr(pipe, 'pipeline_type', 'sdxl')
        use_compel = pipeline_type == "sdxl" and hasattr(pipe, 'compel') and pipe.compel is not None
        
        # Generate image
        try:
            if use_compel:
                # SDXL with Compel for long prompt support
                if self.verbose:
                    print(f"[ModelRegistry] Using Compel for prompt encoding (supports >77 tokens)")
                
                # Build conditioning with Compel
                conditioning, pooled = pipe.compel(prompt)
                
                # Generate with pre-computed embeddings
                result = pipe(
                    prompt_embeds=conditioning,
                    pooled_prompt_embeds=pooled,
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=params["guidance_scale"],
                    width=width,
                    height=height,
                    generator=generator,
                    negative_prompt=params.get("negative_prompt")
                )
            elif pipeline_type == "flux":
                # Flux.1 pipeline
                if self.verbose:
                    print(f"[ModelRegistry] Using Flux pipeline (native long prompt support)")
                
                result = pipe(
                    prompt=prompt,
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=params["guidance_scale"],
                    width=width,
                    height=height,
                    generator=generator
                )
            elif pipeline_type == "sd3":
                # SD3 pipeline
                if self.verbose:
                    print(f"[ModelRegistry] Using SD3 pipeline (native long prompt support)")
                
                result = pipe(
                    prompt=prompt,
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=params["guidance_scale"],
                    width=width,
                    height=height,
                    generator=generator,
                    negative_prompt=params.get("negative_prompt")
                )
            else:
                # Standard SDXL generation (77 token limit)
                if self.verbose:
                    print(f"[ModelRegistry] Using standard prompt encoding (77 token limit)")
                
                result = pipe(
                    prompt=prompt,
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=params["guidance_scale"],
                    width=width,
                    height=height,
                    generator=generator,
                    negative_prompt=params.get("negative_prompt")
                )
            
            image = result.images[0]
            
            if self.verbose:
                print(f"[ModelRegistry] ✅ Image generated successfully")
            
            # Clean up GPU memory after generation to prevent monopolizing GPU
            if config.GPU_MEMORY_CLEANUP:
                self._cleanup_gpu_memory()
            
            # Advanced: Unload model from GPU if enabled (more aggressive)
            if config.UNLOAD_MODEL_AFTER_GENERATION:
                self.unload_model_from_gpu(model_name)
            
            return image
        
        except Exception as e:
            print(f"[ModelRegistry] ❌ Image generation failed: {e}")
            raise
    
    def _cleanup_gpu_memory(self) -> None:
        """
        Clean up GPU memory after image generation.
        
        This prevents TwistedPic from monopolizing GPU and allows Ollama
        to access GPU resources. Without this, Ollama may fall back to CPU
        mode and run at 100% CPU usage.
        """
        if self.device == "cuda" and torch.cuda.is_available():
            # Clear PyTorch CUDA cache
            torch.cuda.empty_cache()
            
            # Run garbage collection to free Python objects
            gc.collect()
            
            if self.verbose:
                # Show GPU memory stats after cleanup
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                print(f"[ModelRegistry] GPU memory after cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def unload_model_from_gpu(self, model_name: str) -> None:
        """
        Unload model from GPU to free GPU memory.
        
        This is an aggressive memory management strategy that completely
        frees GPU memory but requires reloading the model for next use.
        Stores model configuration so it can be recreated on next use.
        Use only if basic cleanup isn't sufficient.
        
        Args:
            model_name: Name of model to unload from GPU
        """
        if self.device == "cuda" and model_name in self.image_models:
            pipe = self.image_models[model_name]
            
            if self.verbose:
                print(f"[ModelRegistry] Unloading {model_name} from GPU to free memory...")
            
            # Store model config for later reloading
            if not hasattr(self, '_unloaded_models'):
                self._unloaded_models = {}
            
            self._unloaded_models[model_name] = {
                'model_id': pipe.model_id if hasattr(pipe, 'model_id') else None,
                'pipeline_type': getattr(pipe, 'pipeline_type', 'sdxl'),
                'generation_config': pipe.generation_config if hasattr(pipe, 'generation_config') else {}
            }
            
            # Explicitly delete pipeline components to free GPU memory
            # This is more aggressive than just deleting the pipeline reference
            pipeline_type = getattr(pipe, 'pipeline_type', 'sdxl')
            
            try:
                # Delete components based on pipeline type
                if pipeline_type == "flux":
                    # Flux components
                    if hasattr(pipe, 'transformer'): del pipe.transformer
                    if hasattr(pipe, 'text_encoder'): del pipe.text_encoder
                    if hasattr(pipe, 'text_encoder_2'): del pipe.text_encoder_2
                    if hasattr(pipe, 'vae'): del pipe.vae
                elif pipeline_type == "sd3":
                    # SD3 components
                    if hasattr(pipe, 'transformer'): del pipe.transformer
                    if hasattr(pipe, 'text_encoder'): del pipe.text_encoder
                    if hasattr(pipe, 'text_encoder_2'): del pipe.text_encoder_2
                    if hasattr(pipe, 'text_encoder_3'): del pipe.text_encoder_3
                    if hasattr(pipe, 'vae'): del pipe.vae
                elif pipeline_type == "sdxl":
                    # SDXL components
                    if hasattr(pipe, 'unet'): del pipe.unet
                    if hasattr(pipe, 'text_encoder'): del pipe.text_encoder
                    if hasattr(pipe, 'text_encoder_2'): del pipe.text_encoder_2
                    if hasattr(pipe, 'vae'): del pipe.vae
            except Exception as e:
                if self.verbose:
                    print(f"[ModelRegistry] Warning during component cleanup: {e}")
            
            # Delete the pipeline itself
            del self.image_models[model_name]
            del pipe
            
            # Aggressive GPU memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()  # Call twice for better cleanup
            
            if self.verbose:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"[ModelRegistry] GPU memory after model unload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def reload_model_to_gpu(self, model_name: str) -> None:
        """
        Reload model to GPU (recreates it if it was unloaded).
        
        Args:
            model_name: Name of model to reload to GPU
        """
        # Check if model is already loaded
        if model_name in self.image_models:
            # Model is already on GPU, nothing to do
            return
        
        # Check if model was previously unloaded
        if hasattr(self, '_unloaded_models') and model_name in self._unloaded_models:
            if self.verbose:
                print(f"[ModelRegistry] Reloading {model_name} to GPU...")
            
            # Retrieve stored config
            model_config = self._unloaded_models[model_name]
            model_id = model_config.get('model_id')
            pipeline_type = model_config.get('pipeline_type', 'sdxl')
            generation_config = model_config.get('generation_config', {})
            
            # If model_id not stored, try to get it from config
            if model_id is None:
                if model_name in config.IMAGE_MODELS:
                    model_id = config.IMAGE_MODELS[model_name]["model_id"]
                else:
                    raise ValueError(f"Cannot reload {model_name}: model_id not found")
            
            # Re-register the model
            self.register_image_model(
                name=model_name,
                model_id=model_id,
                pipeline_type=pipeline_type,
                parameters=generation_config
            )
            
            # Remove from unloaded list
            del self._unloaded_models[model_name]
            
            if self.verbose:
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"[ModelRegistry] GPU memory after model reload: {allocated:.2f}GB allocated")


def initialize_default_models(device: str = "cuda", verbose: bool = False, load_all: bool = False) -> ModelRegistry:
    """
    Initialize ModelRegistry with image generation model(s).
    
    Args:
        device: Device to run models on ("cuda" or "cpu")
        verbose: Enable debug logging
        load_all: If True, load all models; if False, load only default models
        
    Returns:
        Initialized ModelRegistry
    """
    registry = ModelRegistry(device=device, verbose=verbose)
    loaded_models = []
    failed_models = []
    
    # Register models (all or just defaults)
    for model_key, model_info in config.IMAGE_MODELS.items():
        should_load = load_all or model_info.get("default", False)
        
        if should_load:
            if verbose:
                status = "(default)" if model_info.get("default", False) else ""
                print(f"\n[Initialization] Loading {model_key} {status}...")
            
            try:
                registry.register_image_model(
                    name=model_key,
                    model_id=model_info["model_id"],
                    pipeline_type=model_info.get("pipeline_type", "sdxl"),
                    parameters=config.DEFAULT_IMAGE_PARAMS
                )
                loaded_models.append(model_key)
            except Exception as e:
                failed_models.append((model_key, str(e)))
                if verbose:
                    print(f"[Initialization] ⚠️  Skipping {model_key} - failed to load")
                    if "gated" in str(e).lower() or "403" in str(e):
                        print(f"[Initialization] → Model requires HuggingFace access approval")
    
    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"[Initialization] ✅ Loaded {len(loaded_models)} models: {', '.join(loaded_models)}")
        if failed_models:
            print(f"[Initialization] ⚠️  Failed {len(failed_models)} models: {', '.join([m[0] for m in failed_models])}")
        print(f"{'='*60}\n")
    
    # Raise error only if NO models loaded
    if not loaded_models:
        raise RuntimeError(f"Failed to load any models. Check HuggingFace access and internet connection.")
    
    return registry


if __name__ == "__main__":
    # Test model registry
    print("Testing ModelRegistry...")
    
    print("\n1. Initializing registry...")
    registry = initialize_default_models(device="cuda", verbose=True)
    
    print("\n2. Registered models:")
    models = registry.list_models()
    print(f"Found {len(models)} models: {models}")
    
    print("\n3. Test image generation:")
    try:
        image = registry.generate_image(
            model_name="sdxl_base",
            prompt="A serene landscape with mountains and a lake, digital art",
            num_inference_steps=20,  # Fast test
            seed=12345
        )
        print(f"✅ Image generated: {image.size}")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
