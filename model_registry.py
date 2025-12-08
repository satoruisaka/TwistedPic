"""
model_registry.py - Image Model Registry for TwistedPic

Manages Stable Diffusion XL models for image generation.
Adapted from DreamSprout with configurable parameters.
"""

import torch
from diffusers import StableDiffusionXLPipeline
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
        parameters: Optional[Dict[str, Any]] = None,
        dtype: torch.dtype = None
    ) -> None:
        """
        Register a Stable Diffusion XL model.
        
        Args:
            name: Registry name for the model
            model_id: HuggingFace model ID
            parameters: Generation parameters (steps, CFG, resolution, etc.)
            dtype: torch dtype (default: float16 for CUDA, float32 for CPU)
        """
        if self.verbose:
            print(f"[ModelRegistry] Loading {model_id}...")
        
        # Set dtype based on device
        if dtype is None:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Load pipeline
        try:
            if self.device == "cuda":
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    use_safetensors=True,
                    variant="fp16"
                ).to(self.device)
                
                # Enable memory efficient attention
                pipe.enable_xformers_memory_efficient_attention()
            else:
                # CPU mode - simpler loading
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    use_safetensors=True
                ).to(self.device)
            
            # Store generation parameters
            default_params = config.DEFAULT_IMAGE_PARAMS.copy()
            if parameters:
                default_params.update(parameters)
            
            pipe.generation_config = default_params
            
            # Initialize Compel for long prompt support (>77 tokens)
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
        
        Args:
            name: Registry name of the model
            
        Returns:
            Diffusers pipeline or None if not found
        """
        return self.image_models.get(name)
    
    def list_models(self) -> list[str]:
        """Get list of registered model names."""
        return list(self.image_models.keys())
    
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
        
        # Check if prompt exceeds 77 tokens and use Compel if available
        use_compel = hasattr(pipe, 'compel') and pipe.compel is not None
        
        # Generate image
        try:
            if use_compel:
                # Use Compel for long prompt support
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
            else:
                # Standard generation (77 token limit)
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
            
            return image
        
        except Exception as e:
            print(f"[ModelRegistry] ❌ Image generation failed: {e}")
            raise


def initialize_default_models(device: str = "cuda", verbose: bool = False) -> ModelRegistry:
    """
    Initialize ModelRegistry with default SDXL model.
    
    Args:
        device: Device to run models on ("cuda" or "cpu")
        verbose: Enable debug logging
        
    Returns:
        Initialized ModelRegistry
    """
    registry = ModelRegistry(device=device, verbose=verbose)
    
    # Register default SDXL model
    for model_key, model_info in config.IMAGE_MODELS.items():
        if model_info.get("default", False):
            registry.register_image_model(
                name=model_key,
                model_id=model_info["model_id"],
                parameters=config.DEFAULT_IMAGE_PARAMS
            )
            break
    
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
