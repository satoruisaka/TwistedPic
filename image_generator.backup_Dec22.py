"""
image_generator.py - Main Image Generation Orchestrator for TwistedPic

Orchestrates the full pipeline:
1. Distort user prompt via TwistedPair V2
2. Wrap distorted text in image generation template
3. Generate image via Stable Diffusion XL
4. Save image and metadata to outputs/
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image

import config
from twistedpair_client import TwistedPairClient, DistortionResult
from model_registry import ModelRegistry
from prompt_refiner import PromptRefiner


class ImageGenerator:
    """Main orchestrator for TwistedPic image generation."""
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        twistedpair_client: TwistedPairClient,
        verbose: bool = False
    ):
        """
        Initialize image generator.
        
        Args:
            model_registry: Initialized ModelRegistry with image models
            twistedpair_client: TwistedPairClient for distortion
            verbose: Enable debug logging
        """
        self.registry = model_registry
        self.twistedpair = twistedpair_client
        self.refiner = PromptRefiner()
        self.verbose = verbose
        
        # Ensure output directory exists
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    def generate(
        self,
        user_prompt: str,
        use_distortion: bool = True,
        use_refinement: bool = True,
        distortion_mode: str = "echo_er",
        distortion_tone: str = "neutral",
        distortion_gain: int = 5,
        distortion_model: Optional[str] = None,
        image_model: str = "sdxl_base",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        resolution: tuple[int, int] = (768, 512),
        seed: Optional[int] = None,
        use_random_seed: bool = False
    ) -> Dict[str, Any]:
        """
        Generate distorted image from user prompt.
        
        Full pipeline:
        1. Distort prompt with TwistedPair (if use_distortion=True)
        2. Refine distorted output to visual keywords (if use_refinement=True)
        3. Wrap refined keywords for image generation
        4. Generate image with SDXL
        5. Save image and metadata
        
        Args:
            user_prompt: Original user text prompt
            use_distortion: If True, use TwistedPair; if False, use prompt directly
            use_refinement: If True, refine distorted output to visual keywords
            distortion_mode: TwistedPair mode (invert_er, echo_er, etc.)
            distortion_tone: TwistedPair tone (neutral, poetic, etc.)
            distortion_gain: TwistedPair gain (1-10)
            distortion_model: Ollama model for distortion (optional)
            image_model: Image model name from registry
            num_inference_steps: SDXL quality (10-50)
            guidance_scale: SDXL prompt adherence (5-15)
            resolution: (width, height) tuple
            seed: Random seed for reproducibility
            use_random_seed: If True, use random seed
            
        Returns:
            Dict with:
                - image_path: Path to saved PNG
                - metadata_path: Path to saved JSON metadata
                - distorted_prompt: The distorted text used (or None)
                - refined_keywords: Visual keywords extracted (or None)
                - image_prompt: Full prompt sent to SDXL
                - metadata: Full generation metadata
                
        Raises:
            ConnectionError: If TwistedPair server unavailable
            ValueError: If invalid parameters
            RuntimeError: If generation fails
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"TwistedPic Generation Pipeline")
            print(f"{'='*60}")
        
        # Step 1: Distort prompt with TwistedPair (or bypass)
        if use_distortion:
            if self.verbose:
                print(f"\n[Step 1/4] Distorting prompt with TwistedPair...")
                print(f"  Original: {user_prompt}")
                print(f"  Mode: {distortion_mode}, Tone: {distortion_tone}, Gain: {distortion_gain}")
            
            try:
                distortion_result = self.twistedpair.distort(
                    text=user_prompt,
                    mode=distortion_mode,
                    tone=distortion_tone,
                    gain=distortion_gain,
                    model=distortion_model
                )
                distorted_text = distortion_result.output
                
                if self.verbose:
                    preview = distorted_text[:150] if len(distorted_text) > 150 else distorted_text
                    print(f"  Distorted: {preview}...")
            
            except ConnectionError as e:
                raise ConnectionError(
                    "TwistedPair server unavailable. "
                    "Start server with: cd TwistedPair/V2 && uvicorn server:app --port 8001"
                )
            except Exception as e:
                raise RuntimeError(f"Distortion failed: {e}")
        else:
            # Bypass distortion - use prompt as-is
            if self.verbose:
                print(f"\n[Step 1/5] Bypassing distortion (using prompt directly)...")
                print(f"  Original: {user_prompt}")
            distorted_text = user_prompt
            distortion_result = None
        
        # Step 2: Refine distorted output to visual keywords (if enabled)
        refined_keywords = None
        if use_distortion and use_refinement:
            if self.verbose:
                print(f"\n[Step 2/5] Refining distorted output to visual keywords...")
            
            try:
                # Ensure distorted_text is a string (defensive check)
                if not isinstance(distorted_text, str):
                    distorted_text = str(distorted_text)
                
                refinement_result = self.refiner.refine(distorted_text)
                refined_keywords = refinement_result['keywords']
                
                if self.verbose:
                    keyword_count = len([k.strip() for k in refined_keywords.split(',') if k.strip()])
                    print(f"  Extracted {keyword_count} keywords: {refined_keywords[:100]}...")
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Refinement failed ({e}), using verbose distorted text")
                refined_keywords = None
        
        # Step 3: Wrap for image generation
        if self.verbose:
            print(f"\n[Step 3/5] Wrapping prompt for image generation...")
        
        # Use refined keywords if available, otherwise fall back to distorted text
        prompt_for_image = refined_keywords if refined_keywords else distorted_text
        image_prompt = config.PROMPT_WRAPPER_TEMPLATE.format(distorted=prompt_for_image)
        
        if self.verbose:
            prompt_preview = image_prompt[:150] if len(image_prompt) > 150 else image_prompt
            print(f"  Image prompt: {prompt_preview}...")
        
        # Step 4: Generate image with SDXL
        if self.verbose:
            print(f"\n[Step 4/5] Generating image with {image_model}...")
            print(f"  Resolution: {resolution[0]}x{resolution[1]}")
            print(f"  Steps: {num_inference_steps}, CFG: {guidance_scale}")
        
        try:
            image = self.registry.generate_image(
                model_name=image_model,
                prompt=image_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                resolution=resolution,
                seed=seed,
                use_random_seed=use_random_seed,
                negative_prompt=None  # No negative prompt per spec
            )
        
        except Exception as e:
            raise RuntimeError(f"Image generation failed: {e}")
        
        # Step 5: Save image and metadata
        if self.verbose:
            print(f"\n[Step 5/5] Saving outputs...")
        
        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"twistedpic_{timestamp}"
        
        image_path = Path(config.OUTPUT_DIR) / f"{base_filename}.png"
        metadata_path = Path(config.OUTPUT_DIR) / f"{base_filename}_metadata.json"
        
        # Save image
        image.save(image_path, format="PNG")
        
        # Build metadata
        metadata = {
            "timestamp": timestamp,
            "iso_timestamp": datetime.now().isoformat(),
            "user_prompt": user_prompt,
            "distorted_prompt": distorted_text if use_distortion else None,
            "refined_keywords": refined_keywords if refined_keywords else None,
            "image_prompt": image_prompt,
            "distortion": {
                "enabled": use_distortion,
                "mode": distortion_mode if use_distortion else None,
                "tone": distortion_tone if use_distortion else None,
                "gain": distortion_gain if use_distortion else None,
                "model": distortion_result.model if (use_distortion and distortion_result) else None
            },
            "refinement": {
                "enabled": use_refinement,
                "model": config.REFINER_MODEL if use_refinement else None,
                "target_keywords": config.REFINER_TARGET_KEYWORDS if use_refinement else None
            },
            "image_generation": {
                "model": image_model,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "resolution": {
                    "width": resolution[0],
                    "height": resolution[1]
                },
                "seed": seed if not use_random_seed else "random",
                "negative_prompt": None
            },
            "output_files": {
                "image": str(image_path),
                "metadata": str(metadata_path)
            }
        }
        
        # Save metadata
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"  ✅ Image saved: {image_path}")
            print(f"  ✅ Metadata saved: {metadata_path}")
            print(f"\n{'='*60}")
            print(f"Generation Complete!")
            print(f"{'='*60}\n")
        
        return {
            "image_path": str(image_path),
            "metadata_path": str(metadata_path),
            "distorted_prompt": distorted_text,
            "refined_keywords": refined_keywords,
            "image_prompt": image_prompt,
            "metadata": metadata,
            "image": image  # Return PIL Image for display
        }


if __name__ == "__main__":
    # Test image generator
    print("Testing ImageGenerator...")
    
    # Initialize components
    print("\n1. Initializing components...")
    from model_registry import initialize_default_models
    
    registry = initialize_default_models(device="cuda", verbose=True)
    twistedpair = TwistedPairClient(verbose=True)
    generator = ImageGenerator(registry, twistedpair, verbose=True)
    
    # Check TwistedPair availability
    print("\n2. Checking TwistedPair...")
    if not twistedpair.is_healthy():
        print("❌ TwistedPair server offline. Start with:")
        print("   cd TwistedPair/V2 && uvicorn server:app --port 8001")
        exit(1)
    
    # Generate test image
    print("\n3. Generating test image...")
    try:
        result = generator.generate(
            user_prompt="A peaceful garden with colorful flowers",
            distortion_mode="echo_er",
            distortion_tone="poetic",
            distortion_gain=6,
            num_inference_steps=20,  # Fast test
            resolution=(768, 512),
            seed=42
        )
        
        print(f"\n✅ Test successful!")
        print(f"   Image: {result['image_path']}")
        print(f"   Distorted: {result['distorted_prompt'][:100]}...")
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
