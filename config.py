# config.py
# TwistedPic configuration settings for TwistedPair distortion,
# image generation models, pipeline parameters, and file management.

import os

# File locations
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# TwistedPair V2 API settings
TWISTEDPAIR_URL = "http://localhost:8001"
TWISTEDPAIR_TIMEOUT = 120  # seconds (distortion can be slow)

# Ollama settings
OLLAMA_URL = "http://localhost:11434"

# GPU Memory Management Settings
# Set to True to aggressively free GPU memory after each image generation
# This helps prevent TwistedPic from monopolizing GPU and blocking Ollama
GPU_MEMORY_CLEANUP = True

# Advanced: Move model to CPU after generation (slower but frees more GPU memory)
# Only enable if basic cleanup isn't enough and you need maximum GPU availability for Ollama
UNLOAD_MODEL_AFTER_GENERATION = True

# TwistedPair distortion settings
DISTORTION_MODES = [
    "invert_er",    # Challenge assumptions, expose contradictions
    "so_what_er",   # Ask "why does this matter?"
    "echo_er",      # Amplify core message
    "what_if_er",   # Explore hypotheticals
    "cucumb_er",    # Ground with evidence
    "archiv_er"     # Historical context
]

DISTORTION_TONES = [
    "neutral",      # Balanced, factual
    "technical",    # Precise terminology
    "primal",       # Visceral, direct
    "poetic",       # Metaphorical, evocative
    "satirical"     # Ironic, critical
]

# Default distortion settings
DEFAULT_DISTORTION = {
    "mode": "echo_er",
    "tone": "neutral",
    "gain": 5,
    "model": "mistral:latest"
}

# Prompt wrapper template for image generation
# {distorted} will be replaced with distorted text from TwistedPair
# Note: With Compel library, long prompts (>77 tokens) are automatically chunked
# If your distorted prompts are very long, consider simplifying this template
PROMPT_WRAPPER_TEMPLATE = "Digital art, concept art, highly detailed: {distorted}, trending on ArtStation, 4k"

# Available image models (architecture supports multiple)
IMAGE_MODELS = {
    "flux_dev": {
        "name": "Flux.1-dev",
        "model_id": "black-forest-labs/FLUX.1-dev",
        "pipeline_type": "flux",
        "default": False,
        "optimal_steps": 25,
        "optimal_guidance": 3.5,
        "max_sequence_length": 512,  # Flux supports very long prompts
        "notes": "⚠️ Very slow on RTX 5090 (compute 12.0 not optimized yet)"
    },
    "sd3_large": {
        "name": "Stable Diffusion 3.5 Large",
        "model_id": "stabilityai/stable-diffusion-3.5-large",
        "pipeline_type": "sd3",
        "default": True,
        "optimal_steps": 28,
        "optimal_guidance": 4.5,
        "max_sequence_length": 256
    },
    "sdxl_base": {
        "name": "SDXL Base 1.0",
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline_type": "sdxl",
        "default": False,
        "optimal_steps": 30,
        "optimal_guidance": 7.5,
        "max_sequence_length": 77
    }
}

# Default image generation parameters
# Note: Optimal steps/guidance vary by model (see IMAGE_MODELS for per-model values)
DEFAULT_IMAGE_PARAMS = {
    "num_inference_steps": 28,      # SD3: 28, SDXL: 30, Flux: 25
    "guidance_scale": 4.5,          # SD3: 4-5, SDXL: 7-8, Flux: 3-4
    "resolution": (1024, 768),      # Width x Height in pixels
    "seed": 42,                     # Fixed seed for reproducibility
    "use_random_seed": False,       # True = random, False = use fixed seed
    "negative_prompt": None         # No negative prompt (as per spec)
}

# Available resolutions
# Note: Flux.1-dev and SD 3.5 work best at 1024x1024 or higher
# SDXL works well at 768x512 or 1024x1024
RESOLUTIONS = {
    "landscapesm": (512, 384),
    "landscape": (1024, 768),
    "portrait": (768, 1024),
    "square": (1024, 1024),
    "wide": (1344, 768),
    "tall": (768, 1344)
}

# Image quality/style slider ranges
# Note: Different models have different optimal ranges
QUALITY_STEPS_RANGE = (10, 50)     # num_inference_steps
STYLE_CFG_RANGE = (1.0, 10.0)      # guidance_scale (Flux: 2-5, SD3: 3-7, SDXL: 7-10)

# ============================================================================
# Prompt Refiner Configuration
# ============================================================================

# LLM model for refining verbose TwistedPair output into visual keywords
REFINER_MODEL = "ministral-3:latest"

# Target number of visual keywords to extract (configurable range: 15-25)
REFINER_TARGET_KEYWORDS = 20

# Prompt template for refinement (expects {target_keywords} and {distorted_prompt})
REFINER_PROMPT_TEMPLATE = """You are a visual keyword extractor for AI image generation.

Your task: Extract exactly {target_keywords} essential VISUAL keywords from the verbose text below. Focus on:
- Concrete visual elements (objects, colors, textures, lighting)
- Artistic styles and techniques
- Composition and atmosphere
- NO abstract concepts, philosophical ideas, or verbose descriptions

Output format: comma-separated keywords only, no explanations.

Example output: "digital art, glowing neon, cyberpunk city, rain-soaked streets, purple and blue tones, dramatic lighting, high contrast, detailed textures, futuristic architecture, reflective surfaces, moody atmosphere, concept art, trending on ArtStation, 4k"

Verbose text to refine:
{distorted_prompt}

Visual keywords:"""

# Flask server settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
