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

# Available image models (architecture supports multiple, currently using one)
IMAGE_MODELS = {
    "sdxl_base": {
        "name": "SDXL Base 1.0",
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "default": True
    }
    # Future models can be added here:
    # "sdxl_refiner": {
    #     "name": "SDXL Refiner 1.0",
    #     "model_id": "stabilityai/stable-diffusion-xl-refiner-1.0",
    #     "default": False
    # }
}

# Default image generation parameters
DEFAULT_IMAGE_PARAMS = {
    "num_inference_steps": 30,      # Quality: 10 (fast) - 50 (detailed)
    "guidance_scale": 7.5,          # Style: 5 (creative) - 15 (literal)
    "resolution": (768, 512),       # Width x Height in pixels
    "seed": 42,                     # Fixed seed for reproducibility
    "use_random_seed": False,       # True = random, False = use fixed seed
    "negative_prompt": None         # No negative prompt (as per spec)
}

# Available resolutions
RESOLUTIONS = {
    "landscape": (768, 512),
    "portrait": (512, 768),
    "square": (1024, 1024)
}

# Image quality/style slider ranges
QUALITY_STEPS_RANGE = (10, 50)     # num_inference_steps
STYLE_CFG_RANGE = (5.0, 15.0)      # guidance_scale

# ============================================================================
# Prompt Refiner Configuration
# ============================================================================

# LLM model for refining verbose TwistedPair output into visual keywords
REFINER_MODEL = "mistral:instruct"

# Target number of visual keywords to extract (configurable range: 15-25)
REFINER_TARGET_KEYWORDS = 20

# Prompt template for refinement (expects {target_keywords} and {distorted_prompt})
REFINER_PROMPT_TEMPLATE = """You are a visual keyword extractor for Stable Diffusion XL image generation.

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
