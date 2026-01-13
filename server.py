"""
server.py - FastAPI ASGI Server for TwistedPic

Provides REST API endpoints for:
- Health checks (Ollama + TwistedPair status)
- Model list (available Ollama models)
- Image generation (full pipeline)
- Static file serving (HTML, CSS, JS)
"""

import os
import base64
from io import BytesIO
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import traceback

import config
from model_registry import initialize_default_models
from twistedpair_client import TwistedPairClient
from image_generator import ImageGenerator


# Initialize FastAPI app
app = FastAPI(title="TwistedPic", description="Distorted Image Generation API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global instances (initialized on startup)
model_registry = None
twistedpair_client = None
image_generator = None


# Pydantic models for request validation
class GenerateImageRequest(BaseModel):
    user_prompt: str
    use_distortion: bool = True
    distortion_mode: str = "echo_er"
    distortion_tone: str = "neutral"
    distortion_gain: int = 5
    distortion_model: Optional[str] = None
    use_refinement: bool = True
    image_model: str = "sd3_large"  # SD 3.5 Large - best quality + fast on RTX 5090
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    resolution_preset: str = "landscape"
    seed: Optional[int] = None
    use_random_seed: bool = False


def check_ollama_health() -> bool:
    """Check if Ollama server is running."""
    import requests
    try:
        response = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_ollama_models() -> list[str]:
    """Get list of available Ollama models."""
    import requests
    try:
        response = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return models
    except:
        pass
    return []


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global model_registry, twistedpair_client, image_generator
    
    print("Initializing TwistedPic components...")
    try:
        # Initialize empty model registry (lazy loading - no models loaded yet)
        # Models will be loaded on-demand when first generation request arrives
        # This prevents GPU memory occupation until actually needed
        from model_registry import ModelRegistry
        model_registry = ModelRegistry(device="cuda", verbose=True)
        print("[Startup] Model registry initialized (no models loaded yet - lazy loading enabled)")
        
        # Initialize TwistedPair client
        twistedpair_client = TwistedPairClient(verbose=False)
        
        # Initialize image generator
        image_generator = ImageGenerator(
            model_registry=model_registry,
            twistedpair_client=twistedpair_client,
            verbose=True
        )
        
        print("✅ All components initialized successfully")
        print("💡 GPU memory will remain free until first image generation request")
    
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        traceback.print_exc()


# ============================================================================
# Web Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve main web UI."""
    return templates.TemplateResponse("index.html", {"request": request})


# ============================================================================
# API Endpoints
# ============================================================================

@app.get('/api/health')
async def health_check():
    """
    Check health of all services.
    
    Returns:
        {
            "status": "healthy" | "degraded",
            "timestamp": ISO timestamp,
            "services": {
                "ollama": {"status": "online" | "offline", "models": [...]},
                "twistedpair": {"status": "online" | "offline"},
                "image_model": {"status": "loaded" | "not_loaded", "models": [...]}
            }
        }
    """
    # Check Ollama
    ollama_healthy = check_ollama_health()
    ollama_models = get_ollama_models() if ollama_healthy else []
    
    # Check TwistedPair
    twistedpair_healthy = False
    if twistedpair_client:
        twistedpair_healthy = twistedpair_client.is_healthy()
    
    # Check image models
    image_models_loaded = False
    available_image_models = []
    if model_registry:
        available_image_models = model_registry.list_models()
        image_models_loaded = len(available_image_models) > 0
    
    # Overall status
    overall_status = "healthy" if (ollama_healthy and twistedpair_healthy and image_models_loaded) else "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "services": {
            "ollama": {
                "status": "online" if ollama_healthy else "offline",
                "models": ollama_models
            },
            "twistedpair": {
                "status": "online" if twistedpair_healthy else "offline"
            },
            "image_model": {
                "status": "loaded" if image_models_loaded else "not_loaded",
                "models": available_image_models
            }
        }
    }


@app.get('/api/config')
async def get_config():
    """
    Get configuration for UI (modes, tones, resolutions, etc.).
    
    Returns:
        {
            "distortion": {
                "modes": [...],
                "tones": [...],
                "default_mode": "...",
                "default_tone": "...",
                "default_gain": 5
            },
            "image": {
                "resolutions": {...},
                "quality_range": [10, 50],
                "style_range": [5.0, 15.0],
                "default_resolution": "landscape",
                "default_steps": 30,
                "default_cfg": 7.5
            }
        }
    """
    return {
        "distortion": {
            "modes": config.DISTORTION_MODES,
            "tones": config.DISTORTION_TONES,
            "default_mode": config.DEFAULT_DISTORTION["mode"],
            "default_tone": config.DEFAULT_DISTORTION["tone"],
            "default_gain": config.DEFAULT_DISTORTION["gain"]
        },
        "image": {
            "resolutions": {
                "landscape": list(config.RESOLUTIONS["landscape"]),
                "portrait": list(config.RESOLUTIONS["portrait"]),
                "square": list(config.RESOLUTIONS["square"])
            },
            "quality_range": list(config.QUALITY_STEPS_RANGE),
            "style_range": list(config.STYLE_CFG_RANGE),
            "default_resolution": "landscape",
            "default_steps": config.DEFAULT_IMAGE_PARAMS["num_inference_steps"],
            "default_cfg": config.DEFAULT_IMAGE_PARAMS["guidance_scale"]
        }
    }


@app.post('/api/generate')
async def generate_image(data: GenerateImageRequest):
    """
    Generate distorted image from user prompt.
    
    Request body (validated by GenerateImageRequest model):
        {
            "user_prompt": "...",
            "use_distortion": true,
            "distortion_mode": "echo_er",
            "distortion_tone": "neutral",
            "distortion_gain": 5,
            "distortion_model": "mistral:latest" (optional),
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "resolution_preset": "landscape",
            "seed": 42 (optional),
            "use_random_seed": false
        }
    
    Returns:
        {
            "success": true,
            "image_base64": "...",
            "distorted_prompt": "...",
            "image_prompt": "...",
            "metadata": {...},
            "image_path": "...",
            "metadata_path": "..."
        }
    
    Errors:
        {
            "success": false,
            "error": "Error message"
        }
    """
    try:
        # Validate user prompt
        user_prompt = data.user_prompt.strip()
        if not user_prompt:
            raise HTTPException(status_code=400, detail="User prompt is required")
        
        # Get resolution from preset
        resolution = config.RESOLUTIONS.get(data.resolution_preset, config.RESOLUTIONS['landscape'])
        
        # Validate components initialized
        if not image_generator:
            raise HTTPException(
                status_code=500,
                detail="Image generator not initialized. Check server logs."
            )
        
        # Generate image
        result = image_generator.generate(
            user_prompt=user_prompt,
            use_distortion=data.use_distortion,
            use_refinement=data.use_refinement,
            distortion_mode=data.distortion_mode,
            distortion_tone=data.distortion_tone,
            distortion_gain=data.distortion_gain,
            distortion_model=data.distortion_model,
            image_model=data.image_model,
            num_inference_steps=data.num_inference_steps,
            guidance_scale=data.guidance_scale,
            resolution=resolution,
            seed=data.seed,
            use_random_seed=data.use_random_seed
        )
        
        # Convert image to base64 for web display
        image = result['image']
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Ensure string types (not objects)
        distorted_prompt = result.get('distorted_prompt', '')
        if distorted_prompt is None:
            distorted_prompt = ''
        
        refined_keywords = result.get('refined_keywords', '')
        if refined_keywords is None:
            refined_keywords = ''
        
        return {
            "success": True,
            "image_base64": image_base64,
            "distorted_prompt": str(distorted_prompt),
            "refined_keywords": str(refined_keywords),
            "image_prompt": str(result.get('image_prompt', '')),
            "metadata": result['metadata'],
            "image_path": result['image_path'],
            "metadata_path": result['metadata_path']
        }
    
    except HTTPException:
        raise
    
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"TwistedPair server unavailable: {str(e)}")
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameters: {str(e)}")
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    import uvicorn
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║                     TwistedPic Server                     ║
╠═══════════════════════════════════════════════════════════╣
║  Web UI:   http://localhost:{config.FLASK_PORT}                       ║
║  API Docs: http://localhost:{config.FLASK_PORT}/docs                  ║
║  Health:   http://localhost:{config.FLASK_PORT}/api/health            ║
╠═══════════════════════════════════════════════════════════╣
║  Prerequisites:                                           ║
║    • Ollama server running on {config.OLLAMA_URL}     ║
║    • TwistedPair V2 on {config.TWISTEDPAIR_URL}       ║
║    • CUDA GPU available (for SDXL)                       ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "server:app",
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        reload=True
    )
