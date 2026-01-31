"""
server.py - Flask Web Server for TwistedPic

Provides REST API endpoints for:
- Health checks (Ollama + TwistedPair status)
- Model list (available Ollama models)
- Image generation (full pipeline)
- Static file serving (HTML, CSS, JS)
"""

import os
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from datetime import datetime
import traceback

import config
from model_registry import initialize_default_models
from twistedpair_client import TwistedPairClient
from image_generator import ImageGenerator


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# Global instances (initialized on startup)
model_registry = None
twistedpair_client = None
image_generator = None


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


@app.before_request
def initialize_components():
    """Initialize components on first request (lazy loading)."""
    global model_registry, twistedpair_client, image_generator
    
    if model_registry is None:
        print("Initializing TwistedPic components...")
        try:
            # Initialize model registry (this loads SDXL, may take time)
            model_registry = initialize_default_models(device="cuda", verbose=True)
            
            # Initialize TwistedPair client
            twistedpair_client = TwistedPairClient(verbose=False)
            
            # Initialize image generator
            image_generator = ImageGenerator(
                model_registry=model_registry,
                twistedpair_client=twistedpair_client,
                verbose=True
            )
            
            print("✅ All components initialized successfully")
        
        except Exception as e:
            print(f"❌ Component initialization failed: {e}")
            traceback.print_exc()


# ============================================================================
# Web Routes
# ============================================================================

@app.route('/')
def index():
    """Serve main web UI."""
    return render_template('index.html')


# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
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
    
    return jsonify({
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
    })


@app.route('/api/config', methods=['GET'])
def get_config():
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
    return jsonify({
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
    })


@app.route('/api/generate', methods=['POST'])
def generate_image():
    """
    Generate distorted image from user prompt.
    
    Request body:
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
        # Parse request
        data = request.get_json()
        
        # Extract parameters with defaults
        user_prompt = data.get('user_prompt', '').strip()
        if not user_prompt:
            return jsonify({"success": False, "error": "User prompt is required"}), 400
        
        use_distortion = data.get('use_distortion', True)  # Default to True
        distortion_mode = data.get('distortion_mode', config.DEFAULT_DISTORTION['mode'])
        distortion_tone = data.get('distortion_tone', config.DEFAULT_DISTORTION['tone'])
        distortion_gain = int(data.get('distortion_gain', config.DEFAULT_DISTORTION['gain']))
        distortion_model = data.get('distortion_model')  # Optional
        
        num_inference_steps = int(data.get('num_inference_steps', 30))
        guidance_scale = float(data.get('guidance_scale', 7.5))
        resolution_preset = data.get('resolution_preset', 'landscape')
        seed = data.get('seed')
        if seed is not None:
            seed = int(seed)
        use_random_seed = data.get('use_random_seed', False)
        
        # Get resolution from preset
        resolution = config.RESOLUTIONS.get(resolution_preset, config.RESOLUTIONS['landscape'])
        
        # Validate components initialized
        if not image_generator:
            return jsonify({
                "success": False,
                "error": "Image generator not initialized. Check server logs."
            }), 500
        
        # Generate image
        result = image_generator.generate(
            user_prompt=user_prompt,
            use_distortion=use_distortion,
            use_refinement=data.get('use_refinement', True),  # Default to True
            distortion_mode=distortion_mode,
            distortion_tone=distortion_tone,
            distortion_gain=distortion_gain,
            distortion_model=distortion_model,
            image_model="sdxl_base",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            resolution=resolution,
            seed=seed,
            use_random_seed=use_random_seed
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
        
        return jsonify({
            "success": True,
            "image_base64": image_base64,
            "distorted_prompt": str(distorted_prompt),
            "refined_keywords": str(refined_keywords),
            "image_prompt": str(result.get('image_prompt', '')),
            "metadata": result['metadata'],
            "image_path": result['image_path'],
            "metadata_path": result['metadata_path']
        })
    
    except ConnectionError as e:
        return jsonify({
            "success": False,
            "error": f"TwistedPair server unavailable: {str(e)}"
        }), 503
    
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": f"Invalid parameters: {str(e)}"
        }), 400
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"Generation failed: {str(e)}"
        }), 500


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║                     TwistedPic Server                     ║
╠═══════════════════════════════════════════════════════════╣
║  Web UI:   http://localhost:{config.FLASK_PORT}                       ║
║  API Docs: http://localhost:{config.FLASK_PORT}/api/health            ║
╠═══════════════════════════════════════════════════════════╣
║  Prerequisites:                                           ║
║    • Ollama server running on {config.OLLAMA_URL}     ║
║    • TwistedPair V2 on {config.TWISTEDPAIR_URL}       ║
║    • CUDA GPU available (for SDXL)                       ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG
    )
