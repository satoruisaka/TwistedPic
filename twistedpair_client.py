"""
twistedpair_client.py - TwistedPair V2 REST API Client for TwistedPic

Provides interface to TwistedPair V2 server at localhost:8001
for rhetorical distortion of text prompts.

API Endpoint: POST http://localhost:8001/distort-manual
"""

import requests
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import config


@dataclass
class DistortionResult:
    """Result from TwistedPair distortion."""
    output: str
    mode: str
    tone: str
    gain: int
    model: str
    timestamp: str
    
    def __str__(self) -> str:
        return self.output


class TwistedPairClient:
    """Client for TwistedPair V2 API."""
    
    def __init__(self, base_url: str = None, timeout: int = None, verbose: bool = False):
        """
        Initialize TwistedPair client.
        
        Args:
            base_url: TwistedPair server URL (default from config)
            timeout: Request timeout in seconds (default from config)
            verbose: Enable debug logging
        """
        self.base_url = base_url or config.TWISTEDPAIR_URL
        self.timeout = timeout or config.TWISTEDPAIR_TIMEOUT
        self.verbose = verbose
        self.distort_endpoint = f"{self.base_url}/distort-manual"
        self.health_endpoint = f"{self.base_url}/health"
        
    def is_healthy(self) -> bool:
        """
        Check if TwistedPair V2 server is running.
        
        Returns:
            True if server responds, False otherwise
        """
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            healthy = response.status_code == 200
            if self.verbose:
                status = "online" if healthy else "offline"
                print(f"[TwistedPair] Health check: {status}")
            return healthy
        except Exception as e:
            if self.verbose:
                print(f"[TwistedPair] Health check failed: {e}")
            return False
    
    def distort(
        self,
        text: str,
        mode: str,
        tone: str,
        gain: int,
        model: Optional[str] = None
    ) -> DistortionResult:
        """
        Distort text using TwistedPair V2 API.
        
        Args:
            text: User prompt to distort
            mode: Rhetorical mode (invert_er, so_what_er, etc.)
            tone: Verbal style (neutral, technical, etc.)
            gain: Intensity level (1-10)
            model: Ollama model name (optional, server default if None)
        
        Returns:
            DistortionResult with distorted text and metadata
            
        Raises:
            ConnectionError: If server is unreachable
            ValueError: If server returns error
            TimeoutError: If request times out
        """
        # Validate inputs
        if mode not in config.DISTORTION_MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {config.DISTORTION_MODES}")
        
        if tone not in config.DISTORTION_TONES:
            raise ValueError(f"Invalid tone: {tone}. Must be one of {config.DISTORTION_TONES}")
        
        if not (1 <= gain <= 10):
            raise ValueError(f"Invalid gain: {gain}. Must be between 1 and 10")
        
        # Build request payload
        payload = {
            "text": text,
            "mode": mode,
            "tone": tone,
            "gain": gain
        }
        
        if model:
            payload["model"] = model
        
        if self.verbose:
            print(f"[TwistedPair] Distorting with mode={mode}, tone={tone}, gain={gain}")
            print(f"[TwistedPair] Input: {text[:100]}...")
        
        try:
            # Send request
            start_time = time.time()
            response = requests.post(
                self.distort_endpoint,
                json=payload,
                timeout=self.timeout
            )
            elapsed = time.time() - start_time
            
            # Check response
            if response.status_code != 200:
                error_msg = f"TwistedPair API error: {response.status_code}"
                try:
                    error_detail = response.json().get("detail", response.text)
                    error_msg += f" - {error_detail}"
                except:
                    pass
                raise ValueError(error_msg)
            
            # Parse response
            data = response.json()
            
            # Extract output - handle both string and nested dict formats
            output = data.get("output", "")
            if isinstance(output, dict):
                # V2 API returns nested structure like {"output": {"text": "..."}}
                output = output.get("text", str(output))
            
            if not isinstance(output, str):
                output = str(output)
            
            result = DistortionResult(
                output=output,
                mode=data.get("mode", mode),
                tone=data.get("tone", tone),
                gain=data.get("gain", gain),
                model=data.get("model", model or "unknown"),
                timestamp=data.get("timestamp", "")
            )
            
            if self.verbose:
                print(f"[TwistedPair] Distortion completed in {elapsed:.1f}s")
                print(f"[TwistedPair] Output: {result.output[:100]}...")
            
            return result
            
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"TwistedPair request timed out after {self.timeout}s. "
                "Try increasing TWISTEDPAIR_TIMEOUT in config.py"
            )
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to TwistedPair at {self.base_url}. "
                "Ensure TwistedPair V2 server is running on localhost:8001"
            )
        except ValueError:
            raise  # Re-raise validation errors
        except Exception as e:
            raise RuntimeError(f"TwistedPair distortion failed: {e}")
    
    def get_available_models(self) -> list[str]:
        """
        Get list of available Ollama models from TwistedPair server.
        
        Returns:
            List of model names, empty list if server unavailable
        """
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
        except Exception as e:
            if self.verbose:
                print(f"[TwistedPair] Failed to get models: {e}")
        
        return []


# Convenience function for simple distortion
def distort_text(
    text: str,
    mode: str = "echo_er",
    tone: str = "neutral",
    gain: int = 5,
    model: Optional[str] = None
) -> str:
    """
    Simple function to distort text with TwistedPair.
    
    Returns distorted text string or raises exception on error.
    """
    client = TwistedPairClient()
    result = client.distort(text, mode, tone, gain, model)
    return result.output


if __name__ == "__main__":
    # Test client
    print("Testing TwistedPair V2 client...")
    
    client = TwistedPairClient(verbose=True)
    
    # Health check
    print("\n1. Health check:")
    if client.is_healthy():
        print("✅ TwistedPair V2 server is online")
    else:
        print("❌ TwistedPair V2 server is offline")
        exit(1)
    
    # Get models
    print("\n2. Available models:")
    models = client.get_available_models()
    print(f"Found {len(models)} models: {models[:3]}...")
    
    # Test distortion
    print("\n3. Test distortion:")
    try:
        result = client.distort(
            text="A peaceful garden with blooming flowers",
            mode="invert_er",
            tone="satirical",
            gain=7
        )
        print(f"✅ Distortion successful:")
        print(f"   Original: A peaceful garden with blooming flowers")
        print(f"   Distorted: {result.output}")
        print(f"   Model: {result.model}")
    except Exception as e:
        print(f"❌ Distortion failed: {e}")
