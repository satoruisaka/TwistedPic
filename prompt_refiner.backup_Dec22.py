"""
Prompt Refiner Module for TwistedPic

Distills verbose TwistedPair rhetorical output into essential visual keywords
for optimal Stable Diffusion XL image generation.

Author: TwistedPic Team
Date: December 2025
"""

import requests
import logging
from typing import Dict, Optional
from config import (
    OLLAMA_URL,
    REFINER_MODEL,
    REFINER_PROMPT_TEMPLATE,
    REFINER_TARGET_KEYWORDS
)

logger = logging.getLogger(__name__)


class PromptRefiner:
    """
    Refines verbose distorted prompts into concise visual keywords using Ollama LLM.
    
    Takes rhetorical output from TwistedPair and extracts 15-25 essential visual
    elements that work well with SDXL's image generation capabilities.
    """
    
    def __init__(self, ollama_url: str = OLLAMA_URL, model: str = REFINER_MODEL):
        """
        Initialize the PromptRefiner.
        
        Args:
            ollama_url: Base URL for Ollama API (default from config)
            model: LLM model name for refinement (default from config)
        """
        # Ensure we have the generate endpoint
        if not ollama_url.endswith('/api/generate'):
            self.ollama_url = ollama_url.rstrip('/') + '/api/generate'
        else:
            self.ollama_url = ollama_url
        self.model = model
        self.target_keywords = REFINER_TARGET_KEYWORDS
        
        logger.info(f"PromptRefiner initialized with model: {model}")
    
    def is_healthy(self) -> bool:
        """
        Check if Ollama service is available.
        
        Returns:
            bool: True if Ollama is responsive, False otherwise
        """
        try:
            base_url = self.ollama_url.replace('/api/generate', '')
            response = requests.get(base_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
    
    def refine(self, distorted_prompt: str, temperature: float = 0.3) -> Dict[str, str]:
        """
        Refine verbose distorted prompt into visual keywords.
        
        Args:
            distorted_prompt: Verbose rhetorical output from TwistedPair
            temperature: LLM sampling temperature (lower = more focused, default 0.3)
        
        Returns:
            Dict containing:
                - 'keywords': Comma-separated visual keywords (15-25 items)
                - 'original': Original distorted prompt (for transparency)
        
        Raises:
            requests.RequestException: If Ollama API call fails
            ValueError: If distorted_prompt is empty or invalid type
        """
        # Ensure we have a string (handle case where dict is passed)
        if isinstance(distorted_prompt, dict):
            # Extract 'text' or 'output' field if it's a nested dict
            distorted_prompt = distorted_prompt.get('text', distorted_prompt.get('output', str(distorted_prompt)))
        
        if not isinstance(distorted_prompt, str):
            distorted_prompt = str(distorted_prompt)
        
        if not distorted_prompt or not distorted_prompt.strip():
            raise ValueError("distorted_prompt cannot be empty")
        
        # Build refinement prompt using template from config
        refinement_prompt = REFINER_PROMPT_TEMPLATE.format(
            target_keywords=self.target_keywords,
            distorted_prompt=distorted_prompt
        )
        
        logger.info(f"Refining prompt with {self.model} (temp={temperature})")
        logger.debug(f"Original prompt length: {len(distorted_prompt)} chars")
        
        # Call Ollama API
        payload = {
            "model": self.model,
            "prompt": refinement_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 150  # Limit output length (15-25 keywords ~ 100-150 tokens)
            }
        }
        
        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Handle response - might be string or dict
            if isinstance(result, dict):
                keywords = result.get('response', '')
            else:
                keywords = str(result)
            
            # Ensure we have a string
            if not isinstance(keywords, str):
                keywords = str(keywords)
            
            keywords = keywords.strip()
            
            # Clean up output (remove potential markdown, extra whitespace)
            keywords = keywords.replace('**', '').replace('*', '').strip()
            
            # Validate we got something useful
            if not keywords:
                logger.error("Refiner returned empty response")
                raise ValueError("Refiner produced no keywords")
            
            keyword_count = len([k.strip() for k in keywords.split(',') if k.strip()])
            logger.info(f"Refined to {keyword_count} keywords ({len(keywords)} chars)")
            
            return {
                'keywords': keywords,
                'original': distorted_prompt
            }
            
        except requests.RequestException as e:
            logger.error(f"Ollama API call failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during refinement: {e}")
            raise


def test_refiner():
    """
    Quick test function for development.
    Run with: python prompt_refiner.py
    """
    print("Testing PromptRefiner...")
    
    refiner = PromptRefiner()
    
    # Check Ollama health
    if not refiner.is_healthy():
        print("❌ Ollama is not running. Start it with: ollama serve")
        return
    
    print(f"✅ Ollama is healthy")
    print(f"Using model: {refiner.model}")
    print(f"Target keywords: {refiner.target_keywords}")
    
    # Test with sample verbose distortion output
    test_prompt = """
    In this peculiar inversion of creative expression, one might ponder the absence 
    of imagery altogether—a digital canvas rendered blank by the very nature of textual 
    discourse. Consider, if you will, the metaphysical implications of generating visual 
    content from pure linguistic constructs, wherein the syntactic structures themselves 
    become the substrate for algorithmic interpretation. The recursive loop of meaning-making 
    between human intent and machine perception creates a fascinating dialectic that 
    challenges our assumptions about artistic creation in the age of artificial intelligence.
    """
    
    print("\n📝 Original verbose prompt:")
    print(f"   {test_prompt[:100]}...")
    print(f"   ({len(test_prompt)} characters)")
    
    try:
        result = refiner.refine(test_prompt)
        
        print("\n✨ Refined keywords:")
        print(f"   {result['keywords']}")
        
        keyword_count = len([k.strip() for k in result['keywords'].split(',') if k.strip()])
        print(f"\n📊 Keyword count: {keyword_count}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    test_refiner()
