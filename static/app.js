// app.js - TwistedPic Frontend Logic

// Global state
let currentImageData = null;
let healthCheckInterval = null;

// DOM Elements
const elements = {
    // Status indicators
    statusOllama: null,
    statusTwistedPair: null,
    statusImageModel: null,
    
    // Form controls
    userPrompt: null,
    enableDistortion: null,
    enableRefinement: null,
    distortionControls: null,
    distortionModel: null,
    distortionMode: null,
    distortionTone: null,
    distortionGain: null,
    gainValue: null,
    resolution: null,
    quality: null,
    qualityValue: null,
    style: null,
    styleValue: null,
    randomSeed: null,
    
    // Buttons
    btnGenerate: null,
    btnNew: null,
    btnSave: null,
    
    // Output
    progressContainer: null,
    progressFill: null,
    progressText: null,
    imageContainer: null,
    distortedPromptContainer: null,
    distortedPromptText: null,
    refinedKeywordsContainer: null,
    refinedKeywordsText: null,
    imagePromptContainer: null,
    imagePromptText: null,
    errorContainer: null,
    errorText: null
};

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    setupEventListeners();
    startHealthCheck();
    loadConfig();
});

// Initialize DOM element references
function initializeElements() {
    elements.statusOllama = document.querySelector('#status-ollama .status-dot');
    elements.statusTwistedPair = document.querySelector('#status-twistedpair .status-dot');
    elements.statusImageModel = document.querySelector('#status-imagemodel .status-dot');
    
    elements.userPrompt = document.getElementById('user-prompt');
    elements.enableDistortion = document.getElementById('enable-distortion');
    elements.enableRefinement = document.getElementById('enable-refinement');
    elements.distortionControls = document.getElementById('distortion-controls');
    elements.distortionModel = document.getElementById('distortion-model');
    elements.distortionMode = document.getElementById('distortion-mode');
    elements.distortionTone = document.getElementById('distortion-tone');
    elements.distortionGain = document.getElementById('distortion-gain');
    elements.gainValue = document.getElementById('gain-value');
    elements.resolution = document.getElementById('resolution');
    elements.quality = document.getElementById('quality');
    elements.qualityValue = document.getElementById('quality-value');
    elements.style = document.getElementById('style');
    elements.styleValue = document.getElementById('style-value');
    elements.randomSeed = document.getElementById('random-seed');
    
    elements.btnGenerate = document.getElementById('btn-generate');
    elements.btnNew = document.getElementById('btn-new');
    elements.btnSave = document.getElementById('btn-save');
    
    elements.progressContainer = document.getElementById('progress-container');
    elements.progressFill = document.getElementById('progress-fill');
    elements.progressText = document.getElementById('progress-text');
    elements.imageContainer = document.getElementById('image-container');
    elements.distortedPromptContainer = document.getElementById('distorted-prompt-container');
    elements.distortedPromptText = document.getElementById('distorted-prompt-text');
    elements.refinedKeywordsContainer = document.getElementById('refined-keywords-container');
    elements.refinedKeywordsText = document.getElementById('refined-keywords-text');
    elements.imagePromptContainer = document.getElementById('image-prompt-container');
    elements.imagePromptText = document.getElementById('image-prompt-text');
    elements.errorContainer = document.getElementById('error-container');
    elements.errorText = document.getElementById('error-text');
}

// Setup event listeners
function setupEventListeners() {
    // Distortion toggle
    elements.enableDistortion.addEventListener('change', (e) => {
        if (e.target.checked) {
            elements.distortionControls.style.display = 'block';
            elements.distortionControls.classList.remove('disabled');
        } else {
            elements.distortionControls.style.display = 'block';
            elements.distortionControls.classList.add('disabled');
        }
    });
    
    // Slider value updates
    elements.distortionGain.addEventListener('input', (e) => {
        elements.gainValue.textContent = e.target.value;
    });
    
    elements.quality.addEventListener('input', (e) => {
        elements.qualityValue.textContent = e.target.value;
    });
    
    elements.style.addEventListener('input', (e) => {
        elements.styleValue.textContent = e.target.value;
    });
    
    // Button clicks
    elements.btnGenerate.addEventListener('click', handleGenerate);
    elements.btnNew.addEventListener('click', handleNewImage);
    elements.btnSave.addEventListener('click', handleSave);
}

// Health check
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        // Update status indicators
        updateStatusIndicator(elements.statusOllama, data.services.ollama.status === 'online');
        updateStatusIndicator(elements.statusTwistedPair, data.services.twistedpair.status === 'online');
        updateStatusIndicator(elements.statusImageModel, data.services.image_model.status === 'loaded');
        
        // Update model dropdown (preserve current selection)
        if (data.services.ollama.status === 'online' && data.services.ollama.models.length > 0) {
            const currentValue = elements.distortionModel.value;
            elements.distortionModel.innerHTML = '';
            
            data.services.ollama.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                elements.distortionModel.appendChild(option);
            });
            
            // Restore selection if still available
            if (currentValue && Array.from(elements.distortionModel.options).some(opt => opt.value === currentValue)) {
                elements.distortionModel.value = currentValue;
            }
        }
        
        // Enable/disable Generate button
        const allHealthy = data.services.ollama.status === 'online' &&
                          data.services.twistedpair.status === 'online' &&
                          data.services.image_model.status === 'loaded';
        
        elements.btnGenerate.disabled = !allHealthy;
        
    } catch (error) {
        console.error('Health check failed:', error);
        updateStatusIndicator(elements.statusOllama, false);
        updateStatusIndicator(elements.statusTwistedPair, false);
        updateStatusIndicator(elements.statusImageModel, false);
        elements.btnGenerate.disabled = true;
    }
}

function updateStatusIndicator(element, isOnline) {
    if (isOnline) {
        element.classList.remove('offline');
        element.classList.add('online');
    } else {
        element.classList.remove('online');
        element.classList.add('offline');
    }
}

function startHealthCheck() {
    checkHealth(); // Initial check
    healthCheckInterval = setInterval(checkHealth, 10000); // Every 10 seconds
}

// Load config from server
async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        const config = await response.json();
        
        // Set default values from config
        elements.distortionMode.value = config.distortion.default_mode;
        elements.distortionTone.value = config.distortion.default_tone;
        elements.distortionGain.value = config.distortion.default_gain;
        elements.gainValue.textContent = config.distortion.default_gain;
        
        elements.quality.value = config.image.default_steps;
        elements.qualityValue.textContent = config.image.default_steps;
        
        elements.style.value = config.image.default_cfg;
        elements.styleValue.textContent = config.image.default_cfg;
        
    } catch (error) {
        console.error('Failed to load config:', error);
    }
}

// Handle Generate button
async function handleGenerate() {
    const userPrompt = elements.userPrompt.value.trim();
    
    if (!userPrompt) {
        showError('Please enter a prompt');
        return;
    }
    
    // Clear previous state
    hideError();
    hideImage();
    
    const useDistortion = elements.enableDistortion.checked;
    const useRefinement = elements.enableRefinement.checked;
    showProgress(useDistortion ? 'Distorting prompt with TwistedPair...' : 'Preparing image generation...');
    
    // Build request payload
    const payload = {
        user_prompt: userPrompt,
        use_distortion: useDistortion,
        use_refinement: useRefinement,
        distortion_mode: elements.distortionMode.value,
        distortion_tone: elements.distortionTone.value,
        distortion_gain: parseInt(elements.distortionGain.value),
        distortion_model: elements.distortionModel.value,
        num_inference_steps: parseInt(elements.quality.value),
        guidance_scale: parseFloat(elements.style.value),
        resolution_preset: elements.resolution.value,
        use_random_seed: elements.randomSeed.checked
    };
    
    if (!elements.randomSeed.checked) {
        payload.seed = 42; // Fixed seed
    }
    
    try {
        // Disable generate button during generation
        elements.btnGenerate.disabled = true;
        
        // Update progress
        if (payload.use_distortion) {
            updateProgress(20, 'Distorting prompt...');
        } else {
            updateProgress(20, 'Preparing prompt...');
        }
        
        // Send request
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        // Update progress
        updateProgress(40, 'Generating image with Stable Diffusion...');
        
        const data = await response.json();
        
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Generation failed');
        }
        
        // Update progress
        updateProgress(90, 'Finalizing...');
        
        // Store result
        currentImageData = data;
        
        // Display image
        displayImage(data);
        
        // Complete
        updateProgress(100, 'Complete!');
        setTimeout(() => {
            hideProgress();
        }, 1000);
        
    } catch (error) {
        console.error('Generation error:', error);
        showError(error.message || 'Generation failed');
        hideProgress();
    } finally {
        // Re-check health to re-enable button if services still available
        checkHealth();
    }
}

// Handle New Image button
function handleNewImage() {
    // Clear image display but keep settings
    hideImage();
    hideError();
    elements.btnSave.style.display = 'none';
    currentImageData = null;
    
    // Show placeholder
    elements.imageContainer.innerHTML = `
        <div class="placeholder">
            <p>👆 Configure settings and press "Generate Image"</p>
        </div>
    `;
}

// Handle Save button
function handleSave() {
    if (!currentImageData) return;
    
    // Convert base64 to blob
    const base64Data = currentImageData.image_base64;
    const byteCharacters = atob(base64Data);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'image/png' });
    
    // Create object URL
    const url = URL.createObjectURL(blob);
    
    // Generate filename from timestamp
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    const filename = `twistedpic_${timestamp}.png`;
    
    // Try to use File System Access API (Chrome 86+, Edge 86+)
    if ('showSaveFilePicker' in window) {
        // Modern browsers - shows native save dialog
        window.showSaveFilePicker({
            suggestedName: filename,
            types: [{
                description: 'PNG Image',
                accept: { 'image/png': ['.png'] }
            }]
        }).then(handle => {
            return handle.createWritable();
        }).then(writable => {
            return writable.write(blob).then(() => writable.close());
        }).then(() => {
            console.log('File saved successfully');
        }).catch(err => {
            // User cancelled or error - fallback to download
            if (err.name !== 'AbortError') {
                console.log('Save dialog error, using download fallback');
                fallbackDownload(url, filename);
            }
        }).finally(() => {
            URL.revokeObjectURL(url);
        });
    } else {
        // Fallback for older browsers or Firefox - direct download
        fallbackDownload(url, filename);
        setTimeout(() => URL.revokeObjectURL(url), 100);
    }
}

// Fallback download method
function fallbackDownload(url, filename) {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Display generated image
function displayImage(data) {
    // Display image
    elements.imageContainer.innerHTML = `
        <img src="data:image/png;base64,${data.image_base64}" alt="Generated Image">
    `;
    
    // Display distorted prompt (only if distortion was used and it's different from original)
    const distortedPrompt = data.distorted_prompt;
    const refinedKeywords = data.refined_keywords;
    const userPrompt = data.metadata ? data.metadata.user_prompt : '';
    
    if (distortedPrompt && typeof distortedPrompt === 'string' && distortedPrompt !== userPrompt) {
        elements.distortedPromptText.textContent = distortedPrompt;
        elements.distortedPromptContainer.style.display = 'block';
    } else {
        elements.distortedPromptContainer.style.display = 'none';
    }
    
    // Display refined keywords (if refinement was used)
    if (refinedKeywords && typeof refinedKeywords === 'string' && refinedKeywords !== '') {
        elements.refinedKeywordsText.textContent = refinedKeywords;
        elements.refinedKeywordsContainer.style.display = 'block';
    } else {
        elements.refinedKeywordsContainer.style.display = 'none';
    }
    
    // Display image prompt
    elements.imagePromptText.textContent = data.image_prompt;
    elements.imagePromptContainer.style.display = 'block';
    
    // Show save button
    elements.btnSave.style.display = 'block';
}

// Hide image and prompts
function hideImage() {
    elements.imageContainer.innerHTML = `
        <div class="placeholder">
            <p>👆 Configure settings and press "Generate Image"</p>
        </div>
    `;
    elements.distortedPromptContainer.style.display = 'none';
    elements.refinedKeywordsContainer.style.display = 'none';
    elements.imagePromptContainer.style.display = 'none';
}

// Progress bar functions
function showProgress(message) {
    elements.progressContainer.style.display = 'block';
    elements.progressFill.style.width = '0%';
    elements.progressText.textContent = message;
}

function updateProgress(percent, message) {
    elements.progressFill.style.width = `${percent}%`;
    if (message) {
        elements.progressText.textContent = message;
    }
}

function hideProgress() {
    elements.progressContainer.style.display = 'none';
}

// Error display functions
function showError(message) {
    elements.errorText.textContent = `⚠️ ${message}`;
    elements.errorContainer.style.display = 'block';
}

function hideError() {
    elements.errorContainer.style.display = 'none';
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (healthCheckInterval) {
        clearInterval(healthCheckInterval);
    }
});
