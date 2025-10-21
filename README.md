# Raiki SDK 🔍

**Raiki SDK** - Base interfaces and utilities for authentication and detection systems.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **Abstract Base Classes**: Ready-to-extend interfaces for authentication and YOLO-based detection
- **Model Package Registry**: Dynamic model loading with lazy initialization and caching
  - Part-based versioning (P00, P01, P02, etc.)
  - Model version management (v1_1, v1_2, v1_3, etc.)
  - Automatic model discovery and loading
- **Pydantic Schemas**: Type-safe data validation for detection results
- **Image Preprocessing**: Professional image processing utilities
  - Resize and padding operations
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - LoG (Laplacian of Gaussian) filtering
  - Brightness/Contrast adjustments
  - White padding removal

## 📦 Installation

> **Note:** This is a public repository with some restricted permissions.

### Basic Installation (Public Access)

```bash
# Latest version from main branch
pip install git+https://gitlab.com/dat.tram/raiki-sdk.git

# Or with uv (recommended)
uv pip install git+https://gitlab.com/dat.tram/raiki-sdk.git
```

### Install Specific Version

```bash
# Install version 0.1.0
pip install git+https://gitlab.com/dat.tram/raiki-sdk.git@v0.1.0

# Or with uv
uv pip install git+https://gitlab.com/dat.tram/raiki-sdk.git@v0.1.0
```

### Add to requirements.txt

```txt
# Latest version
raiki-sdk @ git+https://gitlab.com/dat.tram/raiki-sdk.git

# Specific version
raiki-sdk @ git+https://gitlab.com/dat.tram/raiki-sdk.git@v0.1.0
```

### Add to pyproject.toml

```toml
[project]
dependencies = [
    "raiki-sdk @ git+https://gitlab.com/dat.tram/raiki-sdk.git@v0.1.0",
]
```

### Alternative: Using SSH

If you have SSH key configured with GitLab:

```bash
# Latest version
pip install git+ssh://git@gitlab.com/dat.tram/raiki-sdk.git

# Specific version
pip install git+ssh://git@gitlab.com/dat.tram/raiki-sdk.git@v0.1.0
```

### Authentication (For Restricted Operations)

If you need access to restricted features or operations, create a GitLab Personal Access Token:

1. Go to GitLab: **Settings → Access Tokens** → [Create new token](https://gitlab.com/-/profile/personal_access_tokens)
2. Set token name: `raiki-sdk-access`
3. Select scopes: `read_repository` (and others as needed)
4. Click **Create personal access token**
5. **Copy the token** (shown only once!)

Then install with authentication:

```bash
# With token
pip install git+https://oauth2:YOUR_TOKEN@gitlab.com/dat.tram/raiki-sdk.git@v0.1.0

# Using environment variable (recommended)
export GITLAB_TOKEN=your_token_here
pip install git+https://oauth2:${GITLAB_TOKEN}@gitlab.com/dat.tram/raiki-sdk.git@v0.1.0
```

⚠️ **Security Warning:** Do NOT commit tokens to git! Use environment variables or secret managers.

## 🎯 Quick Start

### Using Base Classes

```python
from raiki_sdk import AuthenticateBase, YoloBase
from raiki_sdk import AuthResult, YoloResult

class MyAuthenticator(AuthenticateBase):
    """Custom authentication implementation"""
    
    def authenticate(self, image_path: str) -> AuthResult:
        # Your authentication logic here
        return AuthResult(
            is_authentic=True,
            confidence=0.95,
            message="Product is authentic"
        )

class MyDetector(YoloBase):
    """Custom YOLO detection implementation"""
    
    def __init__(self):
        super().__init__(model_path="path/to/your/model.pt")
    
    def detect(self, image_path: str) -> YoloResult:
        # Your detection logic here
        pass
```

### Using Preprocessing Utilities

```python
from raiki_sdk import (
    resize_pwd,
    padding_crop_yolo,
    CLAHE,
    LoGFilter,
    AdjustBrightnessContrast
)
from PIL import Image

# Load image
image = Image.open("product.jpg")

# Resize with aspect ratio preservation
resized = resize_pwd(image, target_size=(640, 640))

# Apply CLAHE enhancement
clahe = CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
enhanced = clahe(resized)

# Apply LoG filter for edge detection
log_filter = LoGFilter(sigma=1.5, kernel_size=5)
edges = log_filter(enhanced)

# Adjust brightness and contrast
adjuster = AdjustBrightnessContrast(brightness_factor=1.2, contrast_factor=1.1)
final_image = adjuster(edges)
```

### Using Model Packages

The SDK provides a registry system for managing different model versions:

```python
from raiki_sdk import get_model, list_available_models, list_models

# List all available model packages (scans filesystem)
available_models = list_available_models()
for model in available_models:
    print(f"{model['category']} - {model['part']} - {model['version']}")

# Load a specific model package (lazy loading)
# Automatically detects if it's an auth or classifier model
model = get_model(category="rolex", part="P01", model_version="v1_3")

# For authentication models (P01, P02, etc.)
detector = model['detector']
authenticator = model['authenticator']

# Use the detector
result = detector.detect("image.jpg")
print(f"Detected: {result.class_names}")

# Use the authenticator
auth_result = authenticator.authenticate("image.jpg")
print(f"Authentic: {auth_result.is_authentic}, Confidence: {auth_result.confidence}")

# For classifier models (P00 only)
classifier_model = get_model(category="rolex", part="P00", model_version="v1_1")
classifier = classifier_model['classifier']

# List currently cached models (already loaded in memory)
cached_models = list_models()
print(f"Cached models: {cached_models}")
```

#### Part Versioning

The SDK supports different product parts with independent versioning:

- **P00**: Classifier models (category classification)
- **P01, P02, P03, ...**: Authentication models (counterfeit detection for specific parts)

#### Model Versioning

Version strings are automatically normalized:
- `"v1.3"` → `"v1_3"`
- `"v1_2"` → `"v1_2"`

This allows for flexible version naming while maintaining consistency.

## 📚 API Reference

### Base Classes

#### `AuthenticateBase`
Abstract base class for authentication systems.

**Methods:**
- `authenticate(image_path: str) -> AuthResult`: Implement your authentication logic

#### `YoloBase`
Abstract base class for YOLO-based detection systems.

**Methods:**
- `detect(image_path: str) -> YoloResult`: Detect objects in image
- `validate(image_path: str) -> YoloValidation`: Validate detection results

### Schemas

#### `AuthResult`
- `is_authentic: bool` - Authentication result
- `confidence: float` - Confidence score (0-1)
- `message: str` - Result message
- `metadata: dict` - Additional metadata

#### `YoloResult`
- `boxes: List` - Detected bounding boxes
- `confidences: List[float]` - Detection confidences
- `class_ids: List[int]` - Class IDs
- `class_names: List[str]` - Class names

### Preprocessing Functions

#### Image Processors
- `resize_pwd(image, target_size)` - Resize with aspect ratio
- `padding_crop_yolo(image, target_size)` - Pad and crop for YOLO
- `remove_white_padding(image)` - Remove white borders
- `pil_to_tensor(image)` - Convert PIL to tensor

#### Image Filters
- `CLAHE()` - Contrast enhancement
- `LoGFilter()` - Edge detection
- `AdjustBrightnessContrast()` - Brightness/contrast adjustment
- `UnsharpMask()` - Image sharpening

### Model Management

#### `get_model(category, part, model_version)`
Unified function to load any model package (auto-detects auth vs classifier).

**Parameters:**
- `category: str` - Brand category ("rolex", "lv", etc.)
- `part: str` - Part identifier ("P00" for classifier, "P01", "P02", etc. for auth)
- `model_version: str` - Version identifier ("v1_1", "v1_2", etc.)

**Returns:** `dict` with model instances and metadata

#### `get_auth_model(category, part, model_version)`
Load authentication model package (P01, P02, etc.).

**Returns:** `dict` with `detector` and `authenticator` instances

#### `list_models()`
List all currently cached models (loaded in memory).

**Returns:** `dict` with model metadata and load status

#### `list_available_models()`
Scan filesystem and list all available model packages without loading them.

**Returns:** `list` of available models with metadata

#### `registry`
Direct access to the global `ModelRegistry` instance for advanced usage.

### Model Package Structure

Model packages follow this structure:

```
raiki_sdk/model_packages/
├── {category}/              # e.g., "rolex", "lv"
│   └── {category}_{part}_{version}/   # e.g., "rolex_p01_v1_3"
│       ├── __init__.py      # Factory functions
│       ├── detector.py      # Detector implementation
│       ├── authenticator.py # Authenticator implementation (for auth models)
│       ├── classifier.py    # Classifier implementation (for P00)
│       ├── config.py        # Model configuration
│       └── preprocessor.py  # Preprocessing logic
```

Each package must provide factory functions in `__init__.py`:
- `get_detector()` - Returns detector instance
- `get_authenticator()` - Returns authenticator instance (auth models)
- `get_classifier()` - Returns classifier instance (P00 models)

## 🔧 Development

### Setup Development Environment

#### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://gitlab.com/dat.tram/raiki-sdk.git
cd raiki-sdk

# Sync dependencies (creates venv automatically)
uv sync

# The package is now installed in editable mode
# Any code changes will be reflected immediately
```

#### Option 2: Using traditional pip

```bash
# Clone the repository
git clone https://gitlab.com/dat.tram/raiki-sdk.git
cd raiki-sdk

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Verify Installation

```bash
# Test imports
python -c "from raiki_sdk import AuthenticateBase, YoloBase; print('✅ Success!')"

# Check version
python -c "import raiki_sdk; print(f'Version: {raiki_sdk.__version__}')"

# Run validation script
python test_import.py
```

## 📋 Requirements

- Python >= 3.10
- huggingface-hub == 0.35.3
- numpy >= 2.2.6
- pillow >= 11.3.0
- pydantic >= 2.12.0
- timm == 1.0.20
- ultralytics >= 8.3.208

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- **Danny Tram** - [GitLab](https://gitlab.com/dat.tram)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues and questions, please use the [GitLab Issues](https://gitlab.com/dat.tram/raiki-sdk/-/issues) page.

## 🗺️ Roadmap

- [x] Add Model Package Registry
- [x] Add Part Versioning
- [x] Add Model Versioning
- [ ] Add Classify Base
- [ ] Add Red Frame service
- [ ] Add unit tests
- [ ] Add CI/CD pipeline
- [ ] Add model performance benchmarking
- [ ] Add model package validation utilities

## 📊 Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

### Latest Release: v0.1.1 (2025-10-21)
- Added Model Package Registry system
- Added Part Versioning (P00, P01, P02, etc.)
- Added Model Versioning with automatic normalization
- Added model discovery and lazy loading
- See full changelog for details