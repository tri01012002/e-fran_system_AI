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

### Basic Installation (Private Access - Using SSH)

```bash
# Latest version
uv pip install git+ssh://git@gitlab.com/dat.tram/raiki-sdk.git

# Specific version
uv pip install git+ssh://git@gitlab.com/dat.tram/raiki-sdk.git@v0.1.0
```

#### Note
After installing the raiki-sdk, run the following command:

```bash
uv pip install "transformers>=4.56,<5" "peft==0.18.0"
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

### Supported Brands and Parts

The SDK supports counterfeit detection for multiple luxury brands with multiple parts per brand:

#### Simplified API

##### `get_auth_feature()` function provides a simplified interface for loading authentication models:

from raiki_sdk import get_auth_feature

```
# Simple one-liner to load any model package
model_package = get_auth_feature(
    category="rolex",    # Brand: "rolex", "bag-lv", "omg", etc.
    part="P01",          # Part identifier: "P01", "P02", etc.
    model_version="v1_3",  # Model version: "v1_3", "v1_4", etc.
    device="cuda"        # Device: "cuda", "mps", "cpu"
)
```

```
# Access detector and authenticator
detector = model_package.detector
authenticator = model_package.authenticator
```

```
# Use for detection
image = cv2.imread("product.jpg")
    yolo_result = detector.detect(image)
```

```
# Use for authentication
cropped_image = detector.crop(image, validation.position)
auth_result = authenticator.forward(cropped_image)

print(f"Real: {auth_result.is_authentic}, Confidence: {auth_result.confidence}")
```

**Key Features:**
- ✅ Unified API for all brands and parts
- ✅ Automatic detector and authenticator loading
- ✅ Built-in caching to avoid redundant model loading
- ✅ Support for multiple devices (CUDA, MPS, CPU)
- ✅ Simple one-liner usage for AI Engineers

### Using RedFrame Service

The SDK provides a RedFrame service for image quality assessment (IQA) and frame validation with two operation modes.

#### Basic Mode - Simple IQA Only

Basic mode uses the standard LIQE model from `pyiqa` library and performs simple image quality validation on the entire image.

```python
from raiki_sdk import get_redframe_service
from PIL import Image
import cv2

# Initialize service in basic mode
redframe_service = get_redframe_service(
    category="rolex",
    part="p01",
    advanced_mode=False,  # Basic mode
    device="cuda",  # or "mps", "cpu"
    threshold=0.1  # Quality threshold (0-1)
)

# Load image
image = cv2.imread("product.jpg")

# Simple IQA validation (no crop_method needed)
result = redframe_service.forward(image)

print(f"Image Quality Valid: {result.result}")
print(f"IQA Score: {result.iqa_result.iqa_score}")
print(f"Threshold: {result.iqa_result.iqa_threshold}")
```

**Basic Mode Features:**
- ✅ Uses standard LIQE via `pyiqa` library
- ✅ Validates entire image (no cropping required)
- ✅ No `crop_method` parameter needed
- ✅ No frame/position validation
- ✅ Fast and simple quality check

#### Advanced Mode - Full RedFrame + Custom IQA

Advanced mode uses a custom LIQE model with additional RedFrame validation including geometric checks and cropped region analysis.

```python
from raiki_sdk import get_redframe_service, get_auth_feature
import cv2

# Get detector for crop method
model_package = get_auth_feature("rolex", "p01", "v1_3", device="cuda")
detector = model_package.detector

# Initialize service in advanced mode
redframe_service = get_redframe_service(
    category="rolex",
    part="p01",
    advanced_mode=True,  # Advanced mode
    device="cuda",
    threshold=0.1
)

# Load image and detect
image = cv2.imread("product.jpg")
    yolo_result = detector.detect(image)
validation = detector.validate(yolo_result)

# Advanced validation with RedFrame checks (crop_method REQUIRED)
result = redframe_service.forward(
    image=image,
    frame="0.1,0.2,0.9,0.8",  # Normalized frame coordinates
    position=validation.position,  # [x1, y1, x2, y2] in pixels
    crop_method=detector.crop  # REQUIRED: Method to crop image
)

print(f"Validation Passed: {result.result}")
print(f"Error Type: {result.error_type}")

if result.iqa_result:
    print(f"IQA Score: {result.iqa_result.iqa_score}")
    print(f"Distortion Type: {result.iqa_result.distortion_type}")
```

**Advanced Mode Features:**
- ✅ Uses custom LIQE model with distortion detection
- ✅ Validates position within frame boundaries
- ✅ Performs XGBoost RedFrame classification
- ✅ Analyzes cropped region quality
- ✅ Provides detailed error types:
  - `out_of_frame_error`: Position outside frame
  - `wide_error`, `angle_error`, `close_error`: RedFrame model predictions
  - `blur_error`, `noise_error`, etc.: IQA distortion types

**Required Parameters for Advanced Mode:**
- `frame`: Normalized coordinates (x1,y1,x2,y2) as string
- `position`: Pixel coordinates [x1, y1, x2, y2] as list
- `crop_method`: Callable to crop image region (e.g., `detector.crop`)

#### Mode Comparison

| Feature | Basic Mode | Advanced Mode |
|---------|-----------|---------------|
| IQA Model | Standard LIQE (pyiqa) | Custom LIQE with distortion detection |
| Input Image | Full image | Cropped region |
| Frame Validation | ❌ No | ✅ Yes |
| RedFrame Check | ❌ No | ✅ Yes (XGBoost model) |
| Distortion Type | ❌ No | ✅ Yes |
| `crop_method` Required | ❌ No | ✅ Yes |
| Performance | Faster | More comprehensive |
| Use Case | Quick quality check | Full authentication pipeline |

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

#### `DetectorBase`
Abstract base class for model-based detection systems.

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
- `get_auth_feature()` - Returns authenticator instance (auth models)
- `get_classification_feature()` - Returns classifier instance (P00 models)

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
python -c "from raiki_sdk import AuthenticateBase, DetectorBase; print('✅ Success!')"

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
- [x] Add Classify Base
- [x] Add Red Frame service
- [x] Add model package validation utilities
- [ ] Add unit tests
- [ ] Add CI/CD pipeline
- [ ] Add model performance benchmarking

## 📊 Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

### Latest Release: v0.1.3 (2025-10-24)
- Expanded model packages for Louis Vuitton and Rolex brands
- Support for multiple parts per brand (6 LV parts, 8 Rolex parts)
- Simplified `get_auth_feature()` API for AI Engineers
- Centralized model package management
- See full changelog for details